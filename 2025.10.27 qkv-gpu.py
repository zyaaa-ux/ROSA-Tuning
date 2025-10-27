import os
import math
import json
import time
import atexit
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3DecoderLayer,
)

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import threading
import os
import atexit


ROSA_INJECT_MODE: str = ""
ROSA_NUM_ROUTES: int = 0
ROSA_QK_VOCAB_SIZE: int = 0
ROSA_V_VOCAB_SIZE:  int = 0

ROSA_TRITON_BLOCK_J: int = 0
ROSA_TRITON_NUM_WARPS: int = 0
ROSA_TRITON_NUM_STAGES: int = 0

ROSA_USE_SOFT_TAU: bool = False
ROSA_SOFT_TAU_VALUE: float = 0.0



MODEL_LOCAL_DIR = ""
DATASET_DIR     = ""
OUTPUT_DIR      = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FLASH_ATTN = True
BF16 = True if torch.cuda.is_available() else False

SEQ_LEN = 0
ATTN_WINDOW = 0
FIRST_GLOBAL_LAYERS = 0



LR_ROSA = 0.0
LR_BACKBONE = 0.0
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 0
NUM_EPOCHS = 0
PER_DEVICE_TRAIN_BSZ = 0
GRAD_ACCUM_STEPS = 0
LOGGING_STEPS = 0
EVAL_STEPS = 0
SEED = 0

SAVE_STATE_DICT_NAME = ""

GRADIENT_CHECKPOINTING = bool(LR_BACKBONE and LR_BACKBONE > 0.0)




import numpy as _np
import torch

import torch
import triton
import triton.language as tl
from torch import Tensor


@torch.no_grad()
def compute_run_ids_and_starts(z_btm: Tensor):
    """
    Input:
      z_btm: int32 [B,T,M]
    Output:
      run_id_bmt:      int32 [B,M,T]
      run_start_by_run int32 [B,M,T]
      is_run_start_bmt bool  [B,M,T]
    """
    assert z_btm.dtype == torch.int32 and z_btm.is_cuda
    B, T, M = z_btm.shape
    z_bmt = z_btm.permute(0, 2, 1).contiguous()

    is_new = torch.ones_like(z_bmt, dtype=torch.bool)
    is_new[:, :, 1:] = (z_bmt[:, :, 1:] != z_bmt[:, :, :-1])

    run_id = is_new.to(torch.int32).cumsum(dim=2) - 1

    N = B * M
    rid_nm = run_id.view(N, T)
    t_idx  = torch.arange(T, device=z_bmt.device, dtype=torch.int32).view(1, T).expand(N, T)
    large = torch.full((N, T), T, dtype=torch.int32, device=z_bmt.device)
    large.scatter_reduce_(1, rid_nm, t_idx, reduce='amin', include_self=True)
    large[large == T] = -1
    run_start_by_run = large.view(B, M, T)
    return run_id, run_start_by_run, is_new



@triton.jit
def _rosa_tau_rcf_kernel(
    q_ptr,
    k_ptr,
    ridk_ptr,
    ridq_ptr,
    rstart_run_ptr,
    dprev_ptr,
    dcurr_ptr,
    tau_out_ptr,
    rcf_out_ptr,
    is_q_start_ptr,
    B: tl.constexpr, M: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    stride_q_b, stride_q_m, stride_q_t,
    stride_k_b, stride_k_m, stride_k_t,
    stride_ridk_b, stride_ridk_m, stride_ridk_t,
    stride_ridq_b, stride_ridq_m, stride_ridq_t,
    stride_rsrt_b, stride_rsrt_m, stride_rsrt_t,
    stride_tau_b, stride_tau_m, stride_tau_t,
    stride_rcf_b, stride_rcf_m, stride_rcf_r, stride_rcf_k,
    stride_d_bm, stride_d_t,
    BLOCK_J: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid - b * M
    if b >= B:
        return

    q_bmt   = q_ptr    + b * stride_q_b    + m * stride_q_m
    k_bmt   = k_ptr    + b * stride_k_b    + m * stride_k_m
    ridk_bmt= ridk_ptr + b * stride_ridk_b + m * stride_ridk_m
    ridq_bmt= ridq_ptr + b * stride_ridq_b + m * stride_ridq_m
    rsrt_bmt= rstart_run_ptr + b * stride_rsrt_b + m * stride_rsrt_m
    tau_bmt = tau_out_ptr   + b * stride_tau_b   + m * stride_tau_m
    rcf_bmrk= rcf_out_ptr   + b * stride_rcf_b   + m * stride_rcf_m
    isqs_bmt= is_q_start_ptr+ b * (stride_q_b)   + m * (stride_q_m)

    dprev_bm = dprev_ptr + pid * stride_d_bm
    dcurr_bm = dcurr_ptr + pid * stride_d_bm


    j_idx_base = tl.arange(0, BLOCK_J)

    for i in range(0, T):
        qi = tl.load(q_bmt + i * stride_q_t, mask=True, other=0)
        rid_i = tl.load(ridk_bmt + i * stride_ridk_t, mask=True, other=0)

        best_score = tl.full((), -1, tl.int32)
        best_j     = tl.full((), -1, tl.int32)

        q_is_start = tl.load(isqs_bmt + i * stride_q_t, mask=True, other=0, eviction_policy='evict_last').to(tl.int1)
        best_s_a = tl.full((K,), -1, tl.int32)
        best_j_a = tl.full((K,), -1, tl.int32)

        for j0 in range(0, T, BLOCK_J):
            j = j0 + j_idx_base
            m_j  = j < T

            kj    = tl.load(k_bmt + j * stride_k_t,    mask=m_j, other=0)
            rid_j = tl.load(ridk_bmt + j * stride_ridk_t, mask=m_j, other=0)

            jm1 = j - 1
            m_jm1 = m_j & (j >= 1)
            dleft = tl.load(dprev_bm + jm1 * stride_d_t, mask=m_jm1, other=0)

            eq = (kj == qi)

            valid = m_j & (j <= i) & (rid_j < rid_i) & eq

            d_ij = tl.where(valid, dleft + 1, -1)

            smax = tl.max(d_ij, axis=0)
            eqmx = (d_ij == smax) & (smax >= 0)
            j_cand = tl.where(eqmx, j, -1)
            jmax   = tl.max(j_cand, axis=0)

            if smax > best_score or (smax == best_score and jmax > best_j):
                best_score = smax
                best_j     = jmax

            if q_is_start:
                for a in tl.static_range(K):
                    eq_a = (kj == a)
                    valid_a = m_j & (j < i) & (rid_j < rid_i) & eq_a
                    d_ij_a = tl.where(valid_a, dleft + 1, -1)

                    smax_a = tl.max(d_ij_a, axis=0)
                    eqmx_a = (d_ij_a == smax_a) & (smax_a >= 0)
                    j_cand_a = tl.where(eqmx_a, j, -1)
                    jmax_a   = tl.max(j_cand_a, axis=0)

                    if smax_a > best_s_a[a] or (smax_a == best_s_a[a] and jmax_a > best_j_a[a]):
                        best_s_a[a] = smax_a
                        best_j_a[a] = jmax_a

        tl.store(tau_bmt + i * stride_tau_t, best_j)

        for jj in range(0, T):
            d_val = tl.load(dcurr_bm + jj * stride_d_t, mask=True, other=0)
            if jj == best_j and best_score >= 0:
                d_val = best_score
            tl.store(dcurr_bm + jj * stride_d_t, d_val, mask=True)

        if q_is_start:
            rid_qi = tl.load(ridq_bmt + i * stride_ridq_t, mask=True, other=0)
            for a in tl.static_range(K):
                jcf_a = best_j_a[a]
                if jcf_a >= 0:
                    rid_jcf = tl.load(ridk_bmt + jcf_a * stride_ridk_t, mask=True, other=0)
                    t0 = tl.load(rsrt_bmt + rid_jcf * stride_rsrt_t, mask=True, other=-1)
                else:
                    t0 = -1
                tl.store(rcf_bmrk + rid_qi * stride_rcf_r + a * stride_rcf_k, t0)

        for jj in range(0, T):
            old_val = tl.load(dcurr_bm + jj * stride_d_t, mask=True, other=0)
            tl.store(dprev_bm + jj * stride_d_t, old_val, mask=True)


def compute_tau_and_rcf_triton(
    q_bmt: Tensor,
    k_bmt: Tensor,
    ridk_bmt: Tensor,
    ridq_bmt: Tensor,
    rstart_run_k_bmt: Tensor,
    is_q_start_bmt: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Triton wrapper
    Input:
      q_bmt, k_bmt, ridk_bmt, ridq_bmt, rstart_run_k_bmt: [B,M,T] int32
      is_q_start_bmt: [B,M,T] bool
    Output:
      tau_time_bmt:   [B,M,T] int32
      r_cf_run_bmrk:  [B,M,T,K] int32
    """
    device = q_bmt.device
    assert all(
        x.dtype == torch.int32 and x.device == device
        for x in [q_bmt, k_bmt, ridk_bmt, ridq_bmt, rstart_run_k_bmt]
    ), "All tensors must be int32 on the same device"
    assert is_q_start_bmt.dtype == torch.bool and is_q_start_bmt.device == device

    B, M, T = q_bmt.shape
    K = globals().get("ROSA_QK_VOCAB_SIZE", 0)

    tau_time_bmt = torch.zeros((B, M, T), dtype=torch.int32, device=device)
    r_cf_run_bmrk= torch.full((B, M, T, K), -1, dtype=torch.int32, device=device)

    dprev = torch.zeros((B * M, T), dtype=torch.int32, device=device)
    dcurr = torch.zeros((B * M, T), dtype=torch.int32, device=device)

    block_j = globals().get("ROSA_TRITON_BLOCK_J", 0)
    nwarps  = globals().get("ROSA_TRITON_NUM_WARPS", 0)
    nstages = globals().get("ROSA_TRITON_NUM_STAGES", 0)

    grid = (B * M,)
    _rosa_tau_rcf_kernel[grid](
        q_bmt, k_bmt, ridk_bmt, ridq_bmt, rstart_run_k_bmt,
        dprev, dcurr,
        tau_time_bmt, r_cf_run_bmrk, is_q_start_bmt,
        B, M, T, K,
        q_bmt.stride(0), q_bmt.stride(1), q_bmt.stride(2),
        k_bmt.stride(0), k_bmt.stride(1), k_bmt.stride(2),
        ridk_bmt.stride(0), ridk_bmt.stride(1), ridk_bmt.stride(2),
        ridq_bmt.stride(0), ridq_bmt.stride(1), ridq_bmt.stride(2),
        rstart_run_k_bmt.stride(0), rstart_run_k_bmt.stride(1), rstart_run_k_bmt.stride(2),
        tau_time_bmt.stride(0), tau_time_bmt.stride(1), tau_time_bmt.stride(2),
        r_cf_run_bmrk.stride(0), r_cf_run_bmrk.stride(1), r_cf_run_bmrk.stride(2), r_cf_run_bmrk.stride(3),
        dprev.stride(0), dprev.stride(1),
        BLOCK_J=block_j,
        num_warps=nwarps,
        num_stages=nstages,
    )
    return tau_time_bmt, r_cf_run_bmrk


def rosa_hard_match_route(
    qk_logits_btmd: Tensor,
    v_logits_btmd: Tensor,
    v_emb_mk_d: Tensor
) -> Tensor:
    """
    ROSA hard routing with Triton-based tau/rcf computation.
    Input:
      qk_logits_btmd: [B,T,M,Dqk]
      v_logits_btmd:  [B,T,M,Dv]
      v_emb_mk_d:     [M, K_v, d_out]
    Output:
      out: [B,T,M,d_out]
    """
    device = qk_logits_btmd.device
    B, T, M, Dqk = qk_logits_btmd.shape
    _, _, _, Dv  = v_logits_btmd.shape
    K_v = v_emb_mk_d.size(1)
    d_out= v_emb_mk_d.size(2)

    q_btm = torch.argmax(qk_logits_btmd, dim=-1).to(torch.int32)
    k_btm = q_btm
    v_btm = torch.argmax(v_logits_btmd, dim=-1).to(torch.int32)

    q_bmt = q_btm.permute(0, 2, 1).contiguous()
    k_bmt = k_btm.permute(0, 2, 1).contiguous()
    v_bmt = v_btm.permute(0, 2, 1).contiguous()

    ridk_bmt, rstart_run_k_bmt, is_k_start_bmt = compute_run_ids_and_starts(k_btm)
    ridq_bmt, rstart_run_q_bmt, is_q_start_bmt = compute_run_ids_and_starts(q_btm)

    tau_time_bmt, r_cf_run_bmrk = compute_tau_and_rcf_triton(
        q_bmt, k_bmt, ridk_bmt, ridq_bmt, rstart_run_k_bmt, is_q_start_bmt
    )

    tau_time_btm = tau_time_bmt.permute(0, 2, 1).contiguous()
    r_cf_run_bmtk= r_cf_run_bmrk.permute(0, 2, 1, 3).contiguous()

    is_q_start_btm = is_q_start_bmt.permute(0, 2, 1).contiguous()
    ridq_btm       = ridq_bmt.permute(0, 2, 1).contiguous()

    out_btmd = torch.zeros(B, T, M, d_out, device=device, dtype=v_emb_mk_d.dtype)

    for b in range(B):
        for m in range(M):
            vcur = v_bmt[b, m, :]
            emb  = v_emb_mk_d[m]

            ev_td = emb[vcur]

            tau  = tau_time_bmt[b, m, :]
            r_cf = r_cf_run_bmrk[b, m, :, :]

            is_qs_t = is_q_start_btm[b, :, m]
            ridq_t  = ridq_btm[b, :, m]

            for t in range(T):
                if is_qs_t[t]:
                    rq = ridq_t[t].item()
                    row_k = r_cf[rq, :]
                    vec_sum = torch.zeros(d_out, device=device, dtype=ev_td.dtype)
                    cnt = 0
                    for kk in range(K_v):
                        tj = row_k[kk].item()
                        if tj >= 0:
                            vec_sum += ev_td[tj]
                            cnt += 1
                    if cnt > 0:
                        out_btmd[b, t, m, :] = vec_sum / cnt
                    else:
                        out_btmd[b, t, m, :] = ev_td[t]
                else:
                    j = tau[t].item()
                    if j >= 0:
                        out_btmd[b, t, m, :] = ev_td[j]
                    else:
                        out_btmd[b, t, m, :] = 0.0

    return out_btmd


class RosaLayer(nn.Module):
    """
    ROSA routing layer with multiple routes
    """
    def __init__(
        self,
        d_model: int,
        num_routes: int = 0,
        qk_vocab_size: int = 0,
        v_vocab_size: int = 0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routes = num_routes
        self.qk_vocab_size = qk_vocab_size
        self.v_vocab_size  = v_vocab_size

        self.rosa_wlm_q_list = nn.ModuleList([
            nn.Linear(d_model, qk_vocab_size, bias=False, device=device, dtype=dtype)
            for _ in range(num_routes)
        ])
        self.rosa_wlm_k_list = nn.ModuleList([
            nn.Linear(d_model, qk_vocab_size, bias=False, device=device, dtype=dtype)
            for _ in range(num_routes)
        ])
        self.rosa_wlm_v_list = nn.ModuleList([
            nn.Linear(d_model, v_vocab_size, bias=False, device=device, dtype=dtype)
            for _ in range(num_routes)
        ])

        self.rosa_v_emb_list = nn.ModuleList([
            nn.Embedding(v_vocab_size, d_model, device=device, dtype=dtype)
            for _ in range(num_routes)
        ])

        self.rosa_alpha = nn.Parameter(
            torch.zeros(num_routes, dtype=dtype if dtype else torch.float32, device=device)
        )

        self._init_weights()

    def _init_weights(self):
        for head in self.rosa_wlm_q_list:
            nn.init.normal_(head.weight, mean=0.0, std=0.0)
        for head in self.rosa_wlm_k_list:
            nn.init.normal_(head.weight, mean=0.0, std=0.0)
        for head in self.rosa_wlm_v_list:
            nn.init.normal_(head.weight, mean=0.0, std=0.0)

        for emb in self.rosa_v_emb_list:
            nn.init.normal_(emb.weight, mean=0.0, std=0.0)
            with torch.no_grad():
                emb.weight[0].zero_()

    def forward(self, x_bt_d: Tensor) -> Tensor:
        """
        Input:  x_bt_d [B, T, d_model]
        Output: out    [B, T, d_model]
        """
        B, T, D = x_bt_d.shape
        M = self.num_routes

        qk_logits_list = []
        for i in range(M):
            lq = self.rosa_wlm_q_list[i](x_bt_d)
            qk_logits_list.append(lq)
        qk_logits_btmd = torch.stack(qk_logits_list, dim=2)

        v_logits_list = []
        for i in range(M):
            lv = self.rosa_wlm_v_list[i](x_bt_d)
            v_logits_list.append(lv)
        v_logits_btmd = torch.stack(v_logits_list, dim=2)

        v_emb_mk_d = torch.stack([emb.weight for emb in self.rosa_v_emb_list], dim=0)

        rosa_out_btmd = rosa_hard_match_route(qk_logits_btmd, v_logits_btmd, v_emb_mk_d)

        weights = torch.softmax(self.rosa_alpha, dim=0)
        final_out = torch.einsum('btmd,m->btd', rosa_out_btmd, weights)

        return final_out


def inject_rosa_into_qwen3_layer(
    layer: Qwen3DecoderLayer,
    num_routes: int,
    qk_vocab_size: int,
    v_vocab_size: int,
    mode: str = "",
):
    """
    Inject ROSA routing into a Qwen3 decoder layer
    """
    hidden_size = layer.self_attn.hidden_size
    device = next(layer.parameters()).device
    dtype  = next(layer.parameters()).dtype

    rosa_module = RosaLayer(
        d_model=hidden_size,
        num_routes=num_routes,
        qk_vocab_size=qk_vocab_size,
        v_vocab_size=v_vocab_size,
        device=device,
        dtype=dtype,
    )

    layer.rosa_qk_vocab_size = qk_vocab_size
    layer.rosa_v_vocab_size  = v_vocab_size
    layer.rosa_wlm_q_list    = rosa_module.rosa_wlm_q_list
    layer.rosa_wlm_k_list    = rosa_module.rosa_wlm_k_list
    layer.rosa_wlm_v_list    = rosa_module.rosa_wlm_v_list
    layer.rosa_v_emb_list    = rosa_module.rosa_v_emb_list
    layer.rosa_alpha         = rosa_module.rosa_alpha

    old_forward = layer.forward

    def new_forward(
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ):
        if mode == "":
            x_in = hidden_states
        else:
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
            x_in = hidden_states

        rosa_out = rosa_module(x_in)
        if mode == "":
            hidden_states = hidden_states + rosa_out
            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)
            hidden_states, self_attn_weights, present_key_value = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = residual + hidden_states
        else:
            hidden_states = hidden_states + rosa_out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs

    layer.forward = new_forward


@dataclass
class FixedLenLMCollator:
    pad_token_id: int
    seq_len: int

    def __call__(self, features):
        max_len = self.seq_len
        input_ids_list = []
        labels_list = []

        for ex in features:
            ids = ex["input_ids"]
            if len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len

            ids = ids[:max_len]
            labels = ids.copy()

            input_ids_list.append(ids)
            labels_list.append(labels)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "labels":    torch.tensor(labels_list,    dtype=torch.long),
        }


def build_model_and_tokenizer():
    """
    Build model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_LOCAL_DIR, trust_remote_code=True)
    if USE_FLASH_ATTN:
        config._attn_implementation = ""

    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        config=config,
        torch_dtype=torch.bfloat16 if BF16 else torch.float32,
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable() if GRADIENT_CHECKPOINTING else None

    num_layers = model.config.num_hidden_layers
    inject_mode = globals().get("ROSA_INJECT_MODE", "")
    num_routes  = globals().get("ROSA_NUM_ROUTES", 0)
    qk_vocab    = globals().get("ROSA_QK_VOCAB_SIZE", 0)
    v_vocab     = globals().get("ROSA_V_VOCAB_SIZE", 0)

    for i, layer in enumerate(model.model.layers):
        if i < FIRST_GLOBAL_LAYERS:
            continue
        inject_rosa_into_qwen3_layer(
            layer,
            num_routes=num_routes,
            qk_vocab_size=qk_vocab,
            v_vocab_size=v_vocab,
            mode=inject_mode,
        )

    if torch.cuda.is_available():
        model = model.to("cuda")

    return model, tokenizer


def save_rosa_only(model: Qwen3ForCausalLM, out_dir: str):
    """
    Save only ROSA parameters
    """
    os.makedirs(out_dir, exist_ok=True)
    state = {}

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "rosa_wlm_q_list"):
            for m, head in enumerate(layer.rosa_wlm_q_list):
                state[f"model.layers.{i}.rosa_wlm_q_list.{m}.weight"] = head.weight.detach().cpu()
        if hasattr(layer, "rosa_wlm_k_list"):
            for m, head in enumerate(layer.rosa_wlm_k_list):
                state[f"model.layers.{i}.rosa_wlm_k_list.{m}.weight"] = head.weight.detach().cpu()
        if hasattr(layer, "rosa_wlm_v_list"):
            for m, head in enumerate(layer.rosa_wlm_v_list):
                state[f"model.layers.{i}.rosa_wlm_v_list.{m}.weight"] = head.weight.detach().cpu()

        if hasattr(layer, "rosa_v_emb_list"):
            for m, emb in enumerate(layer.rosa_v_emb_list):
                state[f"model.layers.{i}.rosa_v_emb_list.{m}.weight"] = emb.weight.detach().cpu()

        if hasattr(layer, "rosa_alpha"):
            state[f"model.layers.{i}.rosa_alpha"] = layer.rosa_alpha.detach().cpu()

    path = os.path.join(out_dir, SAVE_STATE_DICT_NAME)
    torch.save(state, path)
    print(f"[save] saved ROSA(Q/K/V)-only params to: {path}")

def save_results_and_meta(model: Qwen3ForCausalLM, metrics: dict, out_dir: str):
    """
    Save ROSA parameters and metadata
    """
    os.makedirs(out_dir, exist_ok=True)

    save_rosa_only(model, out_dir)

    num_layers = model.config.num_hidden_layers
    k_per_layer_qk = {
        str(i): int(getattr(model.model.layers[i], "rosa_qk_vocab_size",
                            getattr(model.model.layers[i], "rosa_vocab_size", 0)))
        for i in range(num_layers)
    }
    k_per_layer_v = {
        str(i): int(getattr(model.model.layers[i], "rosa_v_vocab_size",
                            getattr(model.model.layers[i], "rosa_vocab_size", 0)))
        for i in range(num_layers)
    }

    meta = {
        "model_local_dir": MODEL_LOCAL_DIR,
        "dataset_dir": DATASET_DIR,
        "seq_len": SEQ_LEN,
        "attn_window": ATTN_WINDOW,
        "first_global_layers": FIRST_GLOBAL_LAYERS,
        "rosa": {
            "num_routes": int(globals().get("ROSA_NUM_ROUTES", 0)),
            "qk_vocab_size": int(globals().get("ROSA_QK_VOCAB_SIZE", 0)),
            "v_vocab_size": int(globals().get("ROSA_V_VOCAB_SIZE", 0)),
            "lcg_topk": int(globals().get("LCG_TOPK", 0)),
            "pos_subsample": float(globals().get("LCG_POS_SUBSAMPLE", 0.0)),
            "lr_rosa": float(globals().get("LR_ROSA", 0.0)),
            "lr_backbone": float(globals().get("LR_BACKBONE", 0.0)),
            "k_per_layer_qk": k_per_layer_qk,
            "k_per_layer_v": k_per_layer_v,
        },
        "metrics": metrics,
        "time": time.asctime(),
    }

    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[eval] metrics:", metrics)
    print(f"[done] meta saved at {os.path.join(out_dir, 'run_meta.json')}")



def build_optimizer_params(model):
    """
    Build optimizer parameter groups
    """
    wlm_q, wlm_k, wlm_v, v_emb, gate, backbone = [], [], [], [], [], []

    for n, p in model.named_parameters():
        if "rosa_wlm_q_list" in n:
            wlm_q.append(p)
        elif "rosa_wlm_k_list" in n:
            wlm_k.append(p)
        elif "rosa_wlm_v_list" in n:
            wlm_v.append(p)
        elif "rosa_v_emb_list" in n:
            v_emb.append(p)
        elif "rosa_alpha" in n:
            gate.append(p)
        else:
            backbone.append(p)

    param_groups = []
    if wlm_q: param_groups.append({"params": wlm_q, "lr": LR_ROSA, "weight_decay": WEIGHT_DECAY})
    if wlm_k: param_groups.append({"params": wlm_k, "lr": LR_ROSA, "weight_decay": WEIGHT_DECAY})
    if wlm_v: param_groups.append({"params": wlm_v, "lr": LR_ROSA, "weight_decay": WEIGHT_DECAY})
    if v_emb: param_groups.append({"params": v_emb, "lr": LR_ROSA, "weight_decay": 0.0})
    if gate:  param_groups.append({"params": gate,  "lr": LR_ROSA, "weight_decay": 0.0})

    if LR_BACKBONE and LR_BACKBONE > 0.0:
        no_decay, has_decay = [], []
        for n, p in model.named_parameters():
            if "rosa_" in n:
                continue
            if any(k in n.lower() for k in ["bias", "norm", "layernorm", "ln"]):
                no_decay.append(p)
            else:
                has_decay.append(p)
        if has_decay:
            param_groups.append({"params": has_decay, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": LR_BACKBONE, "weight_decay": 0.0})
    else:
        for p in backbone:
            p.requires_grad_(False)

    return param_groups




from transformers.trainer_callback import TrainerCallback

def is_main_process() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0


class RosaZeroRowCallback(TrainerCallback):
    def on_init_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            with torch.no_grad():
                for layer in model.model.layers:
                    if hasattr(layer, "rosa_v_emb_list"):
                        for emb in layer.rosa_v_emb_list:
                            emb.weight.data[0].zero_()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            with torch.no_grad():
                for layer in model.model.layers:
                    if hasattr(layer, "rosa_v_emb_list"):
                        for emb in layer.rosa_v_emb_list:
                            emb.weight.data[0].zero_()
        return control




def main():
    set_seed(SEED)

    raw = load_from_disk(DATASET_DIR)
    train_ds = raw["train"]
    test_ds = raw.get("test", raw["validation"] if "validation" in raw else None)
    assert test_ds is not None, "Test or validation split required"

    model, tokenizer = build_model_and_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data_collator = FixedLenLMCollator(pad_token_id=pad_id, seq_len=SEQ_LEN)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        per_device_eval_batch_size=0,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR_ROSA,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="",
        report_to="",
        fp16=(not BF16) and torch.cuda.is_available(),
        bf16=BF16,
        dataloader_num_workers=0,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        optim="",
    )

    optimizer_params = build_optimizer_params(model)

    class _Trainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(optimizer_params, betas=(0.0, 0.0), eps=0.0)
            return self.optimizer

    trainer = _Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[RosaZeroRowCallback()],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")

    trainer.train()
    metrics = trainer.evaluate()


    if is_main_process():
        save_results_and_meta(model, metrics, OUTPUT_DIR)


if __name__ == "__main__":
    main()
