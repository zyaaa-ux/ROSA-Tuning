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

import os as _os
_os.environ.setdefault("ROSA_USE_NUMBA", "")
_os.environ.setdefault("ROSA_NUMBA_PARALLEL", "")
_os.environ.setdefault("LCG_ENABLE", "")

_os.environ["ROSA_INJECT_MODE"] = ""

_USE_NUMBA = _os.environ.get("ROSA_USE_NUMBA", "").lower() not in ("0", "false")
_PARALLEL  = _os.environ.get("ROSA_NUMBA_PARALLEL", "").lower() not in ("0", "false")

try:
    import numba as _nb
    _NUMBA_OK = _USE_NUMBA
except Exception:
    _NUMBA_OK = False


MODEL_LOCAL_DIR = ""
DATASET_DIR     = ""
OUTPUT_DIR      = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FLASH_ATTN = None
BF16 = None

SEQ_LEN = None
ATTN_WINDOW = None
FIRST_GLOBAL_LAYERS = None

ROSA_NUM_ROUTES: int = None
LCG_POS_SUBSAMPLE: float = None

ROSA_QK_VOCAB_SIZE: int = int(os.environ.get("ROSA_QK_VOCAB_SIZE", 0))
ROSA_V_VOCAB_SIZE:  int = int(os.environ.get("ROSA_V_VOCAB_SIZE", 0))

_PER_RANK_CPUS = None

os.environ.setdefault("NUMBA_NUM_THREADS", str(_PER_RANK_CPUS))

LR_ROSA = None
LR_BACKBONE = None
WEIGHT_DECAY = None
WARMUP_STEPS = None
NUM_EPOCHS = None
PER_DEVICE_TRAIN_BSZ = None
GRAD_ACCUM_STEPS = None
LOGGING_STEPS = None
EVAL_STEPS = None
SEED = None

SAVE_STATE_DICT_NAME = ""

GRADIENT_CHECKPOINTING = bool(LR_BACKBONE and LR_BACKBONE > 0.0)


def _env_int(k, default):
    try:
        return int(os.environ.get(k, default))
    except Exception:
        return default


try:
    import torch.cuda.nvtx as nvtx
except Exception:
    class _DummyNVTX:
        def range_push(self, *a, **k): pass
        def range_pop(self): pass
    nvtx = _DummyNVTX()

from concurrent.futures import ThreadPoolExecutor

_ROSA_THREAD_POOL = None
_ROSA_THREAD_POOL_PID = None

class _ImmediateFuture:
    __slots__ = ("_value",)
    def __init__(self, value):
        self._value = value
    def result(self, timeout=None):
        return self._value

def _get_rosa_thread_pool() -> ThreadPoolExecutor | None:
    global _ROSA_THREAD_POOL, _ROSA_THREAD_POOL_PID
    pid = os.getpid()
    n_workers = max(0, _env_int("ROSA_THREAD_WORKERS", 0))
    if n_workers == 0:
        return None
    if (_ROSA_THREAD_POOL is None) or (_ROSA_THREAD_POOL_PID != pid):
        if _ROSA_THREAD_POOL is not None:
            try:
                _ROSA_THREAD_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _ROSA_THREAD_POOL = ThreadPoolExecutor(max_workers=n_workers)
        _ROSA_THREAD_POOL_PID = pid
    return _ROSA_THREAD_POOL


def _wait_event(ev: "torch.cuda.Event"):
    try:
        ev.synchronize()
    except Exception:
        pass


torch.set_num_threads(_PER_RANK_CPUS)

import numpy as _np
import torch


class _SAMFoldedCPU:
    __slots__ = ("next", "link", "length", "e", "last", "size", "K", "c", "last_sym")

    def __init__(self, max_states: int, K: int):
        self.K = int(K)
        S = int(max_states)
        self.next   = _np.full((S, self.K), -1, dtype=_np.int32)
        self.link   = _np.full((S,),       -1, dtype=_np.int32)
        self.length = _np.zeros((S,),          dtype=_np.int32)
        self.e      = _np.full((S,),       -1, dtype=_np.int32)
        self.last   = 0
        self.size   = 1
        self.c: List[int] = []
        self.last_sym: Optional[int] = None

    def _new_state(self, L: int) -> int:
        s = self.size
        self.size += 1
        self.length[s] = L
        self.link[s] = -1
        self.e[s] = -1
        self.next[s, :].fill(-1)
        return s

    def match_next(self, x: int) -> int:
        p = self.last
        nxt = self.next
        link = self.link
        while p != -1 and nxt[p, x] == -1:
            p = link[p]
        return -1 if p == -1 else int(nxt[p, x])

    def nextdiff_from_state(self, q: int) -> int:
        if q == -1:
            return -1
        rpos = int(self.e[q])
        nxt = rpos + 1
        if 0 <= rpos and nxt < len(self.c):
            return int(self.c[nxt])
        return -1

    def extend_run(self, x: int):
        self.c.append(x)
        pos = len(self.c) - 1

        last = self.last
        nxt = self.next
        link = self.link
        length = self.length
        e = self.e

        p = last
        while p != -1 and nxt[p, x] == -1:
            nxt[p, x] = self.size
            e[p] = pos
            p = link[p]

        if p == -1:
            cur = self._new_state(length[last] + 1)
            link[cur] = 0
            e[cur] = pos
        else:
            q = nxt[p, x]
            if length[q] == length[p] + 1:
                cur = q
            else:
                clone = self._new_state(length[p] + 1)
                nxt[clone, :] = nxt[q, :]
                link[clone] = link[q]
                e[clone] = e[q]

                link[q] = clone
                cur = clone

                pp = p
                while pp != -1 and nxt[pp, x] == q:
                    nxt[pp, x] = clone
                    e[pp] = pos
                    pp = link[pp]

        self.last = cur

    def extend(self, x: int):
        if self.last_sym is None or self.last_sym != x:
            self.extend_run(x)
            self.last_sym = x


if _NUMBA_OK:
    from numba import jit
    from numba.typed import Dict as NumbaDict

    @jit(nopython=True, nogil=True, fastmath=True, cache=True)
    def _run_length_encode_numba(arr):
        n = len(arr)
        if n == 0:
            return _np.empty(0, dtype=_np.int64), _np.empty(0, dtype=_np.int64)
        
        symbols_list = []
        lengths_list = []
        current_symbol = arr[0]
        current_length = 1
        
        for i in range(1, n):
            if arr[i] == current_symbol:
                current_length += 1
            else:
                symbols_list.append(current_symbol)
                lengths_list.append(current_length)
                current_symbol = arr[i]
                current_length = 1
        
        symbols_list.append(current_symbol)
        lengths_list.append(current_length)
        
        symbols = _np.array(symbols_list, dtype=_np.int64)
        lengths = _np.array(lengths_list, dtype=_np.int64)
        
        return symbols, lengths

    _run_length_encode = _run_length_encode_numba

    @jit(nopython=True, nogil=True, fastmath=True, cache=True)
    def _sam_extend_numba(
        arr,
        nxt,
        link,
        length,
        e,
        last_idx,
        size_idx,
    ):
        n = len(arr)
        if n == 0:
            return last_idx, size_idx

        K = nxt.shape[1]

        for i in range(n):
            x = arr[i]
            last = last_idx[0]
            
            p = last
            while p != -1:
                if nxt[p, x] != -1:
                    break
                nxt[p, x] = size_idx[0]
                e[p] = i
                p = link[p]

            if p == -1:
                cur = size_idx[0]
                size_idx[0] += 1
                length[cur] = length[last] + 1
                link[cur] = 0
                e[cur] = i
            else:
                q = nxt[p, x]
                if length[q] == length[p] + 1:
                    cur = q
                else:
                    clone = size_idx[0]
                    size_idx[0] += 1
                    length[clone] = length[p] + 1
                    
                    for k in range(K):
                        nxt[clone, k] = nxt[q, k]
                    link[clone] = link[q]
                    e[clone] = e[q]

                    link[q] = clone
                    cur = clone

                    pp = p
                    while pp != -1:
                        if nxt[pp, x] != q:
                            break
                        nxt[pp, x] = clone
                        e[pp] = i
                        pp = link[pp]

            last_idx[0] = cur

        return last_idx, size_idx

    _sam_extend = _sam_extend_numba

    @jit(nopython=True, nogil=True, fastmath=True, parallel=_PARALLEL, cache=True)
    def _batch_build_sam_numba(
        batch_arr,
        max_len,
        K,
        max_states,
    ):
        B = len(batch_arr)
        
        nxt_batch = _np.full((B, max_states, K), -1, dtype=_np.int32)
        link_batch = _np.full((B, max_states), -1, dtype=_np.int32)
        length_batch = _np.zeros((B, max_states), dtype=_np.int32)
        e_batch = _np.full((B, max_states), -1, dtype=_np.int32)
        last_batch = _np.zeros(B, dtype=_np.int32)
        size_batch = _np.ones(B, dtype=_np.int32)

        for b in range(B):
            arr = batch_arr[b]
            symbols, lengths = _run_length_encode_numba(arr)
            
            last_idx = _np.zeros(1, dtype=_np.int32)
            size_idx = _np.ones(1, dtype=_np.int32)
            
            _sam_extend_numba(
                symbols,
                nxt_batch[b],
                link_batch[b],
                length_batch[b],
                e_batch[b],
                last_idx,
                size_idx,
            )
            
            last_batch[b] = last_idx[0]
            size_batch[b] = size_idx[0]

        return nxt_batch, link_batch, length_batch, e_batch, last_batch, size_batch

    _batch_build_sam = _batch_build_sam_numba

else:
    def _run_length_encode(arr):
        if len(arr) == 0:
            return _np.empty(0, dtype=_np.int64), _np.empty(0, dtype=_np.int64)
        
        symbols_list = []
        lengths_list = []
        current_symbol = arr[0]
        current_length = 1
        
        for i in range(1, len(arr)):
            if arr[i] == current_symbol:
                current_length += 1
            else:
                symbols_list.append(current_symbol)
                lengths_list.append(current_length)
                current_symbol = arr[i]
                current_length = 1
        
        symbols_list.append(current_symbol)
        lengths_list.append(current_length)
        
        symbols = _np.array(symbols_list, dtype=_np.int64)
        lengths = _np.array(lengths_list, dtype=_np.int64)
        
        return symbols, lengths

    def _sam_extend(arr, nxt, link, length, e, last_idx, size_idx):
        n = len(arr)
        if n == 0:
            return last_idx, size_idx

        K = nxt.shape[1]

        for i in range(n):
            x = arr[i]
            last = last_idx[0]
            
            p = last
            while p != -1:
                if nxt[p, x] != -1:
                    break
                nxt[p, x] = size_idx[0]
                e[p] = i
                p = link[p]

            if p == -1:
                cur = size_idx[0]
                size_idx[0] += 1
                length[cur] = length[last] + 1
                link[cur] = 0
                e[cur] = i
            else:
                q = nxt[p, x]
                if length[q] == length[p] + 1:
                    cur = q
                else:
                    clone = size_idx[0]
                    size_idx[0] += 1
                    length[clone] = length[p] + 1
                    
                    for k in range(K):
                        nxt[clone, k] = nxt[q, k]
                    link[clone] = link[q]
                    e[clone] = e[q]

                    link[q] = clone
                    cur = clone

                    pp = p
                    while pp != -1:
                        if nxt[pp, x] != q:
                            break
                        nxt[pp, x] = clone
                        e[pp] = i
                        pp = link[pp]

            last_idx[0] = cur

        return last_idx, size_idx

    def _batch_build_sam(batch_arr, max_len, K, max_states):
        B = len(batch_arr)
        
        nxt_batch = _np.full((B, max_states, K), -1, dtype=_np.int32)
        link_batch = _np.full((B, max_states), -1, dtype=_np.int32)
        length_batch = _np.zeros((B, max_states), dtype=_np.int32)
        e_batch = _np.full((B, max_states), -1, dtype=_np.int32)
        last_batch = _np.zeros(B, dtype=_np.int32)
        size_batch = _np.ones(B, dtype=_np.int32)

        for b in range(B):
            arr = batch_arr[b]
            symbols, lengths = _run_length_encode(arr)
            
            last_idx = _np.zeros(1, dtype=_np.int32)
            size_idx = _np.ones(1, dtype=_np.int32)
            
            _sam_extend(
                symbols,
                nxt_batch[b],
                link_batch[b],
                length_batch[b],
                e_batch[b],
                last_idx,
                size_idx,
            )
            
            last_batch[b] = last_idx[0]
            size_batch[b] = size_idx[0]

        return nxt_batch, link_batch, length_batch, e_batch, last_batch, size_batch


@dataclass
class LCGRosaConfig:
    seq_len: int = None
    attn_window: int = None
    num_routes: int = None
    qk_vocab_size: int = None
    v_vocab_size: int = None
    lcg_topk: int = None
    pos_subsample: float = None
    first_global_layers: int = None
    use_flash_attn: bool = None


def _lcg_build_position_sets_per_route_cpu(
    keys_list: List[torch.Tensor],
    nxt_batch: np.ndarray,
    link_batch: np.ndarray,
    length_batch: np.ndarray,
    e_batch: np.ndarray,
    last_batch: np.ndarray,
    size_batch: np.ndarray,
    c_batch: List[np.ndarray],
    K_qk: int,
    topk_per_route: int,
    pos_subsample: float = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    B, T = keys_list[0].shape[0], keys_list[0].shape[1]
    M = len(keys_list)
    device = keys_list[0].device

    indices_list = [torch.zeros((B, T, topk_per_route), dtype=torch.int32, device=device) for _ in range(M)]
    values_list  = [torch.zeros((B, T, topk_per_route), dtype=torch.int32, device=device) for _ in range(M)]

    def process_batch_route(b_idx, route_idx):
        keys_br = keys_list[route_idx][b_idx].cpu().numpy()
        nxt = nxt_batch[b_idx]
        link = link_batch[b_idx]
        length = length_batch[b_idx]
        e_arr = e_batch[b_idx]
        last = last_batch[b_idx]
        size = size_batch[b_idx]
        c = c_batch[b_idx]

        S_max = nxt.shape[0]
        T_seq = len(keys_br)

        if pos_subsample is not None and 0.0 < pos_subsample < 1.0:
            pass

        idx_out = _np.zeros((T_seq, topk_per_route), dtype=_np.int32)
        val_out = _np.zeros((T_seq, topk_per_route), dtype=_np.int32)

        for t in range(T_seq):
            x = keys_br[t]
            
            p = last
            while p != -1 and nxt[p, x] == -1:
                p = link[p]
            q = -1 if p == -1 else nxt[p, x]

            cands = []
            if q != -1:
                rpos = e_arr[q]
                if rpos >= 0:
                    for offset in range(1, topk_per_route + 1):
                        idx = rpos + offset
                        if 0 <= idx < len(c):
                            if c[idx] != x:
                                cands.append((idx, int(c[idx])))
                        else:
                            break
            
            n_found = len(cands)
            for i in range(topk_per_route):
                if i < n_found:
                    idx_out[t, i] = cands[i][0]
                    val_out[t, i] = cands[i][1]
                else:
                    idx_out[t, i] = -1
                    val_out[t, i] = 0

        return (b_idx, route_idx, idx_out, val_out)

    pool = _get_rosa_thread_pool()
    if pool is None:
        results = []
        for b in range(B):
            for m in range(M):
                results.append(process_batch_route(b, m))
    else:
        futures = []
        for b in range(B):
            for m in range(M):
                futures.append(pool.submit(process_batch_route, b, m))
        results = [f.result() for f in futures]

    for (b_idx, route_idx, idx_out, val_out) in results:
        indices_list[route_idx][b_idx] = torch.from_numpy(idx_out).to(device=device, dtype=torch.int32)
        values_list[route_idx][b_idx]  = torch.from_numpy(val_out).to(device=device, dtype=torch.int32)

    return indices_list, values_list


def _forward_one_rosa_route_cpu_mt(
    route_idx: int,
    wlm_k: nn.Linear,
    wlm_q: nn.Linear,
    wlm_v: nn.Linear,
    v_emb: nn.Embedding,
    x: Tensor,
    cfg: LCGRosaConfig,
) -> Tensor:
    device = x.device
    B, T, D = x.shape

    logits_k = wlm_k(x)
    key = logits_k.argmax(dim=-1)

    logits_q = wlm_q(x)
    query = logits_q.argmax(dim=-1)

    logits_v = wlm_v(x)
    value = logits_v.argmax(dim=-1)

    key_cpu = key.cpu().numpy()

    cpu_event = torch.cuda.Event()
    cpu_event.record()

    K_qk = cfg.qk_vocab_size
    max_states = T * 2 + 10
    batch_arr = [key_cpu[b] for b in range(B)]
    
    pool = _get_rosa_thread_pool()
    if pool is None:
        nxt_batch, link_batch, length_batch, e_batch, last_batch, size_batch = \
            _batch_build_sam(batch_arr, T, K_qk, max_states)
        c_batch = [key_cpu[b] for b in range(B)]
        result = _ImmediateFuture((nxt_batch, link_batch, length_batch, e_batch, last_batch, size_batch, c_batch))
    else:
        result = pool.submit(
            _batch_build_sam,
            batch_arr,
            T,
            K_qk,
            max_states,
        )

    pos_subsample = cfg.pos_subsample

    def finalize_cpu():
        nxt_batch, link_batch, length_batch, e_batch, last_batch, size_batch = result.result()[:6]
        c_batch = result.result()[6] if len(result.result()) > 6 else [key_cpu[b] for b in range(B)]

        indices, values_codes = _lcg_build_position_sets_per_route_cpu(
            [query],
            nxt_batch,
            link_batch,
            length_batch,
            e_batch,
            last_batch,
            size_batch,
            c_batch,
            K_qk,
            cfg.lcg_topk,
            pos_subsample,
        )
        
        _wait_event(cpu_event)
        return indices[0], values_codes[0]

    pool2 = _get_rosa_thread_pool()
    if pool2 is None:
        fut_finalize = _ImmediateFuture(finalize_cpu())
    else:
        fut_finalize = pool2.submit(finalize_cpu)

    value_1d = value.view(-1)
    v_out_flat = v_emb(value_1d)
    v_out = v_out_flat.view(B, T, D)

    indices, values_codes = fut_finalize.result()

    mask = (indices >= 0)
    indices_safe = indices.clone()
    indices_safe[~mask] = 0

    B, T, topk = indices_safe.shape

    v_gather = v_out.gather(
        dim=1,
        index=indices_safe.unsqueeze(-1).expand(B, T, topk, D)
    )

    v_codes_1d = values_codes.view(-1)
    v_lookup_flat = v_emb(v_codes_1d)
    v_lookup = v_lookup_flat.view(B, T, topk, D)

    rosa_result = (v_gather * v_lookup).sum(dim=2)
    rosa_result = rosa_result * mask.any(dim=-1, keepdim=True).float()

    return rosa_result


class MultiRouteRosaLayer(nn.Module):
    def __init__(
        self,
        num_routes: int,
        hidden_size: int,
        cfg: LCGRosaConfig,
    ):
        super().__init__()
        self.num_routes = num_routes
        self.hidden_size = hidden_size
        self.cfg = cfg

        qk_vocab = cfg.qk_vocab_size
        v_vocab  = cfg.v_vocab_size

        self.rosa_wlm_q_list = nn.ModuleList([
            nn.Linear(hidden_size, qk_vocab, bias=False) for _ in range(num_routes)
        ])
        self.rosa_wlm_k_list = nn.ModuleList([
            nn.Linear(hidden_size, qk_vocab, bias=False) for _ in range(num_routes)
        ])
        self.rosa_wlm_v_list = nn.ModuleList([
            nn.Linear(hidden_size, v_vocab,  bias=False) for _ in range(num_routes)
        ])

        self.rosa_v_emb_list = nn.ModuleList([
            nn.Embedding(v_vocab + 1, hidden_size, padding_idx=0)
            for _ in range(num_routes)
        ])

        self.rosa_alpha = nn.Parameter(torch.zeros(num_routes))

        self._init_weights()

    def _init_weights(self):
        for m in range(self.num_routes):
            nn.init.normal_(self.rosa_wlm_q_list[m].weight, mean=0.0, std=0.01)
            nn.init.normal_(self.rosa_wlm_k_list[m].weight, mean=0.0, std=0.01)
            nn.init.normal_(self.rosa_wlm_v_list[m].weight, mean=0.0, std=0.01)

            with torch.no_grad():
                self.rosa_v_emb_list[m].weight[0].zero_()
                self.rosa_v_emb_list[m].weight[1:].normal_(mean=0.0, std=0.01)

    def forward(self, x: Tensor) -> Tensor:
        nvtx.range_push("MultiRouteRosaLayer.forward")

        results = []
        for m in range(self.num_routes):
            nvtx.range_push(f"route_{m}")
            
            r_m = _forward_one_rosa_route_cpu_mt(
                route_idx=m,
                wlm_k=self.rosa_wlm_k_list[m],
                wlm_q=self.rosa_wlm_q_list[m],
                wlm_v=self.rosa_wlm_v_list[m],
                v_emb=self.rosa_v_emb_list[m],
                x=x,
                cfg=self.cfg,
            )
            results.append(r_m)
            nvtx.range_pop()

        stacked = torch.stack(results, dim=0)
        alpha = torch.softmax(self.rosa_alpha, dim=0)
        weighted = (stacked * alpha.view(-1, 1, 1, 1)).sum(dim=0)

        nvtx.range_pop()
        return weighted


def patch_qwen3_with_multiroute_rosa(model: Qwen3ForCausalLM):
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    first_global = FIRST_GLOBAL_LAYERS

    cfg = LCGRosaConfig(
        seq_len=SEQ_LEN,
        attn_window=ATTN_WINDOW,
        num_routes=ROSA_NUM_ROUTES,
        qk_vocab_size=ROSA_QK_VOCAB_SIZE,
        v_vocab_size=ROSA_V_VOCAB_SIZE,
        lcg_topk=None,
        pos_subsample=LCG_POS_SUBSAMPLE,
        first_global_layers=first_global,
        use_flash_attn=USE_FLASH_ATTN,
    )

    for i in range(num_layers):
        layer: Qwen3DecoderLayer = model.model.layers[i]

        if i < first_global:
            continue

        rosa_module = MultiRouteRosaLayer(
            num_routes=ROSA_NUM_ROUTES,
            hidden_size=hidden_size,
            cfg=cfg,
        )
        rosa_module.to(device=layer.self_attn.q_proj.weight.device, dtype=layer.self_attn.q_proj.weight.dtype)
        layer.rosa_module = rosa_module

        layer.rosa_qk_vocab_size = ROSA_QK_VOCAB_SIZE
        layer.rosa_v_vocab_size = ROSA_V_VOCAB_SIZE

        old_forward = layer.forward

        def make_new_forward(layer_ref, old_fn):
            def new_forward(
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[Tensor] = None,
                past_key_value = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[Tensor] = None,
                **kwargs,
            ):
                residual = hidden_states

                if _os.environ.get("ROSA_INJECT_MODE", "") == "pre_attn":
                    if hasattr(layer_ref, "rosa_module"):
                        delta = layer_ref.rosa_module(hidden_states)
                        hidden_states = hidden_states + delta

                outputs = old_fn(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )

                hidden_states = outputs[0]

                if _os.environ.get("ROSA_INJECT_MODE", "") == "post_attn":
                    if hasattr(layer_ref, "rosa_module"):
                        delta = layer_ref.rosa_module(residual)
                        hidden_states = hidden_states + delta

                return (hidden_states,) + outputs[1:]

            return new_forward

        layer.forward = make_new_forward(layer, old_forward)


class FixedLenLMCollator:
    def __init__(self, pad_token_id: int, seq_len: int):
        self.pad_token_id = pad_token_id
        self.seq_len = seq_len

    def __call__(self, features: List[Dict[str, List[int]]]):
        batch = []
        for f in features:
            ids = f["input_ids"]
            if len(ids) > self.seq_len:
                ids = ids[:self.seq_len]
            elif len(ids) < self.seq_len:
                ids = ids + [self.pad_token_id] * (self.seq_len - len(ids))
            batch.append(torch.tensor(ids, dtype=torch.long))

        batch_tensor = torch.stack(batch, dim=0)
        labels = batch_tensor.clone()
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": batch_tensor,
            "labels": labels,
        }


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_LOCAL_DIR, trust_remote_code=True)
    if USE_FLASH_ATTN:
        config._attn_implementation = ""
    else:
        config._attn_implementation = ""

    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        config=config,
        torch_dtype=torch.bfloat16 if BF16 else torch.float16,
        low_cpu_mem_usage=True,
    )

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    model.config.use_cache = False

    patch_qwen3_with_multiroute_rosa(model)
    return model, tokenizer

def save_rosa_only(model: Qwen3ForCausalLM, out_dir: str):
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
            "pos_subsample": float(globals().get("LCG_POS_SUBSAMPLE", 1.0)),
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
    return _env_int("RANK", 0) == 0

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
    assert test_ds is not None, "test or validation split required"

    model, tokenizer = build_model_and_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data_collator = FixedLenLMCollator(pad_token_id=pad_id, seq_len=SEQ_LEN)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LR_ROSA,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS,
        save_strategy="no",
        report_to="none",
        fp16=(not BF16) and torch.cuda.is_available(),
        bf16=BF16,
        dataloader_num_workers=0,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        optim="adamw_torch",
    )

    optimizer_params = build_optimizer_params(model)

    class _Trainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(optimizer_params, betas=(0.9, 0.98), eps=1e-8)
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
