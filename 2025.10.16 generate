# Parameters and paths
BASE_MODEL_DIR = ""
ROSA_DIR = ""

# Attention window and layer type configuration
ATTN_WINDOW = None
FIRST_GLOBAL_LAYERS = None
USE_FLASH_ATTN = None
INJECT_MODE = None

# Device and precision settings
import torch
def _choose_dtype():
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32
DEVICE = None
TORCH_DTYPE = None

# General settings
SEED = None

import os
import json
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer


class _SAMFoldedCPU:
    __slots__ = ("next", "link", "length", "e", "last", "size", "K", "c", "last_sym")

    def __init__(self, K: int, init_capacity: int = None):
        import numpy as _np
        self.K = int(K)
        S = int(max(4, init_capacity))
        self.next   = _np.full((S, self.K), -1, dtype=_np.int32)
        self.link   = _np.full((S,),      -1, dtype=_np.int32)
        self.length = _np.zeros((S,),         dtype=_np.int32)
        self.e      = _np.full((S,),      -1, dtype=_np.int32)
        self.last   = 0
        self.size   = 1
        self.c: List[int] = []
        self.last_sym: Optional[int] = None

    def _ensure_capacity(self):
        if self.size + 3 < self.next.shape[0]:
            return
        import numpy as _np
        oldS, K = self.next.shape
        newS = int(oldS * 2)
        nxt2 = _np.full((newS, K), -1, dtype=_np.int32)
        lnk2 = _np.full((newS,),   -1, dtype=_np.int32)
        len2 = _np.zeros((newS,),      dtype=_np.int32)
        e2   = _np.full((newS,),   -1, dtype=_np.int32)
        nxt2[:oldS] = self.next
        lnk2[:oldS] = self.link
        len2[:oldS] = self.length
        e2[:oldS]   = self.e
        self.next, self.link, self.length, self.e = nxt2, lnk2, len2, e2

    def _new_state(self, L: int) -> int:
        self._ensure_capacity()
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

        cur = self._new_state(int(length[last]) + 1)
        p = last
        while p != -1 and nxt[p, x] == -1:
            nxt[p, x] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = int(nxt[p, x])
            if int(length[p]) + 1 == int(length[q]):
                link[cur] = q
            else:
                clone = self._new_state(int(length[p]) + 1)
                nxt[clone, :] = nxt[q, :]
                link[clone]   = link[q]
                e[clone]      = e[q]
                while p != -1 and nxt[p, x] == q:
                    nxt[p, x] = clone
                    p = link[p]
                link[q]   = clone
                link[cur] = clone

        v = cur
        while v != -1 and int(e[v]) != pos:
            e[v] = pos
            v = link[v]

        self.last = cur
        self.last_sym = x

    def query_then_commit(self, x: int) -> int:
        q = self.match_next(x)
        a = self.nextdiff_from_state(q)
        if self.last_sym is None or x != self.last_sym:
            self.extend_run(x)
        return a


class _RosaLayerRuntimeState:
    def __init__(self, K: int, M: int, init_capacity: int = None):
        self.K = int(K)
        self.M = int(M)
        self.init_capacity = int(init_capacity)
        self.sam: List[List[_SAMFoldedCPU]] = []
        self.batch_size: Optional[int] = None

    def reset(self, batch_size: int):
        self.sam = [[_SAMFoldedCPU(self.K, self.init_capacity) for _ in range(self.M)]
                    for _ in range(batch_size)]
        self.batch_size = batch_size

    def _ensure_batch(self, batch_size: int):
        if self.batch_size != batch_size or self.batch_size is None:
            self.reset(batch_size)

    def process(self, z_btm: torch.Tensor) -> torch.Tensor:
        assert z_btm.ndim == 3
        B, T, M = z_btm.shape
        self._ensure_batch(B)
        y = torch.empty((B, T, M), dtype=torch.int64)
        z_np = z_btm.detach().to(torch.int32).cpu().numpy()
        for b in range(B):
            for t in range(T):
                for m in range(M):
                    x = int(z_np[b, t, m])
                    a = self.sam[b][m].query_then_commit(x)
                    y[b, t, m] = int(a)
        return y


def _find_rosa_file(path: str) -> str:
    if os.path.isdir(path):
        cand = os.path.join(path, "rosa_adapters.pt")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"Could not find rosa_adapters.pt in directory: {path}")
        return cand
    return path

def _maybe_load_meta(path_dir: str) -> Dict:
    meta = {}
    for name in ("rosa_meta.json", "run_meta.json"):
        p = os.path.join(path_dir, name)
        if os.path.exists(p):
            try:
                meta = json.load(open(p, "r", encoding="utf-8"))
                break
            except Exception:
                pass
    return meta

def _group_by_layer_and_route(ckpt: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    by_layer: Dict[int, Dict[str, Dict[int, torch.Tensor]]] = {}
    for k, v in ckpt.items():
        if not k.startswith("model.layers."):
            continue
        parts = k.split(".")
        if len(parts) < 7:
            continue
        try:
            li = int(parts[2])
        except Exception:
            continue
        if "rosa_wlm_list" in k and parts[-1] == "weight":
            m = int(parts[4])
            by_layer.setdefault(li, {}).setdefault("wlm", {})[m] = v
        elif "rosa_emb_list" in k and parts[-1] == "weight":
            m = int(parts[4])
            by_layer.setdefault(li, {}).setdefault("emb", {})[m] = v
    return by_layer


def attach_rosa_adapters_for_inference(
    model: Qwen3ForCausalLM,
    rosa_adapters_path_or_dir: str,
    inject_mode: str = None,
    init_capacity: int = None
):
    device = next(model.parameters()).device
    base_dtype = next(model.parameters()).dtype

    adapters_path = _find_rosa_file(rosa_adapters_path_or_dir)
    adapters_dir = os.path.dirname(adapters_path)
    meta = _maybe_load_meta(adapters_dir)
    if not inject_mode and isinstance(meta, dict):
        inject_mode = str(meta.get("inject_mode", "pre_attn")).lower()
    inject_mode = inject_mode.lower()
    assert inject_mode in ("pre_attn", "post_attn")

    ckpt = torch.load(adapters_path, map_location="cpu")
    groups = _group_by_layer_and_route(ckpt)

    H = model.config.hidden_size

    for li, layer in enumerate(model.model.layers):
        if li not in groups:
            continue
        grp = groups[li]
        wlm_map: Dict[int, torch.Tensor] = grp.get("wlm", {})
        emb_map: Dict[int, torch.Tensor] = grp.get("emb", {})
        if len(wlm_map) == 0 or len(emb_map) == 0:
            continue

        M = max(max(wlm_map.keys()), max(emb_map.keys())) + 1
        any_w = next(iter(wlm_map.values()))
        K = int(any_w.shape[0])

        layer.rosa_wlm_list = nn.ModuleList(
            [nn.Linear(H, K, bias=False).to(dtype=base_dtype, device=device) for _ in range(M)]
        )
        for m in range(M):
            layer.rosa_wlm_list[m].weight.data.copy_(wlm_map[m].to(dtype=base_dtype))

        layer.rosa_emb_list = nn.ModuleList(
            [nn.Embedding(K + 1, H).to(dtype=base_dtype, device=device) for _ in range(M)]
        )
        for m in range(M):
            layer.rosa_emb_list[m].weight.data.copy_(emb_map[m].to(dtype=base_dtype))
            with torch.no_grad():
                layer.rosa_emb_list[m].weight.data[0].zero_()

        layer.rosa_num_routes = M
        layer.rosa_vocab_size = K
        layer._rosa_runtime = _RosaLayerRuntimeState(K=K, M=M, init_capacity=init_capacity)

        def _forward_with_rosa_infer(self: Qwen3DecoderLayer,
                                     hidden_states: torch.Tensor,
                                     attention_mask: Optional[torch.Tensor] = None,
                                     position_ids: Optional[torch.LongTensor] = None,
                                     past_key_values=None,
                                     use_cache: Optional[bool] = False,
                                     cache_position: Optional[torch.LongTensor] = None,
                                     position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                                     **kwargs) -> torch.Tensor:
            residual = hidden_states
            B, T, H_local = hidden_states.shape
            assert H_local == H

            need_reset = (getattr(self._rosa_runtime, "batch_size", None) != B)
            if cache_position is not None:
                try:
                    if int(cache_position.min().item()) == 0:
                        need_reset = True
                except Exception:
                    pass
            if need_reset:
                self._rosa_runtime.reset(B)

            u_head = self.input_layernorm(hidden_states)
            logits_list, z_list = [], []
            for head in self.rosa_wlm_list:
                if u_head.dtype is not head.weight.dtype:
                    logits_m = F.linear(u_head.to(head.weight.dtype), head.weight, None).to(u_head.dtype)
                else:
                    logits_m = F.linear(u_head, head.weight, None)
                logits_list.append(logits_m)
                z_list.append(torch.argmax(logits_m, dim=-1))
            logits_all = torch.stack(logits_list, dim=2)
            z_btm = torch.stack(z_list, dim=2).to(torch.int64)

            y_btm = self._rosa_runtime.process(z_btm)
            y_idx = torch.where(y_btm >= 0, y_btm + 1, torch.zeros_like(y_btm))

            e_sum = 0
            for m, emb in enumerate(self.rosa_emb_list):
                e_m = F.embedding(y_idx[:, :, m], emb.weight)
                e_sum = e_sum + e_m
            v = (e_sum / float(self.rosa_num_routes)).to(u_head.dtype)

            if inject_mode == "pre_attn":
                u_attn = self.input_layernorm(hidden_states + v)
                attn_out, _ = self.self_attn(
                    hidden_states=u_attn,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + attn_out
            else:
                attn_out, _ = self.self_attn(
                    hidden_states=u_head,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + attn_out + v

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        layer.forward = _forward_with_rosa_infer.__get__(layer, Qwen3DecoderLayer)

    orig_reorder_cache = getattr(model, "_reorder_cache", None)

    def _reorder_cache_with_rosa(past_key_values, beam_idx: torch.LongTensor):
        if orig_reorder_cache is not None:
            pkv = orig_reorder_cache(past_key_values, beam_idx)
        else:
            pkv = past_key_values
        idx = beam_idx.detach().cpu().tolist()
        for lyr in model.model.layers:
            if hasattr(lyr, "_rosa_runtime") and lyr._rosa_runtime.batch_size:
                old = lyr._rosa_runtime.sam
                new = [old[i] for i in idx]
                lyr._rosa_runtime.sam = new
                lyr._rosa_runtime.batch_size = len(new)
        return pkv

    model._reorder_cache = _reorder_cache_with_rosa

    def _reset_rosa_state(batch_size: int):
        for lyr in model.model.layers:
            if hasattr(lyr, "_rosa_runtime"):
                lyr._rosa_runtime.reset(batch_size)
    model.reset_rosa_state = _reset_rosa_state

    return model


def _set_layer_types_for_sliding(config, first_global_layers: int, window: int, use_flash: bool):
    config.sliding_window = int(window)
    config.max_window_layers = int(first_global_layers)
    if (not hasattr(config, "layer_types")) or (config.layer_types is None):
        config.layer_types = [
            "full_attention" if i < config.max_window_layers else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]
    backend = "flash_attention_2" if use_flash else "sdpa"
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = backend
    else:
        config._attn_implementation = backend
    return config


def load_qwen3_with_rosa_for_inference(
    base_model_dir: str,
    rosa_dir: str,
    device: str = None,
    torch_dtype: torch.dtype = None,
    inject_mode: str = None,
    attn_window: int = None,
    first_global_layers: int = None,
    use_flash_attn: bool = None,
):
    config = AutoConfig.from_pretrained(base_model_dir)
    config = _set_layer_types_for_sliding(config, first_global_layers, attn_window, use_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)

    kwargs = dict(torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if device == "cuda":
        kwargs["device_map"] = {"": "cuda"}
    model = Qwen3ForCausalLM.from_pretrained(base_model_dir, config=config, **kwargs)
    model.eval()
    model.config.use_cache = True

    attach_rosa_adapters_for_inference(model, rosa_dir, inject_mode=inject_mode)

    return model, tokenizer


def main():
    torch.manual_seed(SEED)

    model, tokenizer = load_qwen3_with_rosa_for_inference(
        base_model_dir=BASE_MODEL_DIR,
        rosa_dir=ROSA_DIR,
        device=DEVICE,
        torch_dtype=TORCH_DTYPE,
        inject_mode=INJECT_MODE,
        attn_window=ATTN_WINDOW,
        first_global_layers=FIRST_GLOBAL_LAYERS,
        use_flash_attn=USE_FLASH_ATTN,
    )

    B = 1
    if hasattr(model, "reset_rosa_state"):
        model.reset_rosa_state(batch_size=B)

    prompt = ""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
    )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
