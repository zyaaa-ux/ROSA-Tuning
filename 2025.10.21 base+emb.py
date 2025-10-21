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

_USE_NUMBA = _os.environ.get("ROSA_USE_NUMBA", "").lower() not in ("", "")
_PARALLEL  = _os.environ.get("ROSA_NUMBA_PARALLEL", "").lower() not in ("", "")

_ROSA_L0_METHOD = _os.environ.get("ROSA_L0_METHOD", "").lower()
_ROSA_L0_TRAIN = True

try:
    import numba as _nb
    _NUMBA_OK = _USE_NUMBA
except Exception:
    _NUMBA_OK = False

MODEL_LOCAL_DIR = ""
DATASET_DIR     = ""
OUTPUT_DIR      = ""
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_FLASH_ATTN = True
BF16 = True if torch.cuda.is_available() else False

SEQ_LEN = None
ATTN_WINDOW = None
FIRST_GLOBAL_LAYERS = None

ROSA_NUM_ROUTES: int = None
ROSA_VOCAB_SIZE: int = None
LCG_TOPK: int = None
LCG_POS_SUBSAMPLE: float = None
ROSA_CPU_WORKERS: int = None

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

_ROSA_POOL = None
_ROSA_POOL_PID = None
_ROSA_POOL_LOCK = threading.Lock()

def _env_int(k, default):
    try:
        return int(os.environ.get(k, default))
    except Exception:
        return default

LOCAL_WORLD_SIZE = _env_int("LOCAL_WORLD_SIZE", max(1, torch.cuda.device_count()))
LOCAL_RANK = _env_int("LOCAL_RANK", 0)
GLOBAL_RANK = _env_int("RANK", 0)

try:
    import torch.cuda.nvtx as nvtx
except Exception:
    class _DummyNVTX:
        def range_push(self, *a, **k): pass
        def range_pop(self): pass
    nvtx = _DummyNVTX()

from concurrent.futures import ThreadPoolExecutor
_ROSA_THREAD_POOL = ThreadPoolExecutor(
    max_workers=max(1, _env_int("ROSA_THREAD_WORKERS", 2))
)

def _wait_event(ev: "torch.cuda.Event"):
    try:
        ev.synchronize()
    except Exception:
        pass

def _available_logical_cpus() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass

    def _parse_cpuset(s: str) -> int:
        n = 0
        for part in s.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                n += int(b) - int(a) + 1
            else:
                n += 1
        return n

    try:
        for p in ("", ""):
            if os.path.exists(p):
                with open(p) as f:
                    txt = f.read().strip()
                if txt:
                    return max(1, _parse_cpuset(txt))
    except Exception:
        pass

    return max(1, os.cpu_count() or 1)

_TOTAL_CPUS = _available_logical_cpus()
_PER_RANK_CPUS = max(1, _TOTAL_CPUS // max(1, LOCAL_WORLD_SIZE))

torch.set_num_threads(_PER_RANK_CPUS)

def _rosa_worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "")
    os.environ.setdefault("MKL_NUM_THREADS", "")
    try:
        import torch as _t
        _t.set_num_threads(1)
    except Exception:
        pass

import numpy as _np
import torch

def to_c_np(t: torch.Tensor, dtype=_np.int32):
    return _np.asarray(
        t.detach().to(torch.int32 if dtype is _np.int32 else torch.bool).contiguous().cpu().numpy(),
        order="C",
        dtype=dtype,
    )

def to_c_bool_np(t: torch.Tensor):
    return _np.asarray(t.detach().to(torch.bool).contiguous().cpu().numpy(), order="C", dtype=_np.bool_)

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

        cur = self._new_state(length[last] + 1)
        p = last
        while p != -1 and nxt[p, x] == -1:
            nxt[p, x] = cur
            p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = int(nxt[p, x])
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = self._new_state(length[p] + 1)
                nxt[clone, :] = nxt[q, :]
                link[clone]   = link[q]
                e[clone]      = e[q]
                while p != -1 and nxt[p, x] == q:
                    nxt[p, x] = clone
                    p = link[p]
                link[q]   = clone
                link[cur] = clone

        v = cur
        while v != -1 and e[v] != pos:
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

class _SAMFoldedDict:
    __slots__ = ("next", "link", "length", "e", "last", "c", "last_sym")
    def __init__(self):
        self.next   = [dict()]
        self.link   = [-1]
        self.length = [0]
        self.e      = [-1]
        self.last   = 0
        self.c      = []
        self.last_sym = None

    def _new_state(self, L: int) -> int:
        self.next.append(dict()); self.link.append(-1)
        self.length.append(L);    self.e.append(-1)
        return len(self.next) - 1

    def match_next(self, x: int) -> int:
        p = self.last
        while p != -1 and x not in self.next[p]:
            p = self.link[p]
        return -1 if p == -1 else self.next[p][x]

    def nextdiff_from_state(self, q: int) -> int:
        if q == -1: return -1
        rpos = self.e[q]; nxt = rpos + 1
        return self.c[nxt] if (0 <= rpos and nxt < len(self.c)) else -1

    def extend_run(self, x: int, pos: int):
        last = self.last
        cur = self._new_state(self.length[last] + 1)
        p = last
        while p != -1 and x not in self.next[p]:
            self.next[p][x] = cur
            p = self.link[p]
        if p == -1:
            self.link[cur] = 0
        else:
            q = self.next[p][x]
            if self.length[p] + 1 == self.length[q]:
                self.link[cur] = q
            else:
                clone = self._new_state(self.length[p] + 1)
                self.next[clone] = dict(self.next[q])
                self.link[clone] = self.link[q]
                self.e[clone]    = self.e[q]
                while p != -1 and self.next[p].get(x, None) == q:
                    self.next[p][x] = clone
                    p = self.link[p]
                self.link[q] = clone
                self.link[cur] = clone
        v = cur
        while v != -1 and self.e[v] != pos:
            self.e[v] = pos
            v = self.link[v]
        self.last = cur
        self.last_sym = x

def _rosa_tokens_nextdiff_seq(x_seq: "List[int]") -> "List[int]":
    sam = _SAMFoldedDict()
    y = []
    last_sym = None
    for t, x in enumerate(x_seq):
        q = sam.match_next(int(x))
        y.append(sam.nextdiff_from_state(q))
        if last_sym is None or x != last_sym:
            sam.c.append(int(x))
            sam.extend_run(int(x), len(sam.c) - 1)
            last_sym = x
    return y

def _rosa_tokens_nextdiff_batch(x_bt: torch.Tensor) -> torch.Tensor:
    x_np = x_bt.detach().to("cpu", non_blocking=True).numpy()
    out = []
    for b in range(x_np.shape[0]):
        out.append(_rosa_tokens_nextdiff_seq(list(map(int, x_np[b]))))
    y = torch.tensor(out, dtype=torch.long)
    return y

class _PinnedBufferPool:
    def __init__(self):
        self._pool = {}

    def get(self, tag: str, shape: tuple, dtype: torch.dtype = torch.int32):
        key = (tag, shape, dtype)
        t = self._pool.get(key, None)
        if t is None or t.shape != torch.Size(shape) or t.dtype != dtype:
            t = torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
            self._pool[key] = t
        return t

_PINNED_POOL = _PinnedBufferPool()

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False))
    def _count_runs_nb(arr):
        T = arr.shape[0]
        if T == 0: 
            return 0
        cnt = 1
        last = arr[0]
        for i in range(1, T):
            if arr[i] != last:
                cnt += 1
                last = arr[i]
        return cnt

    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_new_state(next_arr, link, length, e, size, L):
        s = size[0]
        size[0] = s + 1
        length[s] = L
        link[s]   = -1
        e[s]      = -1
        for j in range(next_arr.shape[1]):
            next_arr[s, j] = -1
        return s

    @(_nb.njit(cache=True, fastmath=False))
    def _sam_extend(next_arr, link, length, e, last_arr, size, x, pos):
        last = last_arr[0]
        cur = _sam_new_state(next_arr, link, length, e, size, length[last] + 1)
        p = last
        K = next_arr.shape[1]

        while p != -1 and next_arr[p, x] == -1:
            next_arr[p, x] = cur
            p = link[p]

        if p == -1:
            link[cur] = 0
        else:
            q = next_arr[p, x]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = _sam_new_state(next_arr, link, length, e, size, length[p] + 1)
                for j in range(K):
                    next_arr[clone, j] = next_arr[q, j]
                link[clone] = link[q]
                e[clone]    = e[q]
                while p != -1 and next_arr[p, x] == q:
                    next_arr[p, x] = clone
                    p = link[p]
                link[q]   = clone
                link[cur] = clone

        v = cur
        while v != -1 and e[v] != pos:
            e[v] = pos
            v = link[v]

        last_arr[0] = cur

    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_match_next(next_arr, link, last_state, x):
        p = last_state
        while p != -1 and next_arr[p, x] == -1:
            p = link[p]
        if p == -1:
            return -1
        return next_arr[p, x]

    @(_nb.njit(cache=True, fastmath=False))
    def _rosa_seq_folded_nb(z, K):
        T = z.shape[0]
        y = _np.empty((T,), dtype=_np.int32)
        if T == 0:
            return y

        R = _count_runs_nb(z)
        S_cap = 2 * R + 5
        next_arr = _np.empty((S_cap, K), dtype=_np.int32)
        link     = _np.empty((S_cap,),   dtype=_np.int32)
        length   = _np.empty((S_cap,),   dtype=_np.int32)
        e        = _np.empty((S_cap,),   dtype=_np.int32)
        for j in range(K):
            next_arr[0, j] = -1
        link[0]   = -1
        length[0] = 0
        e[0]      = -1
        size = _np.empty((1,), dtype=_np.int32)
        size[0] = 1
        last_arr = _np.empty((1,), dtype=_np.int32)
        last_arr[0] = 0

        c = _np.empty((T,), dtype=_np.int32)
        c_len = 0
        last_sym = -2147483647

        for t in range(T):
            x = z[t]
            q = _sam_match_next(next_arr, link, last_arr[0], x)
            if q == -1:
                y[t] = -1
            else:
                rpos = e[q]
                nxt  = rpos + 1
                if 0 <= rpos and nxt < c_len:
                    y[t] = c[nxt]
                else:
                    y[t] = -1
            if t == 0 or x != last_sym:
                c[c_len] = x
                c_len += 1
                _sam_extend(next_arr, link, length, e, last_arr, size, x, c_len - 1)
                last_sym = x

        return y

    @(_nb.njit(cache=True, fastmath=False))
    def _rosa_batch_btm_nb(z_btm, K):
        B, T, M = z_btm.shape
        y_btm = _np.empty((B, T, M), dtype=_np.int32)
        if _PARALLEL:
            for b in _nb.prange(B):
                for m in range(M):
                    y_btm[b, :, m] = _rosa_seq_folded_nb(z_btm[b, :, m], K)
        else:
            for b in range(B):
                for m in range(M):
                    y_btm[b, :, m] = _rosa_seq_folded_nb(z_btm[b, :, m], K)
        return y_btm

    @(_nb.njit(cache=True, fastmath=False))
    def _lcg_index_level_seq_nb(z, K, cand_tm, pos_mask):
        T = z.shape[0]
        topk = cand_tm.shape[1]
        out = _np.empty((T, topk), dtype=_np.int16)
        if T == 0:
            return out

        R = _count_runs_nb(z)
        S_cap = 2 * R + 5
        next_arr = _np.empty((S_cap, K), dtype=_np.int32)
        link     = _np.empty((S_cap,),   dtype=_np.int32)
        length   = _np.empty((S_cap,),   dtype=_np.int32)
        e        = _np.empty((S_cap,),   dtype=_np.int32)
        for j in range(K):
            next_arr[0, j] = -1
        link[0]   = -1
        length[0] = 0
        e[0]      = -1
        size = _np.empty((1,), dtype=_np.int32); size[0] = 1
        last_arr = _np.empty((1,), dtype=_np.int32); last_arr[0] = 0
        c = _np.empty((T,), dtype=_np.int32); c_len = 0
        last_sym = -2147483647

        for t in range(T):
            if not pos_mask[t]:
                for j in range(topk):
                    out[t, j] = -2
            else:
                for j in range(topk):
                    k = cand_tm[t, j]
                    q = _sam_match_next(next_arr, link, last_arr[0], k)
                    if q == -1:
                        out[t, j] = -1
                    else:
                        rpos = e[q]; nxt = rpos + 1
                        if 0 <= rpos and nxt < c_len:
                            out[t, j] = _np.int16(c[nxt])
                        else:
                            out[t, j] = -1
            x = z[t]
            if t == 0 or x != last_sym:
                c[c_len] = x; c_len += 1
                _sam_extend(next_arr, link, length, e, last_arr, size, x, c_len - 1)
                last_sym = x

        return out

    @(_nb.njit(cache=True, fastmath=False))
    def _lcg_index_level_bmt_nb(z_btm, K, cand_bmtk, pos_mask_bt):
        B, T, M = z_btm.shape
        topk = cand_bmtk.shape[3]
        out = _np.empty((B, M, T, topk), dtype=_np.int16)
        if _PARALLEL:
            for b in _nb.prange(B):
                for m in range(M):
                    out[b, m] = _lcg_index_level_seq_nb(z_btm[b, :, m], K, cand_bmtk[b, m], pos_mask_bt[b])
        else:
            for b in range(B):
                for m in range(M):
                    out[b, m] = _lcg_index_level_seq_nb(z_btm[b, :, m], K, cand_bmtk[b, m], pos_mask_bt[b])
        return out

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False))
    def _rosa_seq_with_ws_nb(z, K: int, S_cap: int):
        T = z.shape[0]
        y = _np.empty((T,), dtype=_np.int32)
        last_trace = _np.empty((T,), dtype=_np.int32)

        next_arr = _np.empty((S_cap, K), dtype=_np.int32)
        link     = _np.empty((S_cap,),   dtype=_np.int32)
        length   = _np.empty((S_cap,),   dtype=_np.int32)
        e        = _np.empty((S_cap,),   dtype=_np.int32)

        for j in range(K): next_arr[0, j] = -1
        link[0]   = -1
        length[0] = 0
        e[0]      = -1
        size = _np.empty((1,), dtype=_np.int32); size[0] = 1
        last_arr = _np.empty((1,), dtype=_np.int32); last_arr[0] = 0

        c = _np.empty((T,), dtype=_np.int32)
        c_len = 0
        last_sym = -2147483647

        for t in range(T):
            last_trace[t] = last_arr[0]
            x = z[t]
            q = _sam_match_next(next_arr, link, last_arr[0], x)
            if q == -1:
                y[t] = -1
            else:
                rpos = e[q]; nxt = rpos + 1
                if 0 <= rpos and nxt < c_len:
                    y[t] = c[nxt]
                else:
                    y[t] = -1
            if t == 0 or x != last_sym:
                c[c_len] = x; c_len += 1
                _sam_extend(next_arr, link, length, e, last_arr, size, x, c_len - 1)
                last_sym = x

        return y, last_trace, next_arr, link, e, c, _np.int32(c_len)

    @(_nb.njit(cache=True, fastmath=False))
    def _rosa_batch_btm_with_ws_nb(z_btm, K: int):
        B, T, M = z_btm.shape
        S_cap = 2 * T + 5

        y_btm         = _np.empty((B, T, M), dtype=_np.int32)
        last_btm      = _np.empty((B, T, M), dtype=_np.int32)
        next_bmsk     = _np.empty((B, M, S_cap, K), dtype=_np.int32)
        link_bms      = _np.empty((B, M, S_cap),     dtype=_np.int32)
        e_bms         = _np.empty((B, M, S_cap),     dtype=_np.int32)
        c_bmt         = _np.empty((B, M, T),         dtype=_np.int32)
        c_len_bm      = _np.empty((B, M),            dtype=_np.int32)

        if _PARALLEL:
            for idx in _nb.prange(B * M):
                b = idx // M
                m = idx % M
                y, last_tr, nxt, lnk, ee, cc, clen = _rosa_seq_with_ws_nb(z_btm[b, :, m], K, S_cap)
                y_btm[b, :, m] = y
                last_btm[b, :, m] = last_tr
                next_bmsk[b, m, :, :] = nxt
                link_bms[b, m, :] = lnk
                e_bms[b, m, :] = ee
                c_bmt[b, m, :] = 0
                for i in range(T):
                    if i < clen:
                        c_bmt[b, m, i] = cc[i]
                c_len_bm[b, m] = clen
        else:
            for b in range(B):
                for m in range(M):
                    y, last_tr, nxt, lnk, ee, cc, clen = _rosa_seq_with_ws_nb(z_btm[b, :, m], K, S_cap)
                    y_btm[b, :, m] = y
                    last_btm[b, :, m] = last_tr
                    next_bmsk[b, m, :, :] = nxt
                    link_bms[b, m, :] = lnk
                    e_bms[b, m, :] = ee
                    c_bmt[b, m, :] = 0
                    for i in range(T):
                        if i < clen:
                            c_bmt[b, m, i] = cc[i]
                    c_len_bm[b, m] = clen

        return y_btm, last_btm, next_bmsk, link_bms, e_bms, c_bmt, c_len_bm

def _rosa_batch_btm_with_ws_py(z_btm_np: "_np.ndarray", K: int):
    B, T, M = z_btm_np.shape
    S_cap = 2 * T + 5

    y_btm    = _np.empty((B, T, M), dtype=_np.int32)
    last_btm = _np.empty((B, T, M), dtype=_np.int32)
    next_bmsk = _np.empty((B, M, S_cap, K), dtype=_np.int32)
    link_bms  = _np.empty((B, M, S_cap),     dtype=_np.int32)
    e_bms     = _np.empty((B, M, S_cap),     dtype=_np.int32)
    c_bmt     = _np.empty((B, M, T),         dtype=_np.int32)
    c_len_bm  = _np.empty((B, M),            dtype=_np.int32)

    for b in range(B):
        for m in range(M):
            z = [int(v) for v in list(z_btm_np[b, :, m])]
            sam = _SAMFoldedCPU(max_states=S_cap, K=K)
            y_seq = []
            last_trace = []
            last_sym = None
            for t in range(T):
                last_trace.append(int(sam.last))
                x = z[t]
                q = sam.match_next(int(x))
                a = sam.nextdiff_from_state(q)
                y_seq.append(-1 if a == -1 else int(a))
                if (last_sym is None) or (x != last_sym):
                    sam.extend_run(int(x)); last_sym = x

            y_btm[b, :, m] = _np.asarray(y_seq, dtype=_np.int32)
            last_btm[b, :, m] = _np.asarray(last_trace, dtype=_np.int32)
            next_bmsk[b, m, :, :] = sam.next
            link_bms[b, m, :] = sam.link
            e_bms[b, m, :] = sam.e
            c_len = len(sam.c)
            c_len_bm[b, m] = c_len
            c_bmt[b, m, :] = 0
            for i in range(min(T, c_len)):
                c_bmt[b, m, i] = sam.c[i]

    return y_btm, last_btm, next_bmsk, link_bms, e_bms, c_bmt, c_len_bm

def _rosa_batch_btm_with_ws(z_btm_np: "_np.ndarray", K: int):
    if _NUMBA_OK and ('_rosa_batch_btm_with_ws_nb' in globals()):
        return _rosa_batch_btm_with_ws_nb(z_btm_np, int(K))
    return _rosa_batch_btm_with_ws_py(z_btm_np, int(K))

@torch.no_grad()
def _lcg_query_gpu(
    next_bmsk: torch.Tensor,
    link_bms: torch.Tensor,
    e_bms: torch.Tensor,
    c_bmt: torch.Tensor,
    c_len_bm: torch.Tensor,
    last_state_btm: torch.Tensor,
    cand_bmtk: torch.Tensor,
    pos_mask_bt: Optional[torch.Tensor],
) -> torch.Tensor:
    device = cand_bmtk.device
    B, M, S, K = next_bmsk.shape
    T = last_state_btm.size(1)
    topk = cand_bmtk.size(-1)

    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, M, T, topk).reshape(-1)
    m_idx = torch.arange(M, device=device).view(1, M, 1, 1).expand(B, M, T, topk).reshape(-1)
    t_idx = torch.arange(T, device=device).view(1, 1, T, 1).expand(B, M, T, topk).reshape(-1)

    p = last_state_btm.permute(0, 2, 1).contiguous()
    p = p.unsqueeze(-1).expand(-1, -1, -1, topk).reshape(-1).to(torch.long)

    x = cand_bmtk.reshape(-1).to(torch.long)

    q = torch.full_like(p, -1, dtype=torch.long)
    done = (p == -1)

    for _ in range(S):
        p_clamped = torch.clamp(p, 0, S - 1)
        nxt_val = next_bmsk[b_idx, m_idx, p_clamped, x].to(torch.long)
        has = (~done) & (nxt_val != -1)
        q = torch.where(has, nxt_val, q)
        done = done | has | (p == -1)
        if bool(done.all()):
            break
        p_next = link_bms[b_idx, m_idx, p_clamped].to(torch.long)
        p = torch.where(~done, p_next, p)

    q_valid = (q != -1)
    qc = torch.clamp(q, 0, S - 1)
    rpos = torch.where(q_valid, e_bms[b_idx, m_idx, qc].to(torch.long), torch.full_like(q, -1))
    nxtp = rpos + 1
    clen = c_len_bm[b_idx, m_idx].to(torch.long)
    valid2 = (rpos >= 0) & (nxtp < clen)
    nxtp_c = torch.clamp(nxtp, 0, c_bmt.size(2) - 1)
    cval = c_bmt[b_idx, m_idx, nxtp_c].to(torch.long)
    result = torch.where(q_valid & valid2, cval, torch.full_like(cval, -1))

    if pos_mask_bt is not None:
        pmask = pos_mask_bt.unsqueeze(1).unsqueeze(-1).expand(B, M, T, topk).reshape(-1)
        result = torch.where(pmask, result, torch.full_like(result, -2))

    return result.view(B, M, T, topk).to(torch.int16)

class MultiRouteLCGFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_in: torch.Tensor,
        logits_all: torch.Tensor,
        y_idx: torch.Tensor,
        E_stack: torch.Tensor,
        sam_next_bmsk: torch.Tensor,
        sam_link_bms: torch.Tensor,
        sam_e_bms: torch.Tensor,
        sam_c_bmt: torch.Tensor,
        sam_c_len_bm: torch.Tensor,
        sam_last_state_btm: torch.Tensor,
        pos_mask_cpu: Optional[list],
    ):
        ctx.save_for_backward(
            logits_all, y_idx, E_stack,
            sam_next_bmsk, sam_link_bms, sam_e_bms,
            sam_c_bmt, sam_c_len_bm, sam_last_state_btm
        )
        ctx.pos_mask_cpu = pos_mask_cpu
        return v_in

    @staticmethod
    def backward(ctx, grad_v: torch.Tensor):
        (logits_all, y_idx, E_stack,
         sam_next_bmsk, sam_link_bms, sam_e_bms,
         sam_c_bmt, sam_c_len_bm, sam_last_state_btm) = ctx.saved_tensors

        device = logits_all.device
        B, T, M, K = logits_all.shape

        with torch.no_grad():
            p_all = torch.softmax(logits_all.float(), dim=-1)
            p_all_bmtk = p_all.permute(0, 2, 1, 3).contiguous()
            topk = min(LCG_TOPK, K)
            idx_topk_bmt = torch.topk(p_all_bmtk, k=topk, dim=-1).indices

        with torch.no_grad():
            if ctx.pos_mask_cpu is None:
                pos_mask_bt = torch.ones((B, T), device=device, dtype=torch.bool)
            else:
                pos_mask_bt = torch.tensor(ctx.pos_mask_cpu, device=device, dtype=torch.bool)

        with torch.no_grad():
            g_no_scale = grad_v.float()
            S = torch.einsum("btd,mkd->bmtk", g_no_scale, E_stack.float())
            y_old_bmt = y_idx.permute(0, 2, 1).contiguous()
            y_old_exp = y_old_bmt.unsqueeze(-1).expand(-1, -1, -1, topk)

        y_cf_bmtk = _lcg_query_gpu(
            sam_next_bmsk, sam_link_bms, sam_e_bms,
            sam_c_bmt, sam_c_len_bm, sam_last_state_btm,
            idx_topk_bmt, pos_mask_bt
        )

        with torch.no_grad():
            y_cf_bmtk_long = y_cf_bmtk.to(torch.long)
            skip_mask = (y_cf_bmtk_long == -2)
            pad_mask  = (y_cf_bmtk_long == -1)
            idx_new   = torch.where(pad_mask, torch.zeros_like(y_cf_bmtk_long), y_cf_bmtk_long + 1)

            S_new = torch.gather(S, dim=-1, index=idx_new)
            S_old = torch.gather(S, dim=-1, index=y_old_exp)
            deltas = S_new - S_old
            deltas = torch.where(skip_mask, torch.zeros_like(deltas), deltas)

            probs_topk = torch.gather(p_all_bmtk, dim=-1, index=idx_topk_bmt)
            mean_delta = (probs_topk * deltas).sum(dim=-1, keepdim=True)
            grad_topk  = probs_topk * (deltas - mean_delta)

            grad_logits = torch.zeros_like(logits_all, dtype=p_all.dtype, device=device)
            grad_logits_flat = grad_logits.permute(0, 2, 1, 3).contiguous().view(B * M * T, K)
            idx_flat = idx_topk_bmt.contiguous().view(B * M * T, topk)
            src_flat = grad_topk.contiguous().view(B * M * T, topk)
            grad_logits_flat.scatter_add_(dim=1, index=idx_flat, src=src_flat)
            grad_logits = grad_logits_flat.view(B, M, T, K).permute(0, 2, 1, 3).to(logits_all.dtype)

        return grad_v, grad_logits, None, None, None, None, None, None, None, None, None

@dataclass
class FixedLenLMCollator:
    pad_token_id: int
    seq_len: int

    def __call__(self, features):
        input_ids = [f["input_ids"][: self.seq_len] for f in features]
        if "attention_mask" in features[0]:
            attention_mask = [f["attention_mask"][: self.seq_len] for f in features]
        else:
            attention_mask = [[1] * len(x) for x in input_ids]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        if input_ids.shape[1] < self.seq_len:
            pad_len = self.seq_len - input_ids.shape[1]
            pad = torch.full((input_ids.size(0), pad_len), self.pad_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, pad], dim=1)
            attention_mask = torch.cat([attention_mask, torch.zeros_like(pad)], dim=1)

        if "labels" in features[0]:
            labels = [f["labels"][: self.seq_len] for f in features]
            labels = torch.tensor(labels, dtype=torch.long)
            if labels.shape[1] < self.seq_len:
                lab_pad = torch.full((labels.size(0), self.seq_len - labels.shape[1]), -100, dtype=torch.long)
                labels = torch.cat([labels, lab_pad], dim=1)
        else:
            labels = input_ids.clone()
            labels[labels == self.pad_token_id] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer

def patch_qwen3_with_multiroute_rosa(model: Qwen3ForCausalLM):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from typing import List, Tuple
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    M: int = int(globals().get("ROSA_NUM_ROUTES"))
    K: int = int(globals().get("ROSA_VOCAB_SIZE"))
    PER_ROUTE_DIM = globals().get("ROSA_PER_ROUTE_DIM", None)
    inject_mode = os.environ.get("ROSA_INJECT_MODE", "").lower()
    lcg_enable  = os.environ.get("LCG_ENABLE", "") in ("", "")

    base_param = model.model.embed_tokens.weight
    base_dtype = base_param.dtype
    base_device = base_param.device
    hidden_size: int = model.config.hidden_size

    if PER_ROUTE_DIM is None:
        assert hidden_size % M == 0
        d = hidden_size // M
    else:
        d = int(PER_ROUTE_DIM)
        assert d * M == hidden_size
    route_dims: List[int] = [d] * M
    route_slices: List[Tuple[int, int]] = [(i * d, (i + 1) * d) for i in range(M)]

    for li, layer in enumerate(model.model.layers):
        if li == 0:
            continue

        layer.rosa_wlm_list = nn.ModuleList(
            [nn.Linear(route_dims[m], K, bias=False).to(dtype=base_dtype, device=base_device)
             for m in range(M)]
        )
        for w in layer.rosa_wlm_list:
            nn.init.xavier_uniform_(w.weight)

        layer.rosa_emb_list = nn.ModuleList(
            [nn.Embedding(K + 1, route_dims[m], padding_idx=0).to(dtype=base_dtype, device=base_device)
             for m in range(M)]
        )
        for emb in layer.rosa_emb_list:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight.data[0].zero_()

        layer.rosa_alpha = nn.Parameter(
            torch.zeros(hidden_size, dtype=base_dtype, device=base_device)
        )

        layer.rosa_num_routes = M
        layer.rosa_vocab_size = K
        layer.rosa_route_dims = route_dims
        layer.rosa_route_slices = route_slices

        def _forward_with_multiroute_rosa(
            self: Qwen3DecoderLayer,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor = None,
            position_ids: torch.LongTensor = None,
            past_key_values=None,
            use_cache: bool = False,
            cache_position: torch.LongTensor = None,
            position_embeddings=None,
            **kwargs
        ):
            residual = hidden_states
            B, T, H = hidden_states.shape
            M = self.rosa_num_routes
            K = self.rosa_vocab_size
            device = hidden_states.device
            slices = self.rosa_route_slices

            u_head = self.input_layernorm(hidden_states)
            logits_list = []
            for m, (s, e) in enumerate(slices):
                u_slice = u_head[:, :, s:e]
                logits_m = self.rosa_wlm_list[m](u_slice)
                logits_list.append(logits_m)
            logits_all = torch.stack(logits_list, dim=2)

            z_gpu = torch.argmax(logits_all, dim=-1).to(torch.int32)
            nvtx.range_push("ROSA_async_copy_and_run_ws")
            z_host = _PINNED_POOL.get("z_host", (B, T, M), dtype=torch.int32)
            z_host.copy_(z_gpu, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())

            def _wait_and_rosa_ws():
                _wait_event(ev)
                z_np = _np.asarray(z_host, order="C")
                return _rosa_batch_btm_with_ws(z_np, int(K))

            fut_ws = _ROSA_THREAD_POOL.submit(_wait_and_rosa_ws)
            nvtx.range_pop()

            (y_np, last_np, next_np, link_np, e_np, c_np, clen_np) = fut_ws.result()
            y_btm = torch.from_numpy(y_np).to(device=device, dtype=torch.long)
            y_idx = torch.where(y_btm >= 0, y_btm + 1, torch.zeros_like(y_btm))

            sam_next_bmsk      = torch.from_numpy(next_np).to(device=device, dtype=torch.int32)
            sam_link_bms       = torch.from_numpy(link_np).to(device=device, dtype=torch.int32)
            sam_e_bms          = torch.from_numpy(e_np).to(device=device, dtype=torch.int32)
            sam_c_bmt          = torch.from_numpy(c_np).to(device=device, dtype=torch.int32)
            sam_c_len_bm       = torch.from_numpy(clen_np).to(device=device, dtype=torch.int32)
            sam_last_state_btm = torch.from_numpy(last_np).to(device=device, dtype=torch.int32)

            v = torch.zeros((B, T, H), dtype=u_head.dtype, device=device)
            E_block = torch.zeros((M, K + 1, H), dtype=base_dtype, device=device)

            for m, (s, e) in enumerate(slices):
                idx_m = y_idx[:, :, m]
                v_slice = self.rosa_emb_list[m](idx_m)
                v[:, :, s:e] = v_slice.to(v.dtype)
                E_block[m, :, s:e] = self.rosa_emb_list[m].weight

            E_stack_gpu = E_block.detach()

            if lcg_enable:
                pos_mask_cpu = None
                if LCG_POS_SUBSAMPLE < 1.0:
                    mask = (torch.rand((B, T), device=device) < LCG_POS_SUBSAMPLE)
                    pos_mask_cpu = mask.detach().cpu().tolist()

                v = MultiRouteLCGFunction.apply(
                    v, logits_all, y_idx, E_stack_gpu,
                    sam_next_bmsk, sam_link_bms, sam_e_bms,
                    sam_c_bmt, sam_c_len_bm, sam_last_state_btm,
                    pos_mask_cpu
                )

            alpha = torch.sigmoid(self.rosa_alpha).view(1, 1, -1).to(
                dtype=hidden_states.dtype, device=device
            )

            if inject_mode == "pre_attn":
                mix = (1.0 - alpha) * hidden_states + alpha * v
                u_attn = self.input_layernorm(mix)
                attn_out, _ = self.self_attn(
                    hidden_states=u_attn, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
                hidden_states = residual + attn_out
            else:
                attn_out, _ = self.self_attn(
                    hidden_states=u_head, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
                hidden_states = residual + attn_out + v

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        layer.forward = _forward_with_multiroute_rosa.__get__(layer, Qwen3DecoderLayer)

    meta = {
        "apply_layers_from": 1,
        "num_routes_per_layer": M,
        "vocab_per_route": K,
        "inject_mode": inject_mode,
        "route_dim": d,
        "route_slices": route_slices,
    }
    with open(os.path.join(OUTPUT_DIR, ""), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

import inspect

def patch_layer0_with_input_rosa(model: Qwen3ForCausalLM):
    import inspect, torch, torch.nn as nn
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    if _ROSA_L0_METHOD not in ("", "", ""):
        print(f"")
        return

    qwen = model.model
    V = qwen.embed_tokens.weight.shape[0]
    H = qwen.embed_tokens.weight.shape[1]
    base_dtype  = qwen.embed_tokens.weight.dtype
    base_device = qwen.embed_tokens.weight.device

    if not hasattr(qwen, "_orig_forward"):
        qwen._orig_forward = qwen.forward
        def _forward_with_cache(self, *args, **kwargs):
            ba = inspect.signature(self._orig_forward).bind(*args, **kwargs)
            ba.apply_defaults()
            self._cached_input_ids = ba.arguments.get("input_ids", None)
            self._cached_attention_mask = ba.arguments.get("attention_mask", None)
            return self._orig_forward(*args, **kwargs)
        qwen.forward = _forward_with_cache.__get__(qwen, type(qwen))

    if not hasattr(qwen, "rosa_input_emb"):
        qwen.rosa_input_emb = nn.Embedding(V + 1, H, padding_idx=0).to(dtype=base_dtype, device=base_device)
        nn.init.normal_(qwen.rosa_input_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            qwen.rosa_input_emb.weight.data[0].zero_()

    layer0: Qwen3DecoderLayer = qwen.layers[0]
    orig_forward_l0 = layer0.forward
    parent = qwen
    rosa_input_emb = qwen.rosa_input_emb

    def _forward_layer0_with_input_rosa(
        self: Qwen3DecoderLayer,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        past_key_values=None,
        use_cache: bool = False,
        cache_position: torch.LongTensor = None,
        position_embeddings=None,
        **kwargs
    ):
        mode = _ROSA_L0_METHOD
        if mode == "":
            return orig_forward_l0(
                hidden_states, attention_mask, position_ids,
                past_key_values, use_cache, cache_position,
                position_embeddings, **kwargs
            )

        input_ids = getattr(parent, "_cached_input_ids", None)
        attn_mask_ext = getattr(parent, "_cached_attention_mask", None)

        if input_ids is not None:
            with torch.no_grad():
                r_idx = _rosa_tokens_nextdiff_batch(input_ids)
            r_pad = torch.where(r_idx >= 0, r_idx + 1, torch.zeros_like(r_idx))
            xr = rosa_input_emb(r_pad.to(device=hidden_states.device))
            if attn_mask_ext is not None:
                xr = xr * attn_mask_ext.to(device=hidden_states.device, dtype=xr.dtype).unsqueeze(-1)
        else:
            xr = torch.zeros_like(hidden_states)

        if mode == "":
            hidden_states = hidden_states + xr
            residual = hidden_states
            u_head = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(
                hidden_states=u_head, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            hidden_states = residual + attn_out
        else:
            residual = hidden_states
            u_head = self.input_layernorm(hidden_states)
            attn_out, _ = self.self_attn(
                hidden_states=u_head, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=use_cache, cache_position=cache_position,
                position_embeddings=position_embeddings, **kwargs,
            )
            hidden_states = residual + attn_out + xr

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    layer0.forward = _forward_layer0_with_input_rosa.__get__(layer0, type(layer0))
    print(f"")

def build_model_and_tokenizer() -> Tuple[Qwen3ForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(MODEL_LOCAL_DIR)
    config.sliding_window = ATTN_WINDOW
    config.max_window_layers = FIRST_GLOBAL_LAYERS
    if (not hasattr(config, "layer_types")) or (config.layer_types is None):
        config.layer_types = [
            "" if i < config.max_window_layers else ""
            for i in range(config.num_hidden_layers)
        ]
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "" if USE_FLASH_ATTN else ""
    else:
        config._attn_implementation = "" if USE_FLASH_ATTN else ""

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, use_fast=True)
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
    patch_layer0_with_input_rosa(model)

    return model, tokenizer

def save_rosa_only(model: Qwen3ForCausalLM, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = {}

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "rosa_wlm_list"):
            for m, head in enumerate(layer.rosa_wlm_list):
                state[f"model.layers.{i}.rosa_wlm_list.{m}.weight"] = head.weight.detach().cpu()
        if hasattr(layer, "rosa_emb_list"):
            for m, emb in enumerate(layer.rosa_emb_list):
                state[f"model.layers.{i}.rosa_emb_list.{m}.weight"] = emb.weight.detach().cpu()
        if hasattr(layer, "rosa_alpha"):
            state[f"model.layers.{i}.rosa_alpha"] = layer.rosa_alpha.detach().cpu()
    if hasattr(model.model, "rosa_input_emb"):
        state["model.rosa_input_emb.weight"] = model.model.rosa_input_emb.weight.detach().cpu()

    path = os.path.join(out_dir, SAVE_STATE_DICT_NAME)
    torch.save(state, path)
    print(f"")

def build_optimizer_params(model):
    gate_params, wlm_params, emb_params, backbone_params = [], [], [], []
    l0_params = []
    param_groups = []
    for n, p in model.named_parameters():
        if "rosa_wlm_list" in n:             wlm_params.append(p)
        elif "rosa_emb_list" in n:           emb_params.append(p)
        elif "rosa_input_emb" in n:          l0_params.append(p)
        else:                                backbone_params.append(p)

    if l0_params and _ROSA_L0_TRAIN:
        param_groups.append({"params": l0_params, "lr": LR_ROSA, "weight_decay": 0.0})

    if wlm_params:
        param_groups.append({"params": wlm_params, "lr": LR_ROSA, "weight_decay": WEIGHT_DECAY})
    if emb_params:
        param_groups.append({"params": emb_params, "lr": LR_ROSA, "weight_decay": 0.0})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": LR_ROSA, "weight_decay": 0.0})

    if LR_BACKBONE and LR_BACKBONE > 0.0:
        no_decay, has_decay = [], []
        for n, p in model.named_parameters():
            if "rosa_" in n:
                continue
            if any(k in n.lower() for k in ["", "", "", ""]):
                no_decay.append(p)
            else:
                has_decay.append(p)

        if has_decay:
            param_groups.append({"params": has_decay, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY})
        if no_decay:
            param_groups.append({"params": no_decay, "lr": LR_BACKBONE, "weight_decay": 0.0})
    else:
        for p in backbone_params:
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
                    if hasattr(layer, "rosa_emb_list"):
                        for emb in layer.rosa_emb_list:
                            emb.weight.data[0].zero_()
                if hasattr(model.model, "rosa_input_emb"):
                    model.model.rosa_input_emb.weight.data[0].zero_()
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            with torch.no_grad():
                for layer in model.model.layers:
                    if hasattr(layer, "rosa_emb_list"):
                        for emb in layer.rosa_emb_list:
                            emb.weight.data[0].zero_()
                if hasattr(model.model, "rosa_input_emb"):
                    model.model.rosa_input_emb.weight.data[0].zero_()
        return control

def main():
    set_seed(SEED)

    raw = load_from_disk(DATASET_DIR)
    train_ds = raw["train"]
    test_ds = raw.get("test", raw["validation"] if "validation" in raw else None)
    assert test_ds is not None

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
        save_strategy="",
        report_to="",
        fp16=(not BF16) and torch.cuda.is_available(),
        bf16=BF16,
        dataloader_num_workers=2,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        optim="",
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

    print(f"")

    trainer.train()
    metrics = trainer.evaluate()

    meta = {
        "model_local_dir": MODEL_LOCAL_DIR,
        "dataset_dir": DATASET_DIR,
        "seq_len": SEQ_LEN,
        "attn_window": ATTN_WINDOW,
        "first_global_layers": FIRST_GLOBAL_LAYERS,
        "rosa": {
            "num_routes": ROSA_NUM_ROUTES,
            "vocab_size": ROSA_VOCAB_SIZE,
            "lcg_topk": LCG_TOPK,
            "pos_subsample": LCG_POS_SUBSAMPLE,
            "lr_rosa": LR_ROSA, "lr_backbone": LR_BACKBONE,
            "k_per_layer": {str(i): (getattr(model.model.layers[i], "rosa_vocab_size", 0))
                            for i in range(model.config.num_hidden_layers)}
        },
        "metrics": metrics,
        "time": time.asctime(),
    }

    if is_main_process():
        print("", metrics)
        save_rosa_only(model, OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, ""), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"")

if __name__ == "__main__":
    main()
