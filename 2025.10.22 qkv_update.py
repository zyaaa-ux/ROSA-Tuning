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
BF16 = True if torch.cuda.is_available() else False

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

torch.set_num_threads(_PER_RANK_CPUS if isinstance(_PER_RANK_CPUS, int) else 0)

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
    def _compute_k_runcap_bmt_nb(k_run_start_bmt: _np.ndarray,
                                 k_c_len_bm: _np.ndarray,
                                 T: int) -> _np.ndarray:
        B, M, _ = k_run_start_bmt.shape
        rcap = _np.full((B, M, T), -1, dtype=_np.int32)
        for b in _nb.prange(B):
            for m in range(M):
                clen = int(k_c_len_bm[b, m])
                if clen <= 0:
                    continue
                curr = 0
                rs = k_run_start_bmt[b, m]
                for t in range(T):
                    while (curr + 1) < clen:
                        nxt = rs[curr + 1]
                        if nxt != -1 and nxt <= t:
                            curr += 1
                        else:
                            break
                    rcap[b, m, t] = curr
        return rcap

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_new_state_nb(next_arr, link, length, e, size, L):
        s = size[0]
        size[0] = s + 1
        length[s] = L
        link[s]   = -1
        e[s]      = -1
        K = next_arr.shape[1]
        for j in range(K):
            next_arr[s, j] = -1
        return s

    @(_nb.njit(cache=True, fastmath=False))
    def _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, pos):
        last = last_arr[0]
        cur = _sam_new_state_nb(next_arr, link, length, e, size, length[last] + 1)
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
                clone = _sam_new_state_nb(next_arr, link, length, e, size, length[p] + 1)
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

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False))
    def _k_sam_seq_nb(z, K: int, S_cap: int):
        T = z.shape[0]
        next_arr = _np.empty((S_cap, K), dtype=_np.int32)
        link     = _np.empty((S_cap,),   dtype=_np.int32)
        length   = _np.empty((S_cap,),   dtype=_np.int32)
        e        = _np.empty((S_cap,),   dtype=_np.int32)
        for j in range(K): next_arr[0, j] = -1
        link[0] = -1; length[0] = 0; e[0] = -1
        size = _np.empty((1,), dtype=_np.int32); size[0] = 1
        last_arr = _np.empty((1,), dtype=_np.int32); last_arr[0] = 0
        c = _np.empty((T,), dtype=_np.int32)
        run_start = _np.full((T,), -1, dtype=_np.int32)
        c_len = 0
        last_sym = -2147483647
        for t in range(T):
            x = z[t]
            if t == 0 or x != last_sym:
                c[c_len] = x
                run_start[c_len] = t
                c_len += 1
                _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, c_len - 1)
                last_sym = x
        S_use = size[0]
        dfa = _np.empty((S_cap, K), dtype=_np.int32)
        for j in range(K):
            dfa[0, j] = next_arr[0, j]
        for s in range(1, S_use):
            par = link[s]
            for j in range(K):
                v = next_arr[s, j]
                if v != -1:
                    dfa[s, j] = v
                else:
                    dfa[s, j] = dfa[par, j] if par != -1 else -1
        for s in range(S_use, S_cap):
            for j in range(K):
                dfa[s, j] = -1
        return dfa, e, run_start, _np.int32(c_len), c

def _k_sam_batch_btm_with_ws_py(z_btm_np: "_np.ndarray", K: int):
    B, T, M = z_btm_np.shape
    S_cap = 2 * T + 5
    dfa_bmsk      = _np.empty((B, M, S_cap, K), dtype=_np.int32)
    e_bms         = _np.empty((B, M, S_cap),     dtype=_np.int32)
    run_start_bmt = _np.full((B, M, T), -1,      dtype=_np.int32)
    c_len_bm      = _np.empty((B, M),            dtype=_np.int32)
    run_sym_bmt   = _np.full((B, M, T), -1,      dtype=_np.int32)
    for b in range(B):
        for m in range(M):
            z = [int(v) for v in list(z_btm_np[b, :, m])]
            sam = _SAMFoldedCPU(max_states=S_cap, K=K)
            last_sym = None
            run_starts = []
            for t, x in enumerate(z):
                if (last_sym is None) or (x != last_sym):
                    sam.extend_run(int(x))
                    run_starts.append(t)
                    last_sym = x
            S_use = sam.size
            dfa = _np.empty((S_cap, K), dtype=_np.int32)
            for j in range(K):
                dfa[0, j] = sam.next[0, j]
            for s in range(1, S_use):
                par = sam.link[s]
                for j in range(K):
                    v = sam.next[s, j]
                    dfa[s, j] = v if v != -1 else (dfa[par, j] if par != -1 else -1)
            for s in range(S_use, S_cap):
                for j in range(K):
                    dfa[s, j] = -1
            dfa_bmsk[b, m, :, :]  = dfa
            e_bms[b, m, :]        = sam.e
            c_len_bm[b, m]        = len(sam.c)
            run_start_bmt[b, m, :] = -1
            for i in range(min(T, len(run_starts))):
                run_start_bmt[b, m, i] = run_starts[i]
            run_sym_bmt[b, m, :] = -1
            for i in range(min(T, len(sam.c))):
                run_sym_bmt[b, m, i] = sam.c[i]
    return dfa_bmsk, e_bms, run_start_bmt, c_len_bm, run_sym_bmt

def _k_sam_batch_btm_with_ws(z_btm_np: "_np.ndarray", K: int):
    if _NUMBA_OK and ('_k_sam_seq_nb' in globals()):
        B, T, M = z_btm_np.shape
        S_cap = 2 * T + 5
        dfa_bmsk      = _np.empty((B, M, S_cap, K), dtype=_np.int32)
        e_bms         = _np.empty((B, M, S_cap),     dtype=_np.int32)
        run_start_bmt = _np.full((B, M, T), -1,      dtype=_np.int32)
        c_len_bm      = _np.empty((B, M),            dtype=_np.int32)
        run_sym_bmt   = _np.full((B, M, T), -1,      dtype=_np.int32)
        if _PARALLEL:
            for idx in _nb.prange(B * M):
                b = idx // M
                m = idx % M
                dfa, e, rs, clen, cseq = _k_sam_seq_nb(z_btm_np[b, :, m], int(K), S_cap)
                dfa_bmsk[b, m, :, :]   = dfa
                e_bms[b, m, :]         = e
                c_len_bm[b, m]         = clen
                run_start_bmt[b, m, :] = -1
                for i in range(min(T, clen)):
                    run_start_bmt[b, m, i] = rs[i]
                run_sym_bmt[b, m, :]   = -1
                for i in range(min(T, clen)):
                    run_sym_bmt[b, m, i] = cseq[i]
        else:
            for b in range(B):
                for m in range(M):
                    dfa, e, rs, clen, cseq = _k_sam_seq_nb(z_btm_np[b, :, m], int(K), S_cap)
                    dfa_bmsk[b, m, :, :]   = dfa
                    e_bms[b, m, :]         = e
                    c_len_bm[b, m]         = clen
                    run_start_bmt[b, m, :] = -1
                    for i in range(min(T, clen)):
                        run_start_bmt[b, m, i] = rs[i]
                    run_sym_bmt[b, m, :]   = -1
                    for i in range(min(T, clen)):
                        run_sym_bmt[b, m, i] = cseq[i]
        return dfa_bmsk, e_bms, run_start_bmt, c_len_bm, run_sym_bmt
    return _k_sam_batch_btm_with_ws_py(z_btm_np, int(K))

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_match_next_from_nb(next_arr, link, s, x):
        p = s
        while p != -1 and next_arr[p, x] == -1:
            p = link[p]
        return -1 if p == -1 else next_arr[p, x]
else:
    def _sam_match_next_from_nb(next_arr, link, s, x):
        p = int(s)
        while p != -1 and next_arr[p, x] == -1:
            p = link[p]
        return -1 if p == -1 else int(next_arr[p, x])

def _q_prefix_scan_runs_py(
    q_btm: _np.ndarray,
    k_run_sym_bmt: _np.ndarray,
    k_run_start_bmt: _np.ndarray,
    k_c_len_bm: _np.ndarray,
    k_runcap_bmt: _np.ndarray,
    K_qk: int
):
    B, T, M = q_btm.shape
    tau_time_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
    q_run_id_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
    s_before_bmr   = _np.full((B, M, T), -1, dtype=_np.int32)
    rq_len_bm      = _np.zeros((B, M), dtype=_np.int32)
    r_cf_run_bmrk  = _np.full((B, M, T, K_qk), -1, dtype=_np.int32)
    for b in range(B):
        for m in range(M):
            c     = k_run_sym_bmt[b, m]
            rsK   = k_run_start_bmt[b, m]
            clen  = int(k_c_len_bm[b, m])
            rcap  = k_runcap_bmt[b, m]
            S_cap = 2 * T + 5
            next_arr = _np.full((S_cap, K_qk), -1, dtype=_np.int32)
            link     = _np.full((S_cap,),     -1, dtype=_np.int32)
            length   = _np.zeros((S_cap,),        dtype=_np.int32)
            e        = _np.full((S_cap,),     -1, dtype=_np.int32)
            size     = _np.zeros((1,),            dtype=_np.int32); size[0] = 1
            last_arr = _np.zeros((1,),            dtype=_np.int32); last_arr[0] = 0
            def _extend_to(r_target):
                nonlocal r_ext
                while r_ext < r_target and (r_ext + 1) < clen:
                    r_ext += 1
                    x = int(c[r_ext])
                    _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, r_ext)
            rq = 0
            run_beg = _np.empty((T,), dtype=_np.int32)
            run_sym = _np.empty((T,), dtype=_np.int32)
            last = -2147483647
            for t in range(T):
                x = int(q_btm[b, t, m])
                if t == 0 or x != last:
                    run_beg[rq] = t
                    run_sym[rq] = x
                    rq += 1
                    last = x
                q_run_id_bmt[b, m, t] = rq - 1
            rq_len_bm[b, m] = rq
            s = 0
            r_ext = -1
            cur_run = 0
            advanced_in_run = False
            for t in range(T):
                if cur_run < rq and t == run_beg[cur_run]:
                    cap_t = int(rcap[t])
                    if cap_t >= 0:
                        _extend_to(cap_t)
                    s_before_bmr[b, m, cur_run] = s
                    for a in range(K_qk):
                        ns = _sam_match_next_from_nb(next_arr, link, s, a)
                        if ns != -1:
                            rpos = int(e[ns]); nxt = rpos + 1
                            if 0 <= rpos and nxt < (r_ext + 1):
                                r_cf_run_bmrk[b, m, cur_run, a] = nxt
                            else:
                                r_cf_run_bmrk[b, m, cur_run, a] = -1
                        else:
                            r_cf_run_bmrk[b, m, cur_run, a] = -1
                    advanced_in_run = False
                cap_t = int(rcap[t])
                if cap_t >= 0:
                    _extend_to(cap_t)
                x = int(q_btm[b, t, m])
                ns = _sam_match_next_from_nb(next_arr, link, s, x)
                if ns != -1:
                    rpos = int(e[ns]); nxt = rpos + 1
                    if 0 <= rpos and nxt < (r_ext + 1):
                        tau_time_bmt[b, m, t] = rsK[nxt]
                    else:
                        tau_time_bmt[b, m, t] = -1
                    if not advanced_in_run:
                        s = ns
                        advanced_in_run = True
                else:
                    tau_time_bmt[b, m, t] = -1
                if cur_run + 1 < rq and (t + 1) == run_beg[cur_run + 1]:
                    cur_run += 1
    return tau_time_bmt, q_run_id_bmt, s_before_bmr, rq_len_bm, r_cf_run_bmrk

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False))
    def _q_prefix_scan_runs_nb(
        q_btm: _np.ndarray,
        k_run_sym_bmt: _np.ndarray,
        k_run_start_bmt: _np.ndarray,
        k_c_len_bm: _np.ndarray,
        k_runcap_bmt: _np.ndarray,
        K_qk: int
    ):
        B, T, M = q_btm.shape
        tau_time_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
        q_run_id_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
        s_before_bmr   = _np.full((B, M, T), -1, dtype=_np.int32)
        rq_len_bm      = _np.zeros((B, M), dtype=_np.int32)
        r_cf_run_bmrk  = _np.full((B, M, T, K_qk), -1, dtype=_np.int32)
        for b in _nb.prange(B):
            for m in range(M):
                c     = k_run_sym_bmt[b, m]
                rsK   = k_run_start_bmt[b, m]
                clen  = int(k_c_len_bm[b, m])
                rcap  = k_runcap_bmt[b, m]
                S_cap = 2 * T + 5
                next_arr = _np.full((S_cap, K_qk), -1, dtype=_np.int32)
                link     = _np.full((S_cap,),     -1, dtype=_np.int32)
                length   = _np.zeros((S_cap,),        dtype=_np.int32)
                e        = _np.full((S_cap,),     -1, dtype=_np.int32)
                size     = _np.zeros((1,),            dtype=_np.int32); size[0] = 1
                last_arr = _np.zeros((1,),            dtype=_np.int32); last_arr[0] = 0
                r_ext = -1
                s = 0
                rq = 0
                run_beg = _np.empty((T,), dtype=_np.int32)
                run_sym = _np.empty((T,), dtype=_np.int32)
                last = -2147483647
                for t in range(T):
                    x = int(q_btm[b, t, m])
                    if t == 0 or x != last:
                        run_beg[rq] = t
                        run_sym[rq] = x
                        rq += 1
                        last = x
                    q_run_id_bmt[b, m, t] = rq - 1
                rq_len_bm[b, m] = rq
                def _extend_to(r_target):
                    nonlocal r_ext
                    while r_ext < r_target and (r_ext + 1) < clen:
                        r_ext += 1
                        x = int(c[r_ext])
                        _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, r_ext)
                cur_run = 0
                advanced_in_run = False
                for t in range(T):
                    if cur_run < rq and t == run_beg[cur_run]:
                        cap_t = int(rcap[t])
                        if cap_t >= 0:
                            _extend_to(cap_t)
                        s_before_bmr[b, m, cur_run] = s
                        for a in range(K_qk):
                            ns = _sam_match_next_from_nb(next_arr, link, s, a)
                            if ns != -1:
                                rpos = int(e[ns]); nxt = rpos + 1
                                if 0 <= rpos and nxt < (r_ext + 1):
                                    r_cf_run_bmrk[b, m, cur_run, a] = nxt
                                else:
                                    r_cf_run_bmrk[b, m, cur_run, a] = -1
                            else:
                                r_cf_run_bmrk[b, m, cur_run, a] = -1
                        advanced_in_run = False
                    cap_t = int(rcap[t])
                    if cap_t >= 0:
                        _extend_to(cap_t)
                    x = int(q_btm[b, t, m])
                    ns = _sam_match_next_from_nb(next_arr, link, s, x)
                    if ns != -1:
                        rpos = int(e[ns]); nxt = rpos + 1
                        if 0 <= rpos and nxt < (r_ext + 1):
                            tau_time_bmt[b, m, t] = rsK[nxt]
                        else:
                            tau_time_bmt[b, m, t] = -1
                        if not advanced_in_run:
                            s = ns
                            advanced_in_run = True
                    else:
                        tau_time_bmt[b, m, t] = -1
                    if cur_run + 1 < rq and (t + 1) == run_beg[cur_run + 1]:
                        cur_run += 1
        return tau_time_bmt, q_run_id_bmt, s_before_bmr, rq_len_bm, r_cf_run_bmrk

def _q_runs_pipeline_cpu(
    q_btm_torch: torch.Tensor,
    k_run_sym_np: "_np.ndarray",
    k_runstart_np: "_np.ndarray",
    k_clen_np: "_np.ndarray",
    K_qk: int
):
    B, T, M = q_btm_torch.shape
    q_host = _PINNED_POOL.get("q_host_runs", (B, T, M), dtype=torch.int32)
    q_host.copy_(q_btm_torch, non_blocking=True)
    ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
    def _wait_and_run():
        _wait_event(ev)
        q_np = _np.asarray(q_host, order="C")
        k_runcap_np = _compute_k_runcap_bmt_nb(
            k_runstart_np.astype(_np.int32, copy=False),
            k_clen_np.astype(_np.int32, copy=False),
            int(T)
        )
        if _NUMBA_OK and ('_q_prefix_scan_runs_nb' in globals()):
            tau_time_np, q_run_id_np, s_before_np, rq_len_np, r_cf_run_np = _q_prefix_scan_runs_nb(
                q_np.astype(_np.int32, copy=False),
                k_run_sym_np.astype(_np.int32, copy=False),
                k_runstart_np.astype(_np.int32, copy=False),
                k_clen_np.astype(_np.int32, copy=False),
                k_runcap_np.astype(_np.int32, copy=False),
                int(K_qk)
            )
        else:
            tau_time_np, q_run_id_np, s_before_np, rq_len_np, r_cf_run_np = _q_prefix_scan_runs_py(
                q_np.astype(_np.int32, copy=False),
                k_run_sym_np.astype(_np.int32, copy=False),
                k_runstart_np.astype(_np.int32, copy=False),
                k_clen_np.astype(_np.int32, copy=False),
                k_runcap_np.astype(_np.int32, copy=False),
                int(K_qk)
            )
        return tau_time_np, q_run_id_np, r_cf_run_np
    pool = _get_rosa_thread_pool()
    if pool is None:
        return _ImmediateFuture(_wait_and_run())
    else:
        return pool.submit(_wait_and_run)

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

class MultiRouteQKVLcgFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_in: torch.Tensor,
        logits_q_all: torch.Tensor,
        logits_k_all: torch.Tensor,
        logits_v_all: torch.Tensor,
        E_v_compact: torch.Tensor,
        k_run_start_bmt: torch.Tensor,
        tau_time_bmt: torch.Tensor,
        q_run_id_bmt: torch.Tensor,
        r_cf_run_bmrk: torch.Tensor,
        pos_mask_cpu: Optional[list],
        route_slices: list,
    ):
        ctx.save_for_backward(
            logits_q_all, logits_k_all, logits_v_all,
            E_v_compact, k_run_start_bmt,
            tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk
        )
        ctx.pos_mask_cpu = pos_mask_cpu
        ctx.route_slices = route_slices
        return v_in

    @staticmethod
    def backward(ctx, grad_v: torch.Tensor):
        (
            logits_q_all, logits_k_all, logits_v_all,
            E_v_compact, k_run_start_bmt,
            tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk
        ) = ctx.saved_tensors

        device = logits_q_all.device
        B, T, M, K_qk = logits_q_all.shape
        K_v = logits_v_all.size(-1)

        with torch.no_grad():
            if ctx.pos_mask_cpu is None:
                pos_mask_bt = torch.ones((B, T), device=device, dtype=torch.bool)
            else:
                pos_mask_bt = torch.tensor(ctx.pos_mask_cpu, device=device, dtype=torch.bool)
        pos_mask_bmt = pos_mask_bt.unsqueeze(1).expand(B, M, T)

        route_slices = ctx.route_slices
        with torch.no_grad():
            S_list = []
            for m, (s,e) in enumerate(route_slices):
                Sm = torch.einsum("btd,kd->btk", grad_v[:, :, s:e].float(), E_v_compact[m].float())
                S_list.append(Sm)
            S_plus = torch.stack(S_list, dim=1)
            S_no_pad = S_plus[..., 1:]

        grad_logits_v = torch.zeros_like(logits_v_all, dtype=torch.float32)
        with torch.no_grad():
            p_v = torch.softmax(logits_v_all.float(), dim=-1)
            p_v_bmtk = p_v.permute(0,2,1,3).contiguous()
            tau = tau_time_bmt.to(torch.long)
            valid_tau = (tau >= 0) & pos_mask_bmt
            idx_time = torch.clamp(tau, 0, T-1).unsqueeze(-1).expand(B, M, T, K_v)
            p_v_tau  = torch.gather(p_v_bmtk, dim=2, index=idx_time)
            S_tau    = S_no_pad
            mask = valid_tau.unsqueeze(-1)
            p_v_tau = torch.where(mask, p_v_tau, torch.zeros_like(p_v_tau))
            S_tau   = torch.where(mask, S_tau,   torch.zeros_like(S_tau))
            mu = (p_v_tau * S_tau).sum(dim=-1, keepdim=True)
            g_v_tau = p_v_tau * (S_tau - mu)
            grad_v_bmtk = torch.zeros((B, M, T, K_v), dtype=g_v_tau.dtype, device=device)
            grad_v_bmtk.scatter_add_(dim=2, index=idx_time, src=g_v_tau)
            grad_logits_v = grad_v_bmtk.permute(0,2,1,3).contiguous()

        grad_logits_q = torch.zeros_like(logits_q_all, dtype=torch.float32)
        with torch.no_grad():
            p_q = torch.softmax(logits_q_all.float(), dim=-1)
            p_q_bmtk = p_q.permute(0,2,1,3).contiguous()
            run_id = torch.clamp(q_run_id_bmt.to(torch.long), 0, r_cf_run_bmrk.size(2)-1)
            r_cf_bmtk = torch.gather(
                r_cf_run_bmrk.to(torch.long), dim=2,
                index=run_id.unsqueeze(-1).expand(-1,-1,-1,K_qk)
            )
            valid_cf = (r_cf_bmtk >= 0) & pos_mask_bmt.unsqueeze(-1)
            Rmax = k_run_start_bmt.size(2)
            r_idx = torch.clamp(r_cf_bmtk, 0, Rmax - 1)
            tau_cf = torch.take_along_dim(
                k_run_start_bmt.to(torch.long).unsqueeze(2).expand(B,M,T,Rmax),
                r_idx, dim=3
            )
            valid_cf = valid_cf & (tau_cf >= 0)
            v_idx_all = torch.argmax(logits_v_all, dim=-1).to(torch.long)
            v_idx_bmt = v_idx_all.permute(0,2,1).contiguous()
            idx_tau_cf = torch.clamp(tau_cf, 0, T-1)
            v_idx_cf = torch.gather(v_idx_bmt.unsqueeze(-1).expand(-1,-1,-1,K_qk), dim=2, index=idx_tau_cf)
            v_row_cf = torch.where(valid_cf, v_idx_cf + 1, torch.zeros_like(v_idx_cf))
            f_take = torch.gather(S_plus, dim=-1, index=v_row_cf)
            f_take = torch.where(valid_cf, f_take, torch.zeros_like(f_take))
            mu_q = (p_q_bmtk * f_take).sum(dim=-1, keepdim=True)
            g_q  = p_q_bmtk * (f_take - mu_q)
            grad_logits_q = g_q.permute(0,2,1,3).contiguous()

        grad_logits_k = torch.zeros_like(logits_k_all, dtype=torch.float32)
        with torch.no_grad():
            v_idx_all = torch.argmax(logits_v_all, dim=-1).to(torch.long)
            v_idx_bmt = v_idx_all.permute(0,2,1).contiguous()
            t_i = k_run_start_bmt.to(torch.long)
            run_valid = (t_i >= 0)
            t_i_clamp = torch.clamp(t_i, 0, T-1)
            v_idx_run = torch.gather(v_idx_bmt, dim=2, index=t_i_clamp)
            v_row_run = torch.where(run_valid, v_idx_run + 1, torch.zeros_like(v_idx_run))
            Rmax = v_row_run.size(2)
            r_cf_bmtk = torch.gather(
                r_cf_run_bmrk.to(torch.long), dim=2,
                index=torch.clamp(q_run_id_bmt.to(torch.long), 0, r_cf_run_bmrk.size(2)-1).unsqueeze(-1).expand(-1,-1,-1,K_qk)
            )
            valid_cf = (r_cf_bmtk >= 0) & pos_mask_bmt.unsqueeze(-1)
            r_idx = torch.clamp(r_cf_bmtk, 0, Rmax - 1)
            I = torch.take_along_dim(v_row_run.unsqueeze(2).expand(B,M,T,Rmax), r_idx, dim=3)
            I = torch.where(valid_cf, I, torch.zeros_like(I))
            S_plus_alias = S_plus
            f_new = torch.gather(S_plus_alias, dim=-1, index=I.to(torch.long))
            f_new = torch.where(valid_cf, f_new, torch.zeros_like(f_new))
            N = B * M
            U = torch.zeros((N, Rmax, K_qk), dtype=f_new.dtype, device=device)
            src = f_new.view(N, T, K_qk)
            idx = torch.where(valid_cf, r_cf_bmtk, torch.zeros_like(r_cf_bmtk)).view(N, T, K_qk).to(torch.long)
            U.scatter_add_(dim=1, index=idx, src=src)
            U = U.view(B, M, Rmax, K_qk)
            logits_k_bmtk = logits_k_all.permute(0,2,1,3).contiguous()
            logits_run = torch.gather(
                logits_k_bmtk, dim=2, index=t_i_clamp.unsqueeze(-1).expand(B,M,Rmax,K_qk)
            )
            p_run = torch.softmax(logits_run.float(), dim=-1)
            mu_run = (p_run * U).sum(dim=-1, keepdim=True)
            g_run  = p_run * (U - mu_run)
            g_run  = torch.where(run_valid.unsqueeze(-1), g_run, torch.zeros_like(g_run))
            grad_k_bmtk = torch.zeros((B,M,T,K_qk), dtype=g_run.dtype, device=device)
            grad_k_bmtk.scatter_add_(dim=2, index=t_i_clamp.unsqueeze(-1).expand(B,M,Rmax,K_qk), src=g_run)
            grad_logits_k = grad_k_bmtk.permute(0,2,1,3).contiguous()

        return (
            grad_v,
            grad_logits_q.to(logits_q_all.dtype),
            grad_logits_k.to(logits_k_all.dtype),
            grad_logits_v.to(logits_v_all.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )

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
    import os
    import torch
    import torch.nn as nn
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    inject_mode = os.environ.get("ROSA_INJECT_MODE", globals().get("ROSA_INJECT_MODE", "")).lower()
    assert inject_mode in ("pre_attn", "post_attn", "")
    M: int    = int(globals().get("ROSA_NUM_ROUTES")) if globals().get("ROSA_NUM_ROUTES") is not None else 0
    K_qk: int = int(globals().get("ROSA_QK_VOCAB_SIZE"))
    K_v:  int = int(globals().get("ROSA_V_VOCAB_SIZE"))
    PER_ROUTE_DIM = globals().get("ROSA_PER_ROUTE_DIM", None)
    base_param = model.model.embed_tokens.weight
    base_dtype = base_param.dtype
    base_device = base_param.device
    hidden_size: int = model.config.hidden_size
    if PER_ROUTE_DIM is None and M:
        assert hidden_size % M == 0
        d = hidden_size // M
    else:
        d = int(PER_ROUTE_DIM) if PER_ROUTE_DIM is not None else 0
        if M and d:
            assert d * M == hidden_size
    route_slices = [(i * d, (i + 1) * d) for i in range(M)] if M and d else []
    for li, layer in enumerate(model.model.layers):
        if li == 0:
            continue
        layer.rosa_wlm_q_list = nn.ModuleList([nn.Linear(d, K_qk, bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)])
        layer.rosa_wlm_k_list = nn.ModuleList([nn.Linear(d, K_qk, bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)])
        layer.rosa_wlm_v_list = nn.ModuleList([nn.Linear(d, K_v,  bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)])
        for lst in (layer.rosa_wlm_q_list, layer.rosa_wlm_k_list, layer.rosa_wlm_v_list):
            for w in lst:
                nn.init.xavier_uniform_(w.weight)
        layer.rosa_v_emb_list = nn.ModuleList([nn.Embedding(K_v + 1, d, padding_idx=0).to(dtype=base_dtype, device=base_device) for _ in range(M)])
        for emb in layer.rosa_v_emb_list:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                emb.weight.data[0].zero_()
        layer.rosa_alpha = nn.Parameter(torch.zeros(hidden_size, dtype=base_dtype, device=base_device))
        layer.rosa_num_routes     = M
        layer.rosa_qk_vocab_size  = K_qk
        layer.rosa_v_vocab_size   = K_v
        layer.rosa_route_slices   = route_slices
        layer.rosa_inject_mode    = inject_mode
        def _forward_qkv_rosa(
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
            K_qk = self.rosa_qk_vocab_size
            K_v  = self.rosa_v_vocab_size
            device = hidden_states.device
            slices = self.rosa_route_slices
            u_head = self.input_layernorm(hidden_states)
            logits_q_list, logits_k_list, logits_v_list = [], [], []
            for m, (s, e) in enumerate(slices):
                u_slice = u_head[:, :, s:e]
                logits_q_list.append(self.rosa_wlm_q_list[m](u_slice))
                logits_k_list.append(self.rosa_wlm_k_list[m](u_slice))
                logits_v_list.append(self.rosa_wlm_v_list[m](u_slice))
            logits_q_all = torch.stack(logits_q_list, dim=2)
            logits_k_all = torch.stack(logits_k_list, dim=2)
            logits_v_all = torch.stack(logits_v_list, dim=2)
            q_btm = torch.argmax(logits_q_all, dim=-1).to(torch.int32)
            k_btm = torch.argmax(logits_k_all, dim=-1).to(torch.int32)
            v_btm = torch.argmax(logits_v_all, dim=-1).to(torch.int32)
            nvtx.range_push("K_SAM_async")
            k_host = _PINNED_POOL.get("k_host", (B, T, M), dtype=torch.int32)
            k_host.copy_(k_btm, non_blocking=True)
            ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
            def _wait_and_build_k_sam():
                _wait_event(ev)
                z_np = _np.asarray(k_host, order="C")
                return _k_sam_batch_btm_with_ws(z_np, int(K_qk))
            pool = _get_rosa_thread_pool()
            if pool is None:
                k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = _wait_and_build_k_sam()
            else:
                fut_ws = pool.submit(_wait_and_build_k_sam)
                k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = fut_ws.result()
            nvtx.range_pop()
            fut_q = _q_runs_pipeline_cpu(q_btm, k_runsym_np, k_runstart_np, k_clen_np, K_qk)
            tau_time_np, q_run_id_np, r_cf_run_np = fut_q.result()
            k_run_start_bmt = torch.from_numpy(k_runstart_np).to(device=device, dtype=torch.int32)
            tau_time_bmt    = torch.from_numpy(tau_time_np).to(device=device, dtype=torch.int32)
            q_run_id_bmt    = torch.from_numpy(q_run_id_np).to(device=device, dtype=torch.int32)
            r_cf_run_bmrk   = torch.from_numpy(r_cf_run_np).to(device=device, dtype=torch.int32)
            valid_tau = (tau_time_bmt >= 0)
            v_idx_all = v_btm.to(torch.long)
            v_idx_bmt = v_idx_all.permute(0,2,1).contiguous()
            t_tauL = torch.clamp(tau_time_bmt.to(torch.long), 0, T-1)
            v_idx_at_tau = torch.gather(v_idx_bmt, dim=2, index=t_tauL)
            v_parts, E_compact = [], []
            for m, (s, e) in enumerate(slices):
                idx_m_plus = torch.where(valid_tau[:, m, :], v_idx_at_tau[:, m, :] + 1, torch.zeros_like(v_idx_at_tau[:, m, :]))
                v_m = self.rosa_v_emb_list[m](idx_m_plus.to(torch.long))
                v_parts.append(v_m)
                E_compact.append(self.rosa_v_emb_list[m].weight.detach())
            v = torch.cat(v_parts, dim=-1).to(u_head.dtype)
            E_v_compact = torch.stack(E_compact, dim=0)
            if os.environ.get("LCG_ENABLE", "1").lower() in ("1", "true"):
                pos_mask_cpu = None
                if LCG_POS_SUBSAMPLE and LCG_POS_SUBSAMPLE < 1.0:
                    mask = (torch.rand((B, T), device=device) < LCG_POS_SUBSAMPLE)
                    pos_mask_cpu = mask.detach().cpu().tolist()
                v = MultiRouteQKVLcgFunction.apply(
                    v, logits_q_all, logits_k_all, logits_v_all, E_v_compact,
                    k_run_start_bmt,
                    tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk,
                    pos_mask_cpu, slices
                )
            if self.rosa_inject_mode == "pre_attn":
                alpha = torch.sigmoid(self.rosa_alpha).view(1, 1, -1).to(dtype=hidden_states.dtype, device=device)
                mix = (1.0 - alpha) * hidden_states + alpha * v
                u_attn = self.input_layernorm(mix)
                attn_out, _ = self.self_attn(
                    hidden_states=u_attn, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
                hidden_states = residual + attn_out
            elif self.rosa_inject_mode == "post_attn":
                u_attn = u_head
                attn_out, _ = self.self_attn(
                    hidden_states=u_attn, attention_mask=attention_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
                post_mix = attn_out + v
                hidden_states = residual + post_mix
            else:
                raise ValueError(f"Unsupported ROSA inject mode: {self.rosa_inject_mode}")
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states
        layer.forward = _forward_qkv_rosa.__get__(layer, Qwen3DecoderLayer)
    meta = {
        "apply_layers_from": 1,
        "num_routes_per_layer": M,
        "qk_vocab_per_route": K_qk,
        "v_vocab_per_route": K_v,
        "inject_mode": inject_mode,
        "route_dim": d,
        "route_slices": route_slices,
        "qkv_heads": True,
    }
    with open(os.path.join(OUTPUT_DIR, "rosa_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def build_model_and_tokenizer() -> Tuple[Qwen3ForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(MODEL_LOCAL_DIR)
    config.sliding_window = ATTN_WINDOW
    config.max_window_layers = FIRST_GLOBAL_LAYERS
    if (not hasattr(config, "layer_types")) or (config.layer_types is None):
        config.layer_types = [
            "full_attention" if i < config.max_window_layers else "sliding_attention"
            for i in range(config.num_hidden_layers)
        ]
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "sdpa"
    else:
        config._attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "sdpa"
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
    print(f"[save] saved ROSA-only params to: {path}")

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
            "pos_subsample": float(globals().get("LCG_POS_SUBSAMPLE", 1.0)) if globals().get("LCG_POS_SUBSAMPLE") is not None else 1.0,
            "lr_rosa": float(globals().get("LR_ROSA", 0.0)) if globals().get("LR_ROSA") is not None else 0.0,
            "lr_backbone": float(globals().get("LR_BACKBONE", 0.0)) if globals().get("LR_BACKBONE") is not None else 0.0,
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
    assert test_ds is not None, "A test or validation split is required"
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
    print(f"Trainable parameters: {total_params:,}")
    trainer.train()
    metrics = trainer.evaluate()
    if is_main_process():
        save_results_and_meta(model, metrics, OUTPUT_DIR)

if __name__ == "__main__":
    main()
