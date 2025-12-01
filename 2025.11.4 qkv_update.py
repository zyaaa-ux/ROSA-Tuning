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
from transformers.trainer_callback import TrainerCallback

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

from transformers.optimization import get_cosine_schedule_with_warmup


import os as _os
_os.environ.setdefault("ROSA_USE_NUMBA", "1")
_os.environ.setdefault("ROSA_NUMBA_PARALLEL", "1")
_os.environ.setdefault("LCG_ENABLE", "1")

_os.environ["ROSA_INJECT_MODE"] = "post_attn"

_USE_NUMBA = _os.environ.get("ROSA_USE_NUMBA", "1").lower() not in ("0", "false")
_PARALLEL  = _os.environ.get("ROSA_NUMBA_PARALLEL", "1").lower() not in ("0", "false")

try:
    import numba as _nb
    _NUMBA_OK = _USE_NUMBA
except Exception:
    _NUMBA_OK = False


TIE = True

MODEL_LOCAL_DIR = "/path/to/base/model/"
MODEL_DIR = None
DATASET_DIR     = "/path/to/dataset/"
OUTPUT_DIR      = "/path/to/output/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAVE_STRATEGY = "steps"
SAVE_STEPS = 5000

USE_FLASH_ATTN = True
BF16 = True if torch.cuda.is_available() else False

SEQ_LEN = 53248
ATTN_WINDOW = 2048
FIRST_GLOBAL_LAYERS = 1

LCG_POS_SUBSAMPLE: float = 1.0

ROSA_Q_CONTRAST_USE_V_PROB: int = int(os.environ.get("ROSA_Q_CONTRAST_USE_V_PROB", 1))

BITS_PER_ROUTE: int = int(os.environ.get("ROSA_BITS_PER_ROUTE", 4))
assert BITS_PER_ROUTE >= 1, "ROSA_BITS_PER_ROUTE must be >=1"

BINARY_TEMP_Q: float = float(os.environ.get("BINARY_TEMP_Q", 1.0))
BINARY_TEMP_K: float = float(os.environ.get("BINARY_TEMP_K", 1.0))
BINARY_TEMP_V: float = float(os.environ.get("BINARY_TEMP_V", 1.0))

LCG_POS_SUBSAMPLE: float = float(os.environ.get("LCG_POS_SUBSAMPLE", LCG_POS_SUBSAMPLE))

ROSA_NUMBA_THREADS: int = 64
ROSA_ENABLE_NUMBA_PARALLEL: bool = True
ROSA_THREAD_WORKERS: int = 0

WEIGHT_DECAY = 0.01
NUM_EPOCHS = 1
PER_DEVICE_TRAIN_BSZ = 1
GRAD_ACCUM_STEPS = 1
LOGGING_STEPS = 20
EVAL_STEPS = 200
SEED = 42

SAVE_STATE_DICT_NAME = "rosa_adapters.pt"

GRADIENT_CHECKPOINTING = True

TWO_STAGE = True

STAGE_A_TOKENS          = 3000000000
STAGE_A_LR_ROSA         = 1e-3
STAGE_A_WEIGHT_DECAY    = 0.01
STAGE_A_WARMUP_STEPS    = 20
SAVE_STEPS_STAGE_A      = 100
EVAL_STEPS_STAGE_A      = 1000000

STAGE_B_LR_ROSA         = 2e-4
STAGE_B_LR_BACKBONE     = 5e-6
STAGE_B_WEIGHT_DECAY    = 0.01
STAGE_B_WARMUP_STEPS    = 20
SAVE_STEPS_STAGE_B      = 100
EVAL_STEPS_STAGE_B      = 10000000

STAGE_B_ENABLE_GC       = True
STAGE_B_LCG_POS_SUBSAMPLE = 1.0


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

from concurrent.futures import ThreadPoolExecutor

_ROSA_THREAD_POOL = None
_ROSA_THREAD_POOL_PID = None

def _get_rosa_thread_pool() -> ThreadPoolExecutor | None:
    global _ROSA_THREAD_POOL, _ROSA_THREAD_POOL_PID
    n_workers = int(ROSA_THREAD_WORKERS)
    if n_workers <= 0:
        return None
    pid = os.getpid()
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




import numpy as _np
import torch


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
    def _k_rle_batch_btm_nb(z_btm: _np.ndarray):
        B, T, M = z_btm.shape
        run_start_bmt = _np.full((B, M, T), -1, dtype=_np.int32)
        run_sym_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
        c_len_bm      = _np.zeros((B, M),    dtype=_np.int32)
        for b in _nb.prange(B):
            for m in range(M):
                last = -2147483647
                clen = 0
                for t in range(T):
                    x = int(z_btm[b, t, m])
                    if t == 0 or x != last:
                        run_start_bmt[b, m, clen] = t
                        run_sym_bmt[b, m, clen]   = x
                        clen += 1
                        last = x
                c_len_bm[b, m] = clen
        return run_start_bmt, run_sym_bmt, c_len_bm

def _k_rle_batch_btm_py(z_btm_np: _np.ndarray):
    B, T, M = z_btm_np.shape
    run_start_bmt = _np.full((B, M, T), -1, dtype=_np.int32)
    run_sym_bmt   = _np.full((B, M, T), -1, dtype=_np.int32)
    c_len_bm      = _np.zeros((B, M),    dtype=_np.int32)
    for b in range(B):
        for m in range(M):
            last = None
            clen = 0
            for t in range(T):
                x = int(z_btm_np[b, t, m])
                if t == 0 or x != last:
                    run_start_bmt[b, m, clen] = t
                    run_sym_bmt[b, m, clen]   = x
                    clen += 1
                    last = x
            c_len_bm[b, m] = clen
    return run_start_bmt, run_sym_bmt, c_len_bm

def _k_rle_batch_btm(z_btm_np: _np.ndarray):
    if _NUMBA_OK:
        return _k_rle_batch_btm_nb(z_btm_np.astype(_np.int32, copy=False))
    return _k_rle_batch_btm_py(z_btm_np.astype(_np.int32, copy=False))


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


if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, parallel=True))
    def _k_rle_batch_btr_nb(z_btr: _np.ndarray):
        B, T, R = z_btr.shape
        run_start_brt = _np.full((B, R, T), -1, dtype=_np.int32)
        run_sym_brt   = _np.full((B, R, T), -1, dtype=_np.int32)
        c_len_br      = _np.zeros((B, R),    dtype=_np.int32)

        for br in _nb.prange(B * R):
            b = br // R
            r = br - b * R
            last = -2147483647
            clen = 0
            for t in range(T):
                x = int(z_btr[b, t, r])
                if t == 0 or x != last:
                    run_start_brt[b, r, clen] = t
                    run_sym_brt[b, r, clen]   = x
                    clen += 1
                    last = x
            c_len_br[b, r] = clen
        return run_start_brt, run_sym_brt, c_len_br


if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, parallel=True))
    def _compute_k_runcap_brt_nb(k_run_start_brt: _np.ndarray,
                                 k_c_len_br: _np.ndarray,
                                 T: int) -> _np.ndarray:
        B, R, _ = k_run_start_brt.shape
        rcap = _np.full((B, R, T), -1, dtype=_np.int32)

        for br in _nb.prange(B * R):
            b = br // R
            r = br - b * R
            clen = int(k_c_len_br[b, r])
            if clen <= 0:
                continue
            curr = 0
            rs = k_run_start_brt[b, r]
            for t in range(T):
                while (curr + 1) < clen:
                    nxt = rs[curr + 1]
                    if nxt != -1 and nxt <= t:
                        curr += 1
                    else:
                        break
                rcap[b, r, t] = curr
        return rcap


if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_new_state_nb(next_arr, link, length, e, size, L):
        s = size[0]; size[0] = s + 1
        length[s] = L; link[s] = -1; e[s] = -1
        K = next_arr.shape[1]
        for j in range(K):
            next_arr[s, j] = -1
        return s

    @(_nb.njit(cache=True, fastmath=False))
    def _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, pos):
        last = last_arr[0]
        cur = _sam_new_state_nb(next_arr, link, length, e, size, length[last] + 1)
        p = last; K = next_arr.shape[1]
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

    @(_nb.njit(cache=True, fastmath=False, inline='always'))
    def _sam_match_next_from_nb(next_arr, link, s, x):
        p = s
        while p != -1 and next_arr[p, x] == -1:
            p = link[p]
        return -1 if p == -1 else next_arr[p, x]


def _q_prefix_scan_bits_py(
    q_btr: _np.ndarray,
    k_run_sym_brt: _np.ndarray,
    k_run_start_brt: _np.ndarray,
    k_c_len_br: _np.ndarray,
    k_runcap_brt: _np.ndarray,
    Mbits: int,
    K: int
):
    B, T, R = q_btr.shape
    dest_time_brt  = _np.full((B, R, T), -1, dtype=_np.int32)
    q_run_id_brt   = _np.full((B, R, T), -1, dtype=_np.int32)
    rq_len_br      = _np.zeros((B, R), dtype=_np.int32)
    r_cf_run_brrjb = _np.full((B, R, T, Mbits, 2), -1, dtype=_np.int32)

    for b in range(B):
        for r in range(R):
            c     = k_run_sym_brt[b, r]
            rsK   = k_run_start_brt[b, r]
            clen  = int(k_c_len_br[b, r])
            rcap  = k_runcap_brt[b, r]

            S_cap = 2*T + 5
            next_arr = _np.full((S_cap, K), -1, dtype=_np.int32)
            link     = _np.full((S_cap,),   -1, dtype=_np.int32)
            length   = _np.zeros((S_cap,),     dtype=_np.int32)
            e        = _np.full((S_cap,),   -1, dtype=_np.int32)
            size     = _np.zeros((1,),        dtype=_np.int32); size[0] = 1
            last_arr = _np.zeros((1,),        dtype=_np.int32); last_arr[0] = 0

            rq = 0
            run_beg = _np.empty((T,), dtype=_np.int32)
            last = -2147483647
            for t in range(T):
                x = int(q_btr[b, t, r])
                if t == 0 or x != last:
                    run_beg[rq] = t
                    rq += 1
                    last = x
                q_run_id_brt[b, r, t] = rq - 1
            rq_len_br[b, r] = rq

            r_ext = -1
            s = 0
            cur_run = 0
            advanced_in_run = False

            def _extend_to(rt):
                nonlocal r_ext
                while r_ext < rt and (r_ext + 1) < clen:
                    r_ext += 1
                    x = int(c[r_ext])
                    _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, r_ext)

            for t in range(T):
                if cur_run < rq and t == run_beg[cur_run]:
                    cap_t = int(rcap[t])
                    if cap_t >= 0:
                        _extend_to(cap_t)
                    base_a = int(q_btr[b, t, r])
                    for j in range(Mbits):
                        maskj = 1 << j
                        for bit in (0, 1):
                            a_mod = (base_a & ~maskj) | (maskj if bit == 1 else 0)
                            ns = _sam_match_next_from_nb(next_arr, link, s, a_mod)
                            if ns != -1:
                                rpos = int(e[ns]); nxt = rpos + 1
                                if 0 <= rpos and nxt < (r_ext + 1):
                                    r_cf_run_brrjb[b, r, cur_run, j, bit] = nxt
                                else:
                                    r_cf_run_brrjb[b, r, cur_run, j, bit] = -1
                            else:
                                r_cf_run_brrjb[b, r, cur_run, j, bit] = -1
                    advanced_in_run = False

                cap_t = int(rcap[t])
                if cap_t >= 0:
                    _extend_to(cap_t)
                a_now = int(q_btr[b, t, r])
                ns = _sam_match_next_from_nb(next_arr, link, s, a_now)
                if ns != -1:
                    rpos = int(e[ns]); nxt = rpos + 1
                    if 0 <= rpos and nxt < (r_ext + 1):
                        dest_time_brt[b, r, t] = rsK[nxt]
                    else:
                        dest_time_brt[b, r, t] = -1
                    if not advanced_in_run:
                        s = ns
                        advanced_in_run = True
                else:
                    dest_time_brt[b, r, t] = -1

                if cur_run + 1 < rq and (t + 1) == run_beg[cur_run + 1]:
                    cur_run += 1

    return dest_time_brt, q_run_id_brt, rq_len_br, r_cf_run_brrjb

if _NUMBA_OK:
    @(_nb.njit(cache=True, fastmath=False, parallel=True))
    def _q_prefix_scan_bits_nb(
        q_btr: _np.ndarray,
        k_run_sym_brt: _np.ndarray,
        k_run_start_brt: _np.ndarray,
        k_c_len_br: _np.ndarray,
        k_runcap_brt: _np.ndarray,
        Mbits: int,
        K: int
    ):
        B, T, R = q_btr.shape
        dest_time_brt  = _np.full((B, R, T), -1, dtype=_np.int32)
        q_run_id_brt   = _np.full((B, R, T), -1, dtype=_np.int32)
        rq_len_br      = _np.zeros((B, R), dtype=_np.int32)
        r_cf_run_brrjb = _np.full((B, R, T, Mbits, 2), -1, dtype=_np.int32)

        for br in _nb.prange(B * R):
            b = br // R
            r = br - b * R

            c     = k_run_sym_brt[b, r]
            rsK   = k_run_start_brt[b, r]
            clen  = int(k_c_len_br[b, r])
            rcap  = k_runcap_brt[b, r]

            S_cap = 2 * T + 5
            next_arr = _np.full((S_cap, K), -1, dtype=_np.int32)
            link     = _np.full((S_cap,),   -1, dtype=_np.int32)
            length   = _np.zeros((S_cap,),     dtype=_np.int32)
            e        = _np.full((S_cap,),   -1, dtype=_np.int32)
            size     = _np.zeros((1,),        dtype=_np.int32); size[0] = 1
            last_arr = _np.zeros((1,),        dtype=_np.int32); last_arr[0] = 0

            rq = 0
            run_beg = _np.empty((T,), dtype=_np.int32)
            last = -2147483647
            for t in range(T):
                x = int(q_btr[b, t, r])
                if t == 0 or x != last:
                    run_beg[rq] = t
                    rq += 1
                    last = x
                q_run_id_brt[b, r, t] = rq - 1
            rq_len_br[b, r] = rq

            r_ext = -1
            s = 0
            cur_run = 0
            advanced_in_run = False

            for t in range(T):
                if cur_run < rq and t == run_beg[cur_run]:
                    cap_t = int(rcap[t])
                    if cap_t >= 0:
                        while r_ext < cap_t and (r_ext + 1) < clen:
                            r_ext += 1
                            x = int(c[r_ext])
                            _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, r_ext)

                    base_a = int(q_btr[b, t, r])
                    for j in range(Mbits):
                        maskj = 1 << j
                        for bit in (0, 1):
                            a_mod = (base_a & ~maskj) | (maskj if bit == 1 else 0)
                            ns = _sam_match_next_from_nb(next_arr, link, s, a_mod)
                            if ns != -1:
                                rpos = int(e[ns]); nxt = rpos + 1
                                if 0 <= rpos and nxt < (r_ext + 1):
                                    r_cf_run_brrjb[b, r, cur_run, j, bit] = nxt
                                else:
                                    r_cf_run_brrjb[b, r, cur_run, j, bit] = -1
                            else:
                                r_cf_run_brrjb[b, r, cur_run, j, bit] = -1
                    advanced_in_run = False

                cap_t = int(rcap[t])
                if cap_t >= 0:
                    while r_ext < cap_t and (r_ext + 1) < clen:
                        r_ext += 1
                        x = int(c[r_ext])
                        _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, r_ext)
                a_now = int(q_btr[b, t, r])
                ns = _sam_match_next_from_nb(next_arr, link, s, a_now)
                if ns != -1:
                    rpos = int(e[ns]); nxt = rpos + 1
                    if 0 <= rpos and nxt < (r_ext + 1):
                        dest_time_brt[b, r, t] = rsK[nxt]
                    else:
                        dest_time_brt[b, r, t] = -1
                    if not advanced_in_run:
                        s = ns
                        advanced_in_run = True
                else:
                    dest_time_brt[b, r, t] = -1

                if cur_run + 1 < rq and (t + 1) == run_beg[cur_run + 1]:
                    cur_run += 1

        return dest_time_brt, q_run_id_brt, rq_len_br, r_cf_run_brrjb

def _rosa_binary_pipeline_cpu(
    q_cat_btr_torch: torch.Tensor,
    k_cat_btr_torch: torch.Tensor,
    Mbits: int
):
    assert q_cat_btr_torch.is_cuda and k_cat_btr_torch.is_cuda
    B, T, R = q_cat_btr_torch.shape
    use_i16 = (T <= 32767)
    K = 1 << int(Mbits)

    if _NUMBA_OK and ROSA_ENABLE_NUMBA_PARALLEL:
        try:
            _nb.set_num_threads(int(ROSA_NUMBA_THREADS))
        except Exception:
            pass

    q_host = _PINNED_POOL.get("q_cat_host", (B, T, R), dtype=torch.int32)
    k_host = _PINNED_POOL.get("k_cat_host", (B, T, R), dtype=torch.int32)
    q_host.copy_(q_cat_btr_torch, non_blocking=True)
    k_host.copy_(k_cat_btr_torch, non_blocking=True)
    ev_q = torch.cuda.Event(); ev_q.record(torch.cuda.current_stream())
    ev_k = torch.cuda.Event(); ev_k.record(torch.cuda.current_stream())

    _dtype_t = torch.int16 if use_i16 else torch.int32
    k_run_start_cpu = _PINNED_POOL.get("k_run_start_brt", (B, R, T), dtype=_dtype_t)
    dest_time_cpu   = _PINNED_POOL.get("dest_time_brt",   (B, R, T), dtype=_dtype_t)
    q_run_id_cpu    = _PINNED_POOL.get("q_run_id_brt",    (B, R, T), dtype=_dtype_t)
    r_cf_bit_cpu    = _PINNED_POOL.get("r_cf_run_brrjb",  (B, R, T, Mbits, 2), dtype=_dtype_t)

    def _wait_and_run():
        _wait_event(ev_q); _wait_event(ev_k)
        q_np = _np.asarray(q_host, order="C")
        k_np = _np.asarray(k_host, order="C")

        k_runstart_np, k_runsym_np, k_clen_np = _k_rle_batch_btr_nb(k_np.astype(_np.int32, copy=False))
        k_runcap_np = _compute_k_runcap_brt_nb(
            k_runstart_np.astype(_np.int32, copy=False),
            k_clen_np.astype(_np.int32, copy=False),
            int(T)
        )
        if _NUMBA_OK and ('_q_prefix_scan_bits_nb' in globals()):
            dest_np, q_run_id_np, _rq_len, r_cf_np = _q_prefix_scan_bits_nb(
                q_np.astype(_np.int32, copy=False),
                k_runsym_np.astype(_np.int32, copy=False),
                k_runstart_np.astype(_np.int32, copy=False),
                k_clen_np.astype(_np.int32, copy=False),
                k_runcap_np.astype(_np.int32, copy=False),
                int(Mbits),
                int(K)
            )
        else:
            dest_np, q_run_id_np, _rq_len, r_cf_np = _q_prefix_scan_bits_py(
                q_np.astype(_np.int32, copy=False),
                k_runsym_np.astype(_np.int32, copy=False),
                k_runstart_np.astype(_np.int32, copy=False),
                k_clen_np.astype(_np.int32, copy=False),
                k_runcap_np.astype(_np.int32, copy=False),
                int(Mbits),
                int(K)
            )

        _np.copyto(_np.asarray(k_run_start_cpu), k_runstart_np, casting="unsafe")
        _np.copyto(_np.asarray(dest_time_cpu),   dest_np,       casting="unsafe")
        _np.copyto(_np.asarray(q_run_id_cpu),    q_run_id_np,   casting="unsafe")
        _np.copyto(_np.asarray(r_cf_bit_cpu),    r_cf_np,       casting="unsafe")

        return k_run_start_cpu, dest_time_cpu, q_run_id_cpu, r_cf_bit_cpu

    pool = _get_rosa_thread_pool()
    if pool is None:
        return _ImmediateFuture(_wait_and_run())
    else:
        return pool.submit(_wait_and_run)



class MultiRouteBinaryROSAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                y_in: torch.Tensor,
                q_vec: torch.Tensor,
                k_vec: torch.Tensor,
                v_vec: torch.Tensor,
                dest_time_brt: torch.Tensor,
                q_run_id_brt: torch.Tensor,
                k_run_start_brt: torch.Tensor,
                r_cf_run_brrjb: torch.Tensor,
                delta_vec: torch.Tensor,
                bits_per_route: int,
                pos_mask_bt: torch.Tensor | None
                ):
        ctx.save_for_backward(q_vec, k_vec, v_vec,
                              dest_time_brt.to(torch.int64),
                              q_run_id_brt.to(torch.int64),
                              k_run_start_brt.to(torch.int64),
                              r_cf_run_brrjb.to(torch.int64),
                              delta_vec)
        ctx.bits_per_route = int(bits_per_route)
        ctx.pos_mask_bt = None if pos_mask_bt is None else pos_mask_bt.bool()
        return y_in

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        (q_vec, k_vec, v_vec,
         dest_time_brt, q_run_id_brt, k_run_start_brt,
         r_cf_run_brrjb, delta_vec) = ctx.saved_tensors

        B, T, C = q_vec.shape
        Mbits = ctx.bits_per_route
        assert C % Mbits == 0
        R = C // Mbits
        device = q_vec.device
        out_dtype = grad_y.dtype
        f32 = torch.float32

        if ctx.pos_mask_bt is not None:
            grad_y = grad_y * ctx.pos_mask_bt.to(device=device).unsqueeze(-1)

        delta = delta_vec.to(device=device, dtype=out_dtype)
        theta_btc = grad_y * delta.view(1, 1, C)
        theta_brtm_32 = theta_btc.view(B, T, R, Mbits).permute(0, 2, 1, 3).contiguous().to(f32)

        T_lim = v_vec.size(1)
        valid_dest_brt = (dest_time_brt >= 0)
        dest_clamp = dest_time_brt.clamp(0, T_lim - 1).long()

        S_v_32 = torch.zeros((B, R, T_lim, Mbits), device=device, dtype=f32)
        S_v_32.scatter_add_(
            dim=2,
            index=dest_clamp.unsqueeze(-1).expand(-1, -1, -1, Mbits),
            src=theta_brtm_32 * valid_dest_brt.unsqueeze(-1).to(f32)
        )

        pV = torch.sigmoid(BINARY_TEMP_V * v_vec)
        gV_scale_32 = pV.to(f32).view(B, T, R, Mbits).permute(0, 2, 1, 3).contiguous()
        gV_scale_32 = gV_scale_32 * (1.0 - gV_scale_32)
        grad_v_bits_32 = gV_scale_32 * S_v_32
        grad_v = grad_v_bits_32.permute(0, 2, 1, 3).contiguous().view(B, T, C).to(out_dtype)

        v_bits_brtm_32 = (v_vec.view(B, T, R, Mbits) > 0).permute(0, 2, 1, 3).to(f32)
        v_prob_brtm_32 = pV.view(B, T, R, Mbits).permute(0, 2, 1, 3).to(f32)
        v_field_q_32   = v_prob_brtm_32 if (ROSA_Q_CONTRAST_USE_V_PROB == 1) else v_bits_brtm_32
        v_field_k_32   = v_bits_brtm_32

        r_cf_brtjb = torch.gather(
            r_cf_run_brrjb,
            dim=2,
            index=q_run_id_brt.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, Mbits, 2)
        )

        r_idx0 = r_cf_brtjb[..., 0]
        r_idx1 = r_cf_brtjb[..., 1]
        mask0  = (r_idx0 >= 0)
        mask1  = (r_idx1 >= 0)

        Rmax   = k_run_start_brt.size(2)
        r_idx0c = r_idx0.clamp(0, Rmax - 1).long()
        r_idx1c = r_idx1.clamp(0, Rmax - 1).long()

        k_rs_expand = k_run_start_brt.unsqueeze(3).expand(-1, -1, -1, Mbits)
        idx0_time = torch.gather(k_rs_expand, dim=2, index=r_idx0c)
        idx1_time = torch.gather(k_rs_expand, dim=2, index=r_idx1c)

        J = Mbits
        idx0J_M = idx0_time.unsqueeze(-1).expand(-1, -1, -1, J, Mbits)
        idx1J_M = idx1_time.unsqueeze(-1).expand(-1, -1, -1, J, Mbits)

        theta_jm_32 = theta_brtm_32.unsqueeze(-2).expand(-1, -1, -1, J, -1)

        vq_allm_32 = v_field_q_32.unsqueeze(-2).expand(-1, -1, -1, J, -1)
        v0_q_jm = torch.gather(vq_allm_32, dim=2, index=idx0J_M.long()) * mask0.unsqueeze(-1).to(f32)
        v1_q_jm = torch.gather(vq_allm_32, dim=2, index=idx1J_M.long()) * mask1.unsqueeze(-1).to(f32)
        dot_diff_j_32 = (theta_jm_32 * (v1_q_jm - v0_q_jm)).sum(dim=-1)

        pQ = torch.sigmoid(BINARY_TEMP_Q * q_vec).view(B, T, R, Mbits).permute(0, 2, 1, 3).contiguous()
        pQ_32 = pQ.to(f32)
        grad_q_bits_32 = pQ_32 * (1.0 - pQ_32) * dot_diff_j_32
        grad_q = grad_q_bits_32.permute(0, 2, 1, 3).contiguous().view(B, T, C).to(out_dtype)

        vk_allm_32 = v_field_k_32.unsqueeze(-2).expand(-1, -1, -1, J, -1)
        v0_k_jm = torch.gather(vk_allm_32, dim=2, index=idx0J_M) * mask0.unsqueeze(-1).to(f32)
        v1_k_jm = torch.gather(vk_allm_32, dim=2, index=idx1J_M) * mask1.unsqueeze(-1).to(f32)
        dot0_j_32 = (theta_jm_32 * v0_k_jm).sum(dim=-1)
        dot1_j_32 = (theta_jm_32 * v1_k_jm).sum(dim=-1)

        U1_32 = torch.zeros((B, R, Rmax, J), device=device, dtype=f32)
        U0_32 = torch.zeros_like(U1_32)
        U1_32.scatter_add_(dim=2, index=r_idx1c, src=dot1_j_32 * mask1.to(f32))
        U0_32.scatter_add_(dim=2, index=r_idx0c, src=dot0_j_32 * mask0.to(f32))
        diffU_32 = U1_32 - U0_32

        pK = torch.sigmoid(BINARY_TEMP_K * k_vec).view(B, T, R, Mbits).permute(0, 2, 1, 3).contiguous()
        pK_32 = pK.to(f32)
        k_start_long = k_run_start_brt.clamp(0, T_lim - 1).long()
        pK_run_32 = torch.gather(pK_32, dim=2, index=k_start_long.unsqueeze(-1).expand(-1, -1, -1, J))
        gK_run_32 = (pK_run_32 * (1.0 - pK_run_32)) * diffU_32

        grad_k_bits_32 = torch.zeros_like(pK_32)
        grad_k_bits_32.scatter_add_(dim=2,
                                    index=k_start_long.unsqueeze(-1).expand(-1, -1, -1, J),
                                    src=gK_run_32)
        grad_k = grad_k_bits_32.permute(0, 2, 1, 3).contiguous().view(B, T, C).to(out_dtype)

        grad_y_out = grad_y

        return grad_y_out, grad_q, grad_k, grad_v, None, None, None, None, None, None, None


from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class FixedLenLMCollator:
    pad_token_id: int = 0
    seq_len: int = 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        cols = ("input_ids", "attention_mask", "labels")
        if not features:
            raise ValueError("FixedLenLMCollator: received empty features list.")

        first = features[0]
        for c in cols:
            if c not in first:
                raise KeyError(
                    f"Sample missing column '{c}'. Expected columns: {cols}."
                )

        batch: Dict[str, torch.Tensor] = {}
        for c in cols:
            xs = [ex[c] for ex in features]
            if isinstance(xs[0], torch.Tensor):
                batch[c] = torch.stack(xs, dim=0)
            else:
                batch[c] = torch.tensor(xs)
        return batch


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer

def patch_qwen3_with_multiroute_rosa(model: Qwen3ForCausalLM):
    import torch
    import torch.nn as nn
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    inject_mode = os.environ.get("ROSA_INJECT_MODE", globals().get("ROSA_INJECT_MODE", "pre_attn")).lower()
    assert inject_mode in ("pre_attn", "post_attn")

    base_param = model.model.embed_tokens.weight
    base_dtype = base_param.dtype
    base_device = base_param.device
    C: int = model.config.hidden_size
    assert C % BITS_PER_ROUTE == 0, f"hidden_size {C} must be divisible by BITS_PER_ROUTE={BITS_PER_ROUTE}"
    R: int = C // BITS_PER_ROUTE
    K_sym: int = 1 << BITS_PER_ROUTE

    def _pack_bits_btC_to_btr(bits_btC: torch.Tensor, Mbits: int) -> torch.Tensor:
        B, T, Cx = bits_btC.shape
        assert Cx % Mbits == 0
        Rloc = Cx // Mbits
        x = bits_btC.view(B, T, Rloc, Mbits).to(torch.int32)
        out = torch.zeros((B, T, Rloc), dtype=torch.int32, device=bits_btC.device)
        for j in range(Mbits):
            out |= ((x[..., j] & 1) << j)
        return out

    def _unpack_btr_to_bits_brtm(cat_btr: torch.Tensor, Mbits: int) -> torch.Tensor:
        B, T, Rloc = cat_btr.shape
        x = cat_btr.to(torch.int32)
        bits = []
        for j in range(Mbits):
            bits.append(((x >> j) & 1))
        out = torch.stack(bits, dim=-1).to(torch.int32)
        return out

    for li, layer in enumerate(model.model.layers):
        if li == 0:
            continue

        layer.rosa_q_proj = nn.Linear(C, C, bias=False).to(dtype=base_dtype, device=base_device)
        layer.rosa_k_proj = nn.Linear(C, C, bias=False).to(dtype=base_dtype, device=base_device)
        layer.rosa_v_proj = nn.Linear(C, C, bias=False).to(dtype=base_dtype, device=base_device)
        nn.init.xavier_uniform_(layer.rosa_q_proj.weight)
        nn.init.xavier_uniform_(layer.rosa_k_proj.weight)
        nn.init.xavier_uniform_(layer.rosa_v_proj.weight)

        layer.rosa_e0 = nn.Parameter(torch.zeros(C, dtype=base_dtype, device=base_device))
        layer.rosa_e1 = nn.Parameter(torch.zeros(C, dtype=base_dtype, device=base_device))
        layer.rosa_out = nn.Linear(C, C, bias=False).to(dtype=base_dtype, device=base_device)
        nn.init.xavier_uniform_(layer.rosa_out.weight)

        layer.rosa_alpha = nn.Parameter(torch.zeros(C, dtype=base_dtype, device=base_device))

        layer.rosa_bits_per_route = BITS_PER_ROUTE
        layer.rosa_num_routes = R
        layer.rosa_k_symbols = K_sym
        layer.rosa_inject_mode = inject_mode

        def _forward_qkv_brosa(
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
            B, T, Cx = hidden_states.shape
            assert Cx == C

            u = self.input_layernorm(hidden_states)
            q_vec = self.rosa_q_proj(u)
            k_vec = self.rosa_k_proj(u)
            v_vec = self.rosa_v_proj(u)

            q_bits = (q_vec > 0).to(torch.int32)
            k_bits = (k_vec > 0).to(torch.int32)
            v_bits = (v_vec > 0).to(torch.int32)

            q_cat = _pack_bits_btC_to_btr(q_bits, BITS_PER_ROUTE)
            k_cat = _pack_bits_btC_to_btr(k_bits, BITS_PER_ROUTE)
            v_cat = _pack_bits_btC_to_btr(v_bits, BITS_PER_ROUTE)

            nvtx.range_push("bROSA_CPU_launch")
            fut_cpu = _rosa_binary_pipeline_cpu(q_cat, k_cat, BITS_PER_ROUTE)
            nvtx.range_pop()

            if self.rosa_inject_mode == "post_attn":
                attn_out, _ = self.self_attn(
                    hidden_states=u,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                nvtx.range_push("bROSA_CPU_wait")
                k_run_start_cpu, dest_time_cpu, q_run_id_cpu, r_cf_bit_cpu = fut_cpu.result()
                nvtx.range_pop()
            else:
                nvtx.range_push("bROSA_CPU_wait")
                k_run_start_cpu, dest_time_cpu, q_run_id_cpu, r_cf_bit_cpu = fut_cpu.result()
                nvtx.range_pop()
                attn_out = None

            k_run_start_brt = k_run_start_cpu.to(
                device=hidden_states.device, dtype=torch.int32, non_blocking=True
            )
            dest_time_brt = dest_time_cpu.to(
                device=hidden_states.device, dtype=torch.int32, non_blocking=True
            )
            q_run_id_brt = q_run_id_cpu.to(
                device=hidden_states.device, dtype=torch.int32, non_blocking=True
            )
            r_cf_run_brrjb = r_cf_bit_cpu.to(
                device=hidden_states.device, dtype=torch.int32, non_blocking=True
            )

            dest_clamp = dest_time_brt.clamp(0, T - 1).to(torch.long)
            valid_dest_brt = (dest_time_brt >= 0)

            v_cat_brt = v_cat.permute(0, 2, 1).contiguous()
            v_cat_at_dest = torch.take_along_dim(v_cat_brt, indices=dest_clamp, dim=2)
            v_cat_at_dest = v_cat_at_dest * valid_dest_brt.to(v_cat_at_dest.dtype)

            v_bits_at_dest_brtm = _unpack_btr_to_bits_brtm(
                v_cat_at_dest.permute(0, 2, 1), BITS_PER_ROUTE
            ).permute(0, 2, 1, 3).contiguous()

            b_bits_btc = v_bits_at_dest_brtm.permute(0, 2, 1, 3).contiguous().view(B, T, C)

            delta = (self.rosa_e1 - self.rosa_e0)
            y_valid = self.rosa_e0.view(1, 1, C) + delta.view(1, 1, C) * b_bits_btc.to(delta.dtype)

            valid_mask_btc = valid_dest_brt.permute(0, 2, 1).unsqueeze(-1).expand(B, T, R, BITS_PER_ROUTE).reshape(B, T, C)
            y_final = y_valid * valid_mask_btc.to(y_valid.dtype)

            pos_mask_cpu = None
            if LCG_POS_SUBSAMPLE < 1.0:
                mask = (torch.rand((B, T), device=hidden_states.device) < LCG_POS_SUBSAMPLE)
                pos_mask_cpu = mask.detach()

            y_hook = MultiRouteBinaryROSAFunction.apply(
                y_final,
                q_vec,
                k_vec,
                v_vec,
                dest_time_brt,
                q_run_id_brt,
                k_run_start_brt,
                r_cf_run_brrjb,
                delta.detach(),
                BITS_PER_ROUTE,
                None if pos_mask_cpu is None else pos_mask_cpu,
            )
            inj = self.rosa_out(y_hook)

            if self.rosa_inject_mode == "post_attn":
                post_mix = attn_out + inj
                hidden_states = residual + post_mix
            else:
                alpha = torch.sigmoid(self.rosa_alpha).view(1, 1, C).to(dtype=hidden_states.dtype, device=hidden_states.device)
                mix = (1.0 - alpha) * hidden_states + alpha * inj
                u2 = self.input_layernorm(mix)
                attn_out2, _ = self.self_attn(
                    hidden_states=u2,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                hidden_states = residual + attn_out2

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states
            return hidden_states

        layer.forward = _forward_qkv_brosa.__get__(layer, Qwen3DecoderLayer)

    meta = {
        "apply_layers_from": 1,
        "bits_per_route": BITS_PER_ROUTE,
        "num_routes": R,
        "k_symbols": K_sym,
        "inject_mode": inject_mode,
        "C": C
    }
    with open(os.path.join(OUTPUT_DIR, "rosa_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def build_model_and_tokenizer() -> Tuple[Qwen3ForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(MODEL_LOCAL_DIR)

    if hasattr(config, "use_sliding_window"):
        config.use_sliding_window = True

    config.sliding_window = int(ATTN_WINDOW) if ATTN_WINDOW is not None else config.sliding_window
    config.tie_word_embeddings = TIE

    config.max_window_layers = int(FIRST_GLOBAL_LAYERS) if FIRST_GLOBAL_LAYERS is not None else 0

    n_layers = int(getattr(config, "num_hidden_layers"))
    config.layer_types = [
        "full_attention" if i < config.max_window_layers else "sliding_attention"
        for i in range(n_layers)
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

    model.config.use_cache = False
    patch_qwen3_with_multiroute_rosa(model)

    if MODEL_DIR is not None:
        from safetensors import safe_open
        state_dict = {}
        with safe_open(MODEL_DIR, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(unexpected) > 0:
            print(f"[warn] Unexpected keys when loading adapter: {len(unexpected)} (showing first 10) -> {unexpected[:10]}")
        if len(missing) > 0:
            print(f"[warn] Missing keys when loading adapter: {len(missing)} (showing first 10) -> {missing[:10]}")

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model, tokenizer


def save_rosa_only(model: Qwen3ForCausalLM, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    state = {}
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer, "rosa_q_proj"):
            state[f"model.layers.{i}.rosa_q_proj.weight"] = layer.rosa_q_proj.weight.detach().cpu()
        if hasattr(layer, "rosa_k_proj"):
            state[f"model.layers.{i}.rosa_k_proj.weight"] = layer.rosa_k_proj.weight.detach().cpu()
        if hasattr(layer, "rosa_v_proj"):
            state[f"model.layers.{i}.rosa_v_proj.weight"] = layer.rosa_v_proj.weight.detach().cpu()
        if hasattr(layer, "rosa_out"):
            state[f"model.layers.{i}.rosa_out.weight"] = layer.rosa_out.weight.detach().cpu()
        if hasattr(layer, "rosa_e0"):
            state[f"model.layers.{i}.rosa_e0"] = layer.rosa_e0.detach().cpu()
        if hasattr(layer, "rosa_e1"):
            state[f"model.layers.{i}.rosa_e1"] = layer.rosa_e1.detach().cpu()
        if hasattr(layer, "rosa_alpha"):
            state[f"model.layers.{i}.rosa_alpha"] = layer.rosa_alpha.detach().cpu()

    path = os.path.join(out_dir, SAVE_STATE_DICT_NAME)
    torch.save(state, path)
    print(f"[save] saved bROSA-only params to: {path}")


def save_meta(model: Qwen3ForCausalLM, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    num_layers = model.config.num_hidden_layers
    k_info = {
        "bits_per_route": int(BITS_PER_ROUTE),
        "num_routes": int(model.config.hidden_size // BITS_PER_ROUTE),
        "k_symbols": int(1 << BITS_PER_ROUTE)
    }

    meta = {
        "model_local_dir": MODEL_LOCAL_DIR,
        "dataset_dir": DATASET_DIR,
        "seq_len": SEQ_LEN,
        "attn_window": ATTN_WINDOW,
        "first_global_layers": FIRST_GLOBAL_LAYERS,
        "rosa": {
            "type": "binary",
            "bits_per_route": int(BITS_PER_ROUTE),
            "temperature": {
                "q": float(BINARY_TEMP_Q),
                "k": float(BINARY_TEMP_K),
                "v": float(BINARY_TEMP_V),
            },
            "pos_subsample": float(LCG_POS_SUBSAMPLE)
        },
        "time": time.asctime(),
        "k_info": k_info
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[done] meta saved at {os.path.join(out_dir, 'run_meta.json')}")


def build_optimizer_params(model):
    inject_mode = os.environ.get("ROSA_INJECT_MODE", globals().get("ROSA_INJECT_MODE", "pre_attn")).lower()

    heads_decay, emb_nodecay, gate_nodecay = [], [], []
    backbone_decay, backbone_nodecay = [], []

    for n, p in model.named_parameters():
        if any(k in n for k in [
            "rosa_q_proj.weight", "rosa_k_proj.weight", "rosa_v_proj.weight", "rosa_out.weight"
        ]):
            heads_decay.append(p)
        elif "rosa_e0" in n or "rosa_e1" in n:
            emb_nodecay.append(p)
        elif "rosa_alpha" in n:
            if inject_mode == "pre_attn":
                gate_nodecay.append(p)
            else:
                p.requires_grad_(False)
        else:
            if any(k in n.lower() for k in ["bias", "norm", "layernorm", "ln"]):
                backbone_nodecay.append(p)
            else:
                backbone_decay.append(p)

    for p in backbone_decay + backbone_nodecay:
        if p.requires_grad:
            p.requires_grad_(False)
        p.grad = None

    groups = []
    if heads_decay:
        groups.append({
            "name": "rosa_heads_decay",
            "params": heads_decay,
            "lr": STAGE_A_LR_ROSA,
            "weight_decay": STAGE_A_WEIGHT_DECAY
        })
    if emb_nodecay:
        groups.append({
            "name": "emb_nodecay",
            "params": emb_nodecay,
            "lr": STAGE_A_LR_ROSA,
            "weight_decay": 0.0
        })
    if gate_nodecay:
        groups.append({
            "name": "gate_nodecay",
            "params": gate_nodecay,
            "lr": STAGE_A_LR_ROSA,
            "weight_decay": 0.0
        })
    if backbone_decay:
        groups.append({
            "name": "backbone_decay",
            "params": backbone_decay,
            "lr": 0.0,
            "weight_decay": STAGE_B_WEIGHT_DECAY
        })
    if backbone_nodecay:
        groups.append({
            "name": "backbone_nodecay",
            "params": backbone_nodecay,
            "lr": 0.0,
            "weight_decay": 0.0
        })
    return groups


def _get_world_size() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
    except Exception:
        pass
    return _env_int("WORLD_SIZE", 1)

from transformers.trainer_callback import TrainerCallback
import torch

class TwoStageSwitchCallback(TrainerCallback):
    def __init__(self):
        self.switched = False
        self.switch_step = None

    def _tokens_per_update(self, args) -> int:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        return int(SEQ_LEN) * int(args.per_device_train_batch_size) * int(args.gradient_accumulation_steps) * world_size

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        try:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                print("[TwoStage] ensured input requires_grad at train begin.")
        except Exception as e:
            print(f"[TwoStage] WARN: enable_input_require_grads at begin failed: {e}")
        return control

    @staticmethod
    def _constant_warmup_lambda_factory(warmup_steps: int):
        def lr_lambda(current_step: int):
            if warmup_steps <= 0:
                return 1.0
            return float(current_step) / float(max(1, warmup_steps)) if current_step < warmup_steps else 1.0
        return lr_lambda

    def on_step_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        if self.switched or (not TWO_STAGE):
            return control

        trained_tokens = state.global_step * self._tokens_per_update(args)
        if trained_tokens < STAGE_A_TOKENS:
            return control

        print(f"[TwoStage] Switch to Stage-B at step={state.global_step}, tokens≈{trained_tokens:,}")

        if model is not None:
            for n, p in model.named_parameters():
                if any(k in n for k in ["rosa_q_proj", "rosa_k_proj", "rosa_v_proj", "rosa_out", "rosa_e0", "rosa_e1", "rosa_alpha"]):
                    continue
                if not p.requires_grad:
                    p.requires_grad_(True)
                    p.grad = None
            print("[TwoStage] backbone params unfrozen for Stage-B.")

        for pg in optimizer.param_groups:
            name = pg.get("name", "")
            if name in ("backbone_decay", "backbone_nodecay"):
                pg["lr"] = STAGE_B_LR_BACKBONE
                pg["weight_decay"] = STAGE_B_WEIGHT_DECAY if name == "backbone_decay" else 0.0
            elif name in ("rosa_heads_decay", "emb_nodecay", "gate_nodecay"):
                pg["lr"] = STAGE_B_LR_ROSA

        try:
            from torch.optim.lr_scheduler import LambdaLR
            if isinstance(lr_scheduler, LambdaLR):
                new_lambda = self._constant_warmup_lambda_factory(STAGE_B_WARMUP_STEPS)
                lr_scheduler.lr_lambdas = [new_lambda for _ in optimizer.param_groups]
                lr_scheduler.last_epoch = -1
                print(f"[TwoStage] LR scheduler -> constant_with_warmup, warmup={STAGE_B_WARMUP_STEPS}.")
            else:
                print("[TwoStage] WARN: lr_scheduler is not LambdaLR; skip reset.")
        except Exception as e:
            print(f"[TwoStage] WARN: scheduler reset skipped: {e}")

        try:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
                print("[TwoStage] enabled input require_grads for Stage-B.")
        except Exception as e:
            print(f"[TwoStage] WARN: enable_input_require_grads failed: {e}")

        if STAGE_B_ENABLE_GC and hasattr(model, "gradient_checkpointing_enable"):
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                args.gradient_checkpointing = True
            except Exception:
                pass

        args.eval_steps = EVAL_STEPS_STAGE_B
        args.save_steps = SAVE_STEPS_STAGE_B
        global LCG_POS_SUBSAMPLE
        LCG_POS_SUBSAMPLE = STAGE_B_LCG_POS_SUBSAMPLE
        print(f"[TwoStage] LCG_POS_SUBSAMPLE -> {LCG_POS_SUBSAMPLE}")

        self.switched = True
        self.switch_step = state.global_step
        return control


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

    model, tokenizer = build_model_and_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data_collator = FixedLenLMCollator(pad_token_id=pad_id, seq_len=SEQ_LEN)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BSZ,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,

        learning_rate=STAGE_A_LR_ROSA,

        lr_scheduler_type="constant_with_warmup",
        warmup_steps=STAGE_A_WARMUP_STEPS,

        logging_steps=LOGGING_STEPS,
        eval_steps=EVAL_STEPS_STAGE_A,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS_STAGE_A,
        report_to="none",
        fp16=(not BF16) and torch.cuda.is_available(),
        bf16=BF16,
        dataloader_num_workers=2,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=16,
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
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[RosaZeroRowCallback(), TwoStageSwitchCallback()],
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")

    from transformers.trainer_utils import get_last_checkpoint

    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    trainer.train(resume_from_checkpoint=last_ckpt)

    if is_main_process():
        save_meta(model, OUTPUT_DIR)


if __name__ == "__main__":
    main()
