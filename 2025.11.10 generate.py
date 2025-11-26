#!/usr/bin/env python3
# -*- coding: utf-8 -*-

TIE = False

MODEL_LOCAL_DIR: str = "/path/to/base/model/"
ROSA_ADAPTER_PATH: str | None = "/path/to/adapter/model.safetensors"
ROSA_META_JSON_PATH = None
RUN_META_JSON_PATH = None

MAX_NEW_TOKENS: int = 128
DO_SAMPLE: bool = False
TEMPERATURE: float = 0.8
TOP_P: float = 0.95
TOP_K: int = 50
NUM_BEAMS: int = 1
REPETITION_PENALTY: float = 1.05

BITS_PER_ROUTE: int = 4
ROSA_INJECT_MODE: str = "post_attn"
APPLY_FROM_LAYER: int = 1

ATTN_WINDOW: int = 1024
FIRST_GLOBAL_LAYERS: int = 1

USE_FLASH_ATTN: bool = True
USE_BF16: bool = True
DEVICE_FALLBACK_CPU: bool = False

ROSA_USE_NUMBA: bool = True
ROSA_NUMBA_PARALLEL: bool = True
ROSA_NUMBA_THREADS: int = 64
ROSA_THREAD_WORKERS: int = 0

SEED: int = 42

import os
import json
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer

try:
    import torch.cuda.nvtx as nvtx
except Exception:
    class _DummyNVTX:
        def range_push(self, *a, **k): pass
        def range_pop(self): pass
    nvtx = _DummyNVTX()

_HAS_SAFETENSORS = False
try:
    from safetensors import safe_open
    _HAS_SAFETENSORS = True
except Exception:
    _HAS_SAFETENSORS = False

try:
    import numba as nb
    _NUMBA_OK = bool(ROSA_USE_NUMBA)
except Exception:
    _NUMBA_OK = False

import os
import numpy as np

try:
    import numba as nb
    _NUMBA_OK = True
except Exception:
    _NUMBA_OK = False

from concurrent.futures import ThreadPoolExecutor
_ROSA_THREAD_POOL = None
_ROSA_THREAD_POOL_PID = None
def _get_rosa_thread_pool():
    global _ROSA_THREAD_POOL, _ROSA_THREAD_POOL_PID
    n_workers = int(ROSA_THREAD_WORKERS)
    if n_workers <= 0:
        return None
    pid = os.getpid()
    if (_ROSA_THREAD_POOL is None) or (_ROSA_THREAD_POOL_PID != pid):
        if _ROSA_THREAD_POOL is not None:
            try: _ROSA_THREAD_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception: pass
        _ROSA_THREAD_POOL = ThreadPoolExecutor(max_workers=n_workers)
        _ROSA_THREAD_POOL_PID = pid
    return _ROSA_THREAD_POOL

if _NUMBA_OK:
    try:
        nb.set_num_threads(int(ROSA_NUMBA_THREADS))
    except Exception:
        pass

if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False, inline='always')
    def _sam_new_state_nb(next_arr, link, length, e, size, L):
        s = size[0]; size[0] = s + 1
        length[s] = L; link[s] = -1; e[s] = -1
        K = next_arr.shape[1]
        for j in range(K): next_arr[s, j] = -1
        return s

    @nb.njit(cache=True, fastmath=False)
    def _sam_extend_nb(next_arr, link, length, e, last_arr, size, x, pos):
        last = last_arr[0]
        cur = _sam_new_state_nb(next_arr, link, length, e, size, length[last] + 1)
        p = last; K = next_arr.shape[1]
        while p != -1 and next_arr[p, x] == -1:
            next_arr[p, x] = cur; p = link[p]
        if p == -1:
            link[cur] = 0
        else:
            q = next_arr[p, x]
            if length[p] + 1 == length[q]:
                link[cur] = q
            else:
                clone = _sam_new_state_nb(next_arr, link, length, e, size, length[p] + 1)
                for j in range(K): next_arr[clone, j] = next_arr[q, j]
                link[clone] = link[q]; e[clone] = e[q]
                while p != -1 and next_arr[p, x] == q:
                    next_arr[p, x] = clone; p = link[p]
                link[q] = clone; link[cur] = clone
        v = cur
        while v != -1 and e[v] != pos:
            e[v] = pos; v = link[v]
        last_arr[0] = cur

    @nb.njit(cache=True, fastmath=False, inline='always')
    def _sam_match_next_from_nb(next_arr, link, s, x):
        p = s
        while p != -1 and next_arr[p, x] == -1:
            p = link[p]
        return -1 if p == -1 else next_arr[p, x]

if _NUMBA_OK:
    @nb.njit(cache=True, fastmath=False, inline='always')
    def _store_bits_at_row(dst_row, x, J):
        for j in range(J):
            dst_row[j] = (x >> j) & 1

    @nb.njit(cache=True, fastmath=False, parallel=True)
    def _prefill_kernel_nb(q_btr, k_btr, v_btr,
                           sam_next, sam_link, sam_len, sam_e, sam_size, sam_last,
                           run_start, run_sym, v_bits_runs, E_idx,
                           s_q, q_last, nu, t_idx,
                           J, K):
        B, T, R = q_btr.shape
        N = B * R
        out_bits = np.zeros((B, T, R, J), dtype=np.int32)
        out_valid = np.zeros((B, R, T), dtype=np.int32)

        for br in nb.prange(N):
            b = br // R
            r = br - b * R

            tloc = -1
            E = -1
            s = 0
            qlast = -1
            nu_local = -1

            for t in range(T):
                tloc += 1
                q = int(q_btr[b, t, r])
                k = int(k_btr[b, t, r])
                v = int(v_btr[b, t, r])

                if E == -1 or k != run_sym[br, E]:
                    E += 1
                    run_sym[br, E] = k
                    run_start[br, E] = tloc
                    _sam_extend_nb(sam_next[br], sam_link[br], sam_len[br], sam_e[br],
                                   sam_last[br:br+1], sam_size[br:br+1], k, E)
                    _store_bits_at_row(v_bits_runs[br, E], v, J)

                if q != qlast:
                    ns = _sam_match_next_from_nb(sam_next[br], sam_link[br], s, q)
                    if ns != -1:
                        s = ns
                        nu_local = sam_e[br, s] + 1
                    else:
                        s = -1
                        nu_local = -1
                    qlast = q

                val = 1 if (nu_local != -1 and nu_local <= E) else 0
                out_valid[b, r, t] = val
                if val == 1:
                    for j in range(J):
                        out_bits[b, t, r, j] = v_bits_runs[br, nu_local, j]

            E_idx[br]   = E
            s_q[br]     = s
            q_last[br]  = qlast
            nu[br]      = nu_local
            t_idx[br]   = tloc

        return out_bits, out_valid

    @nb.njit(cache=True, fastmath=False, parallel=True)
    def _step_kernel_nb(q_br, k_br, v_br,
                        sam_next, sam_link, sam_len, sam_e, sam_size, sam_last,
                        run_start, run_sym, v_bits_runs, E_idx,
                        s_q, q_last, nu, t_idx,
                        J, K):
        N = q_br.shape[0]
        out_bits = np.zeros((N, J), dtype=np.int32)
        out_valid = np.zeros((N,), dtype=np.int32)

        for br in nb.prange(N):
            tloc = t_idx[br] + 1
            q = int(q_br[br]); k = int(k_br[br]); v = int(v_br[br])
            E = E_idx[br]; s = s_q[br]; ql = q_last[br]; nu_local = nu[br]

            if E == -1 or k != run_sym[br, E]:
                E += 1
                run_sym[br, E] = k
                run_start[br, E] = tloc
                _sam_extend_nb(sam_next[br], sam_link[br], sam_len[br], sam_e[br],
                               sam_last[br:br+1], sam_size[br:br+1], k, E)
                _store_bits_at_row(v_bits_runs[br, E], v, J)

            if q != ql:
                ns = _sam_match_next_from_nb(sam_next[br], sam_link[br], s, q)
                if ns != -1:
                    s = ns
                    nu_local = sam_e[br, s] + 1
                else:
                    s = -1
                    nu_local = -1
                ql = q

            val = 1 if (nu_local != -1 and nu_local <= E) else 0
            out_valid[br] = val
            if val == 1:
                for j in range(J):
                    out_bits[br, j] = v_bits_runs[br, nu_local, j]

            t_idx[br] = tloc
            E_idx[br] = E
            s_q[br]   = s
            q_last[br]= ql
            nu[br]    = nu_local

        return out_bits, out_valid

import torch

class _BrosaOnlineEngineNB:
    def __init__(self, B: int, R: int, J: int, K: int):
        self.B = int(B); self.R = int(R); self.J = int(J); self.K = int(K)
        self.N = int(B*R)
        self._cap_runs  = 1024
        self._cap_state = 2048
        self._alloc_all()

    def reset_for_prefill(self, B: int, R: int, T: int):
        self.B = int(B); self.R = int(R); self.N = int(B*R)
        cap_r = max(8, T + 4)
        cap_s = max(16, 2*T + 8)
        self._cap_runs = cap_r
        self._cap_state = cap_s
        self._alloc_all()

    def _alloc_all(self):
        N, K, J = self.N, self.K, self.J
        Rcap, Scap = self._cap_runs, self._cap_state

        self.sam_next   = np.full((N, Scap, K), -1, dtype=np.int32)
        self.sam_link   = np.full((N, Scap),    -1, dtype=np.int32)
        self.sam_len    = np.zeros((N, Scap),       dtype=np.int32)
        self.sam_e      = np.full((N, Scap),    -1, dtype=np.int32)
        self.sam_size   = np.ones((N,), dtype=np.int32)
        self.sam_last   = np.zeros((N,), dtype=np.int32)

        self.run_start  = np.full((N, Rcap), -1, dtype=np.int32)
        self.run_sym    = np.full((N, Rcap), -1, dtype=np.int32)
        self.v_bits_runs= np.zeros((N, Rcap, J), dtype=np.int32)

        self.E_idx      = np.full((N,), -1, dtype=np.int32)
        self.s_q        = np.zeros((N,), dtype=np.int32)
        self.q_last     = np.full((N,), -1, dtype=np.int32)
        self.nu         = np.full((N,), -1, dtype=np.int32)
        self.t_idx      = np.full((N,), -1, dtype=np.int32)

    def _maybe_grow_for_step(self, may_new_run: bool = True, may_new_states_per_ext: int = 2):
        grow = False
        if may_new_run:
            need = self.E_idx.max() + 1
            if need >= self._cap_runs:
                self._grow_runs(max(self._cap_runs*2, need+8))
                grow = True
        need_s = self.sam_size.max() + may_new_states_per_ext
        if need_s >= self._cap_state:
            self._grow_states(max(self._cap_state*2, need_s+16))
            grow = True
        return grow

    def _grow_runs(self, new_cap):
        N, J = self.N, self.J
        Rcap = int(new_cap)
        run_start_new   = np.full((N, Rcap), -1, dtype=np.int32)
        run_sym_new     = np.full((N, Rcap), -1, dtype=np.int32)
        v_bits_runs_new = np.zeros((N, Rcap, J), dtype=np.int32)
        run_start_new[:, :self.run_start.shape[1]] = self.run_start
        run_sym_new[:,   :self.run_sym.shape[1]]   = self.run_sym
        v_bits_runs_new[:,:self.v_bits_runs.shape[1],:] = self.v_bits_runs
        self.run_start, self.run_sym, self.v_bits_runs = run_start_new, run_sym_new, v_bits_runs_new
        self._cap_runs = Rcap

    def _grow_states(self, new_cap):
        N, K = self.N, self.K
        Scap = int(new_cap)
        sam_next_new = np.full((N, Scap, K), -1, dtype=np.int32)
        sam_link_new = np.full((N, Scap),    -1, dtype=np.int32)
        sam_len_new  = np.zeros((N, Scap),       dtype=np.int32)
        sam_e_new    = np.full((N, Scap),    -1, dtype=np.int32)
        S0 = self.sam_next.shape[1]
        sam_next_new[:, :S0, :] = self.sam_next
        sam_link_new[:, :S0]    = self.sam_link
        sam_len_new[:,  :S0]    = self.sam_len
        sam_e_new[:,    :S0]    = self.sam_e
        self.sam_next, self.sam_link, self.sam_len, self.sam_e = (
            sam_next_new, sam_link_new, sam_len_new, sam_e_new
        )
        self._cap_state = Scap

    def prefill_numpy(self, q_btr_np, k_btr_np, v_btr_np):
        if not _NUMBA_OK:
            raise RuntimeError("Numba not available")
        B, T, R = q_btr_np.shape
        assert R == self.R and B == self.B
        self.reset_for_prefill(B, R, T)
        out_bits_btrj, out_valid_brt = _prefill_kernel_nb(
            q_btr_np, k_btr_np, v_btr_np,
            self.sam_next, self.sam_link, self.sam_len, self.sam_e, self.sam_size, self.sam_last,
            self.run_start, self.run_sym, self.v_bits_runs, self.E_idx,
            self.s_q, self.q_last, self.nu, self.t_idx,
            self.J, self.K
        )
        C = self.R * self.J
        bits_btc = out_bits_btrj.transpose(0,1,2,3).reshape(B, T, C)
        return bits_btc, out_valid_brt

    def step_numpy(self, q_br_np, k_br_np, v_br_np):
        if not _NUMBA_OK:
            raise RuntimeError("Numba not available")
        B, R = q_br_np.shape
        assert B == self.B and R == self.R
        self._maybe_grow_for_step(may_new_run=True, may_new_states_per_ext=2)

        q_flat = q_br_np.reshape(self.N)
        k_flat = k_br_np.reshape(self.N)
        v_flat = v_br_np.reshape(self.N)
        bits_nj, valid_n = _step_kernel_nb(
            q_flat, k_flat, v_flat,
            self.sam_next, self.sam_link, self.sam_len, self.sam_e, self.sam_size, self.sam_last,
            self.run_start, self.run_sym, self.v_bits_runs, self.E_idx,
            self.s_q, self.q_last, self.nu, self.t_idx,
            self.J, self.K
        )
        B, R, J = self.B, self.R, self.J
        C = R * J
        bits_b1c = bits_nj.reshape(B, R, J).transpose(0,2,1).reshape(B, 1, C)
        valid_br1 = valid_n.reshape(B, R, 1)
        return bits_b1c, valid_br1

def _ensure_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if DEVICE_FALLBACK_CPU:
        return torch.device("cpu")
    raise RuntimeError("CUDA device required")

DEVICE = _ensure_device()
DTYPE_INFER = torch.bfloat16 if (USE_BF16 and DEVICE.type == "cuda") else torch.float16

def _pack_bits_btC_to_btr(bits_btC: torch.Tensor, Mbits: int) -> torch.Tensor:
    B, T, C = bits_btC.shape
    assert C % Mbits == 0
    R = C // Mbits
    x = bits_btC.view(B, T, R, Mbits).to(torch.int32)
    out = torch.zeros((B, T, R), dtype=torch.int32, device=bits_btC.device)
    for j in range(Mbits):
        out |= ((x[..., j] & 1) << j)
    return out

class _SAM:
    __slots__ = ("K", "next", "link", "length", "endpos", "last")
    def __init__(self, K: int):
        self.K = int(K)
        self.next   = [[-1]*self.K]
        self.link   = [-1]
        self.length = [0]
        self.endpos = [-1]
        self.last   = 0

    def _new_state(self, L: int) -> int:
        self.next.append([-1]*self.K)
        self.link.append(-1)
        self.length.append(int(L))
        self.endpos.append(-1)
        return len(self.next)-1

    def extend(self, x: int, pos: int):
        x = int(x)
        last = self.last
        cur = self._new_state(self.length[last] + 1)
        p = last
        while p != -1 and self.next[p][x] == -1:
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
                self.next[clone] = self.next[q][:]
                self.link[clone] = self.link[q]
                self.endpos[clone] = self.endpos[q]
                while p != -1 and self.next[p][x] == q:
                    self.next[p][x] = clone
                    p = self.link[p]
                self.link[q] = self.link[cur] = clone
        v = cur
        while v != -1 and self.endpos[v] != pos:
            self.endpos[v] = pos
            v = self.link[v]
        self.last = cur

    def match_next_from(self, s: int, x: int) -> int:
        p = int(s)
        while p != -1 and self.next[p][x] == -1:
            p = self.link[p]
        return -1 if p == -1 else self.next[p][x]

from typing import List, Tuple
import numpy as np
import torch

def _int_to_bits_list(x: int, J: int) -> List[int]:
    return [ (int(x) >> j) & 1 for j in range(J) ]

class _RouteState:
    __slots__ = ("sam", "E", "S", "Gamma", "q_last", "s_q", "nu", "v_bits_runs")
    def __init__(self, K: int):
        self.sam = _SAM(K)
        self.E = -1
        self.S: List[int] = []
        self.Gamma: List[int] = []
        self.q_last = None
        self.s_q = 0
        self.nu = -1
        self.v_bits_runs: List[List[int]] = []

class _BrosaOnlineState:
    def __init__(self, B: int, R: int, J: int, K: int, device: torch.device):
        self.B = int(B); self.R = int(R); self.J = int(J); self.K = int(K)
        self.device = device
        self.routes: List[List[_RouteState]] = [[_RouteState(K) for _ in range(R)] for _ in range(B)]
        self.t: List[int] = [ -1 for _ in range(B) ]

    def _ensure_shapes(self, B: int, R: int):
        if B != self.B or R != self.R:
            self.__init__(B, R, self.J, self.K, self.device)

    def _process_one_step(self,
                          b: int,
                          t_global: int,
                          q_cat_r: np.ndarray,
                          k_cat_r: np.ndarray,
                          v_cat_r: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
        R, J = self.R, self.J
        valid_br = np.zeros((R,), dtype=np.int32)
        b_bits_c = np.zeros((R*J,), dtype=np.int32)

        for r in range(R):
            rs: _RouteState = self.routes[b][r]
            k = int(k_cat_r[r]); q = int(q_cat_r[r]); v = int(v_cat_r[r])

            if rs.E == -1 or k != rs.Gamma[rs.E]:
                rs.E += 1
                rs.Gamma.append(k)
                rs.S.append(t_global)
                rs.sam.extend(k, rs.E)
                rs.v_bits_runs.append(_int_to_bits_list(v, J))

            if rs.q_last is None or q != rs.q_last:
                ns = rs.sam.match_next_from(rs.s_q, q)
                if ns != -1:
                    rs.s_q = ns
                    rs.nu = rs.sam.endpos[rs.s_q] + 1
                else:
                    rs.s_q = -1
                    rs.nu = -1
                rs.q_last = q

            if rs.nu != -1 and rs.nu <= rs.E:
                valid_br[r] = 1
                bits = rs.v_bits_runs[rs.nu]
                base = r*J
                for j in range(J):
                    b_bits_c[base+j] = bits[j]
            else:
                pass

        return valid_br, b_bits_c

    def prefill(self,
                q_cat_btr: torch.Tensor,
                k_cat_btr: torch.Tensor,
                v_cat_btr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, R = q_cat_btr.shape
        self._ensure_shapes(B, R)

        q_np = q_cat_btr.detach().to("cpu", non_blocking=False).numpy()
        k_np = k_cat_btr.detach().to("cpu", non_blocking=False).numpy()
        v_np = v_cat_btr.detach().to("cpu", non_blocking=False).numpy()

        C = R * self.J
        b_bits = np.zeros((B, T, C), dtype=np.int32)
        valid  = np.zeros((B, R, T), dtype=np.int32)

        for b in range(B):
            self.t[b] = -1
            for t in range(T):
                self.t[b] += 1
                vb, cb = self._process_one_step(
                    b=b,
                    t_global=self.t[b],
                    q_cat_r=q_np[b, t],
                    k_cat_r=k_np[b, t],
                    v_cat_r=v_np[b, t]
                )
                valid[b, :, t]   = vb
                b_bits[b, t, :]  = cb

        b_bits_t = torch.from_numpy(b_bits).to(self.device, dtype=torch.int32)
        valid_t  = torch.from_numpy(valid).to(self.device, dtype=torch.int32)
        return b_bits_t, valid_t

    def step(self,
             q_cat_btR_last: torch.Tensor,
             k_cat_btR_last: torch.Tensor,
             v_cat_btR_last: torch.Tensor
             ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, R = q_cat_btR_last.shape
        self._ensure_shapes(B, R)

        q_np = q_cat_btR_last.detach().to("cpu").numpy()
        k_np = k_cat_btR_last.detach().to("cpu").numpy()
        v_np = v_cat_btR_last.detach().to("cpu").numpy()

        C = R * self.J
        b_bits = np.zeros((B, 1, C), dtype=np.int32)
        valid  = np.zeros((B, R, 1), dtype=np.int32)

        for b in range(B):
            self.t[b] = (self.t[b] + 1) if (self.t[b] is not None) else 0
            vb, cb = self._process_one_step(
                b=b,
                t_global=self.t[b],
                q_cat_r=q_np[b],
                k_cat_r=k_np[b],
                v_cat_r=v_np[b]
            )
            valid[b, :, 0]  = vb
            b_bits[b, 0, :] = cb

        b_bits_t = torch.from_numpy(b_bits).to(self.device, dtype=torch.int32)
        valid_t  = torch.from_numpy(valid).to(self.device, dtype=torch.int32)
        return b_bits_t, valid_t

def _sam_new_state_py(next_arr, link, length, e, size, L):
    s = size[0]; size[0] = s + 1
    length[s] = L; link[s] = -1; e[s] = -1
    next_arr[s, :next_arr.shape[1]] = -1
    return s

def _sam_extend_py(next_arr, link, length, e, last_arr, size, x, pos):
    last = last_arr[0]
    cur = _sam_new_state_py(next_arr, link, length, e, size, length[last] + 1)
    p = last
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
            clone = _sam_new_state_py(next_arr, link, length, e, size, length[p] + 1)
            next_arr[clone, :] = next_arr[q, :]
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

def _sam_match_next_from_py(next_arr, link, s, x):
    p = int(s)
    while p != -1 and next_arr[p, x] == -1:
        p = link[p]
    return -1 if p == -1 else int(next_arr[p, x])

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3DecoderLayer

def patch_qwen3_with_multiroute_rosa_infer_online_nb(
    model: Qwen3ForCausalLM,
    bits_per_route: int,
    inject_mode: str,
    apply_from_layer: int
):
    inject_mode = inject_mode.lower()
    assert inject_mode in ("pre_attn", "post_attn")
    C = int(model.config.hidden_size)
    J = int(bits_per_route)
    assert C % J == 0
    R = C // J
    K = 1 << J

    base_param = model.model.embed_tokens.weight
    base_dtype = base_param.dtype
    base_device = base_param.device

    def _pack_bits_btC_to_btr(bits_btC: torch.Tensor, J: int) -> torch.Tensor:
        B, T, Cx = bits_btC.shape
        assert Cx % J == 0
        Rloc = Cx // J
        x = bits_btC.view(B, T, Rloc, J).to(torch.int32)
        out = torch.zeros((B, T, Rloc), dtype=torch.int32, device=bits_btC.device)
        for j in range(J):
            out |= ((x[..., j] & 1) << j)
        return out

    pool = _get_rosa_thread_pool()

    for li, layer in enumerate(model.model.layers):
        if li < apply_from_layer:
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

        layer.rosa_bits_per_route = J
        layer.rosa_num_routes = R
        layer.rosa_k_symbols = K
        layer.rosa_inject_mode = inject_mode

        layer._brosa_nb: _BrosaOnlineEngineNB | None = None

        def _forward_brosa_online_nb(
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

            B, T, Cx = hidden_states.shape

            assert Cx == C
            device = hidden_states.device
            dtype  = hidden_states.dtype
            residual = hidden_states

            if (self._brosa_nb is None) or (self._brosa_nb.B != B):
                self._brosa_nb = _BrosaOnlineEngineNB(B=B, R=R, J=J, K=K)

            u = self.input_layernorm(hidden_states)
            q_vec = self.rosa_q_proj(u)
            k_vec = self.rosa_k_proj(u)
            v_vec = self.rosa_v_proj(u)

            if past_key_values is None:
                q_cat = _pack_bits_btC_to_btr((q_vec > 0).to(torch.int32), J)
                k_cat = _pack_bits_btC_to_btr((k_vec > 0).to(torch.int32), J)
                v_cat = _pack_bits_btC_to_btr((v_vec > 0).to(torch.int32), J)

                q_np = q_cat.detach().to("cpu").numpy()
                k_np = k_cat.detach().to("cpu").numpy()
                v_np = v_cat.detach().to("cpu").numpy()

                def _cpu_job_prefill():
                    if _NUMBA_OK:
                        try: nb.set_num_threads(int(ROSA_NUMBA_THREADS))
                        except Exception: pass
                    return self._brosa_nb.prefill_numpy(q_np, k_np, v_np)

                if self.rosa_inject_mode == "post_attn" and pool is not None:
                    fut = pool.submit(_cpu_job_prefill)
                    attn_out, _ = self.self_attn(
                        hidden_states=u, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs
                    )
                    bits_btc_np, valid_brt_np = fut.result()
                else:
                    bits_btc_np, valid_brt_np = _cpu_job_prefill()
                    attn_out = None
                    if self.rosa_inject_mode == "post_attn":
                        attn_out, _ = self.self_attn(
                            hidden_states=u, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, cache_position=cache_position,
                            position_embeddings=position_embeddings, **kwargs
                        )

                Cfull = R * J
                bits_btc = torch.from_numpy(bits_btc_np).to(device=device, dtype=torch.int32)
                valid_brt = torch.from_numpy(valid_brt_np).to(device=device, dtype=torch.bool)

                delta = (self.rosa_e1 - self.rosa_e0).to(dtype)
                y_valid = self.rosa_e0.view(1,1,C).to(dtype) + delta.view(1,1,C) * bits_btc.to(dtype)
                mask_btc = valid_brt.permute(0,2,1).unsqueeze(-1).expand(B, T, R, J).reshape(B, T, Cfull).to(dtype)
                y = y_valid * mask_btc
                inj = self.rosa_out(y)

                if self.rosa_inject_mode == "post_attn":
                    hidden_states = residual + (attn_out + inj)
                else:
                    alpha = torch.sigmoid(self.rosa_alpha).view(1,1,C).to(device=device, dtype=dtype)
                    mix = (1.0 - alpha) * hidden_states + alpha * inj
                    u2 = self.input_layernorm(mix)
                    attn_out2, _ = self.self_attn(
                        hidden_states=u2, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs
                    )
                    hidden_states = residual + attn_out2

            else:
                q_cat_last = _pack_bits_btC_to_btr((q_vec[:, -1:, :] > 0).to(torch.int32), J)[:, 0, :]
                k_cat_last = _pack_bits_btC_to_btr((k_vec[:, -1:, :] > 0).to(torch.int32), J)[:, 0, :]
                v_cat_last = _pack_bits_btC_to_btr((v_vec[:, -1:, :] > 0).to(torch.int32), J)[:, 0, :]

                q_np = q_cat_last.detach().to("cpu").numpy()
                k_np = k_cat_last.detach().to("cpu").numpy()
                v_np = v_cat_last.detach().to("cpu").numpy()

                def _cpu_job_step():
                    if _NUMBA_OK:
                        try: nb.set_num_threads(int(ROSA_NUMBA_THREADS))
                        except Exception: pass
                    return self._brosa_nb.step_numpy(q_np, k_np, v_np)

                if self.rosa_inject_mode == "post_attn" and pool is not None:
                    fut = pool.submit(_cpu_job_step)
                    attn_out, _ = self.self_attn(
                        hidden_states=u, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs
                    )
                    bits_bt1c_np, valid_br1_np = fut.result()
                else:
                    bits_bt1c_np, valid_br1_np = _cpu_job_step()
                    attn_out = None
                    if self.rosa_inject_mode == "post_attn":
                        attn_out, _ = self.self_attn(
                            hidden_states=u, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=past_key_values,
                            use_cache=use_cache, cache_position=cache_position,
                            position_embeddings=position_embeddings, **kwargs
                        )

                Cfull = R * J
                bits_bt1c = torch.from_numpy(bits_bt1c_np).to(device=device, dtype=torch.int32)
                valid_br1 = torch.from_numpy(valid_br1_np).to(device=device, dtype=torch.bool)

                delta = (self.rosa_e1 - self.rosa_e0).to(dtype)
                y_valid = self.rosa_e0.view(1,1,C).to(dtype) + delta.view(1,1,C) * bits_bt1c.to(dtype)
                mask_bt1c = valid_br1.permute(0,2,1).unsqueeze(-1).expand(B, 1, R, J).reshape(B, 1, Cfull).to(dtype)
                y = y_valid * mask_bt1c
                inj = self.rosa_out(y)

                if self.rosa_inject_mode == "post_attn":
                    hidden_states = residual + (attn_out + inj)
                else:
                    alpha = torch.sigmoid(self.rosa_alpha).view(1,1,C).to(device=device, dtype=dtype)
                    mix = (1.0 - alpha) * hidden_states + alpha * inj
                    u2 = self.input_layernorm(mix)
                    attn_out2, _ = self.self_attn(
                        hidden_states=u2, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_values=past_key_values,
                        use_cache=use_cache, cache_position=cache_position,
                        position_embeddings=position_embeddings, **kwargs
                    )
                    hidden_states = residual + attn_out2

            residual2 = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual2 + hidden_states
            return hidden_states

        layer.forward = _forward_brosa_online_nb.__get__(layer, Qwen3DecoderLayer)

    print(f"[patched] layers_from={apply_from_layer}, J={J}, R={R}, K={K}, "
          f"inject={inject_mode}, numba_threads={ROSA_NUMBA_THREADS}, overlap_workers={ROSA_THREAD_WORKERS}")

def _maybe_load_json(path: str | None) -> dict | None:
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def build_model_and_tokenizer() -> Tuple[Qwen3ForCausalLM, AutoTokenizer]:
    global BITS_PER_ROUTE, ROSA_INJECT_MODE, APPLY_FROM_LAYER, ATTN_WINDOW, FIRST_GLOBAL_LAYERS

    rosa_meta = _maybe_load_json(ROSA_META_JSON_PATH)
    if rosa_meta:
        BITS_PER_ROUTE   = int(rosa_meta.get("bits_per_route", BITS_PER_ROUTE))
        ROSA_INJECT_MODE = str(rosa_meta.get("inject_mode", ROSA_INJECT_MODE))
        APPLY_FROM_LAYER = int(rosa_meta.get("apply_layers_from", APPLY_FROM_LAYER))

    run_meta = _maybe_load_json(RUN_META_JSON_PATH)
    if run_meta:
        ATTN_WINDOW         = int(run_meta.get("attn_window", ATTN_WINDOW))
        FIRST_GLOBAL_LAYERS = int(run_meta.get("first_global_layers", FIRST_GLOBAL_LAYERS))

    cfg = AutoConfig.from_pretrained(MODEL_LOCAL_DIR)
    cfg.tie_word_embeddings = TIE
    cfg.sliding_window = ATTN_WINDOW
    cfg.max_window_layers = FIRST_GLOBAL_LAYERS
    if (not hasattr(cfg, "layer_types")) or (cfg.layer_types is None):
        cfg.layer_types = [
            "full_attention" if i < cfg.max_window_layers else "sliding_attention"
            for i in range(cfg.num_hidden_layers)
        ]
    if hasattr(cfg, "attn_implementation"):
        cfg.attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "sdpa"
    else:
        cfg._attn_implementation = "flash_attention_2" if USE_FLASH_ATTN else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, use_fast=True)

    model = Qwen3ForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        config=cfg,
        torch_dtype=DTYPE_INFER,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()

    patch_qwen3_with_multiroute_rosa_infer_online_nb(
        model,
        bits_per_route=BITS_PER_ROUTE,
        inject_mode=ROSA_INJECT_MODE,
        apply_from_layer=APPLY_FROM_LAYER,
    )

    if ROSA_ADAPTER_PATH and os.path.isfile(ROSA_ADAPTER_PATH):
        try:
            state_dict = {}
            if _HAS_SAFETENSORS and (ROSA_ADAPTER_PATH.endswith(".safetensors") or True):
                with safe_open(ROSA_ADAPTER_PATH, framework="pt") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
                print(f"[adapter] loaded via safetensors: {ROSA_ADAPTER_PATH}")
            else:
                raise RuntimeError("safetensors not available")

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

        except Exception as e:
            print(f"[warn] safetensors failed ({e}); fallback to torch.load")
            state = torch.load(ROSA_ADAPTER_PATH, map_location="cpu")
            missing, unexpected = model.load_state_dict(state, strict=False)

        if len(unexpected) > 0:
            print(f"[warn] unexpected keys: {len(unexpected)}")
        if len(missing) > 0:
            print(f"[warn] missing keys: {len(missing)}")
        else:
            print("[adapter] loaded")

    else:
        print("[adapter] not provided or file not found")

    return model, tokenizer

@torch.inference_mode()
def _sample_from_logits(logits: torch.Tensor,
                        temperature: float = 1.0,
                        top_p: float = 1.0,
                        top_k: int = 0,
                        do_sample: bool = True) -> int:
    if temperature <= 0.0: temperature = 1.0
    logits = logits / temperature
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_keep = values[-1]
        logits = torch.where(logits < min_keep, torch.full_like(logits, float("-inf")), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        cutoff = (cumsum > top_p).float().argmax(dim=-1)
        cutoff = max(int(cutoff.item()), 1)
        keep_idx = sorted_indices[:cutoff]
        mask = torch.full_like(logits, float("-inf"))
        mask[keep_idx] = logits[keep_idx]
        logits = mask
    if do_sample:
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())
    else:
        return int(torch.argmax(logits, dim=-1).item())

import torch
import torch.nn.functional as F

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    do_sample: bool = False,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    eos_token_id: int | None = None,
    add_special_tokens: bool = False,
) -> str:
    device = next(model.parameters()).device
    model.eval()

    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    def _apply_repetition_penalty_(logits: torch.Tensor, generated_ids: list[int], penalty: float):
        if penalty == 1.0 or len(generated_ids) == 0:
            return logits
        unique_ids = set(generated_ids)
        if len(unique_ids) == 0:
            return logits
        idx = torch.tensor(list(unique_ids), device=logits.device, dtype=torch.long)
        sel = logits.index_select(0, idx)
        sel = torch.where(sel > 0, sel / penalty, sel * penalty)
        logits.index_copy_(0, idx, sel)
        return logits

    def _sample_from_logits_(logits: torch.Tensor) -> int:
        if do_sample:
            t = max(1e-5, float(temperature))
            logits = logits / t

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            thresh = torch.topk(logits, k).values[-1]
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)

        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            cutoff = torch.searchsorted(cumsum, torch.tensor(top_p, device=logits.device)).clamp(min=1)
            mask = torch.full_like(logits, float("-inf"))
            keep_idx = sorted_idx[:cutoff]
            mask.scatter_(0, keep_idx, logits.index_select(0, keep_idx))
            logits = mask

        if do_sample:
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        else:
            next_id = int(torch.argmax(logits).item())
        return next_id

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=add_special_tokens)
    input_ids = enc["input_ids"].to(device)
    out = model(input_ids=input_ids, use_cache=True)
    past = getattr(out, "past_key_values", None)
    if past is None and hasattr(out, "cache"):
        past = out.cache

    generated = input_ids[0].tolist()

    next_logits = out.logits[0, -1, :]

    for _ in range(max_new_tokens):
        if repetition_penalty != 1.0:
            next_logits = _apply_repetition_penalty_(next_logits, generated, repetition_penalty)

        next_id = _sample_from_logits_(next_logits)

        generated.append(next_id)

        if eos_token_id is not None and next_id == eos_token_id:
            break

        step_ids = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
        out = model(input_ids=step_ids, use_cache=True, past_key_values=past)
        past = getattr(out, "past_key_values", None)
        if past is None and hasattr(out, "cache"):
            past = out.cache

        next_logits = out.logits[0, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=True)

def main():
    torch.manual_seed(SEED)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if _NUMBA_OK and ROSA_NUMBA_PARALLEL:
        try: nb.set_num_threads(int(ROSA_NUMBA_THREADS))
        except Exception: pass

    model, tokenizer = build_model_and_tokenizer()

    demo = "Explain the concept of attention mechanism in transformers."
    t0 = time.time()
    text = generate(model, tokenizer, demo)
    dt = time.time() - t0

    print("\n=== Prompt ===\n", demo)
    print("\n=== Output ===\n", text)
    print(f"\n[Done] elapsed: {dt:.3f}s")

if __name__ == "__main__":
    main()
