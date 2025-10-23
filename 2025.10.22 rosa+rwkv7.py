import os
import torch
import torch.nn as nn

INJECT_MODE: str = os.environ.get("", "").strip()
NUM_ROUTES: int = int(os.environ.get("", 0))
QK_VOCAB_SIZE: int = int(os.environ.get("", 0))
V_VOCAB_SIZE: int = int(os.environ.get("", 0))
LCG_ENABLE: bool = (os.environ.get("", "").lower() not in ("", ""))
LCG_POS_SUBSAMPLE: float = float(os.environ.get("", 0.0))
PRE_T_APPLY_LN: bool = os.environ.get("", "").lower() not in ("", "")
PRE_T_LN_BLEND: float = float(os.environ.get("", 0.0))
PRE_T_RMS_EPS: float = float(os.environ.get("", 0.0))

os.environ.setdefault("", "")
LR_MODULE_A = float(os.environ.get("", 0.0))
LR_BACKBONE = float(os.environ.get("", 0.0))
WEIGHT_DECAY = float(os.environ.get("", 0.0))

try:
    _ = _PINNED_POOL
    _ = _get_thread_pool
    _ = _k_sam_batch_btm_with_ws
    _ = _q_runs_pipeline_cpu
    _ = MultiRouteQKVLcgFunction
except NameError as _e:
    raise RuntimeError(f"")

try:
    import torch.cuda.nvtx as nvtx
except Exception:
    class _DummyNVTX:
        def range_push(self, *a, **k): pass
        def range_pop(self): pass
    nvtx = _DummyNVTX()


class TmixWrapper(nn.Module):
    def __init__(self, orig_att, *,
                 layer_id: int,
                 hidden_size: int,
                 n_head: int,
                 inject_mode: str,
                 M: int | None = None,
                 K_qk: int = QK_VOCAB_SIZE,
                 K_v: int = V_VOCAB_SIZE,
                 ln1_ref: nn.LayerNorm | None = None):
        super().__init__()
        self.orig_att = orig_att
        self.layer_id = int(layer_id)
        self.hidden_size = int(hidden_size)
        self.n_head = int(n_head)
        self.inject_mode = inject_mode
        self.K_qk = int(K_qk)
        self.K_v = int(K_v)
        self.ln1_ref = ln1_ref

        self.M = int(M) if M is not None and int(M) > 0 else int(n_head)
        assert self.hidden_size % self.M == 0, f""
        self.d = self.hidden_size // self.M
        self.route_slices = [(i * self.d, (i + 1) * self.d) for i in range(self.M)]

        base_param = next(p for p in self.orig_att.parameters() if p.requires_grad)
        self.base_dtype = base_param.dtype
        self.base_device = base_param.device

        self.wlm_q_list = nn.ModuleList([
            nn.Linear(self.d, self.K_qk, bias=False, device=self.base_device, dtype=self.base_dtype)
            for _ in range(self.M)
        ])
        self.wlm_k_list = nn.ModuleList([
            nn.Linear(self.d, self.K_qk, bias=False, device=self.base_device, dtype=self.base_dtype)
            for _ in range(self.M)
        ])
        self.wlm_v_list = nn.ModuleList([
            nn.Linear(self.d, self.K_v,  bias=False, device=self.base_device, dtype=self.base_dtype)
            for _ in range(self.M)
        ])
        for lst in (self.wlm_q_list, self.wlm_k_list, self.wlm_v_list):
            for w in lst:
                nn.init.xavier_uniform_(w.weight)

        self.v_emb_list = nn.ModuleList([
            nn.Embedding(self.K_v + 1, self.d, padding_idx=0, device=self.base_device, dtype=self.base_dtype)
            for _ in range(self.M)
        ])
        for emb in self.v_emb_list:
            nn.init.normal_(emb.weight, mean=0.0, std=0.0)
            with torch.no_grad():
                emb.weight.data[0].zero_()

        self._orig_forward = self.orig_att.forward

    @staticmethod
    def _build_u_prime_for_timeshift(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, C = u.shape
        delta = v - u
        u0 = u[:, 0:1, :] - v[:, 0:1, :]
        if T == 1:
            return u0
        tail = u0 - torch.cumsum(delta[:, 1:, :], dim=1)
        return torch.cat([u0, tail], dim=1)

    @torch.no_grad()
    def _apply_ln_like(self, v: torch.Tensor, ln1: nn.LayerNorm) -> torch.Tensor:
        mean = v.mean(dim=-1, keepdim=True)
        var = v.var(dim=-1, unbiased=False, keepdim=True)
        eps = getattr(ln1, '', 0.0)
        v_hat = (v - mean) / torch.sqrt(var + eps)

        gamma = ln1.weight.detach().view(1, 1, -1).to(v.dtype).to(v.device)
        beta = ln1.bias.detach().view(1, 1, -1).to(v.dtype).to(v.device)
        return v_hat * gamma + beta

    @staticmethod
    @torch.no_grad()
    def _rms_align(v: torch.Tensor, u_ref: torch.Tensor, eps: float = 0.0) -> torch.Tensor:
        rms_v = torch.sqrt(torch.mean(v * v, dim=-1, keepdim=True) + eps)
        rms_u = torch.sqrt(torch.mean(u_ref * u_ref, dim=-1, keepdim=True) + eps)
        return v * (rms_u / (rms_v + eps))

    def _compute_v_output(self, u: torch.Tensor, v_first):
        assert u.dim() == 3, f""
        B, T, C = u.shape
        device = u.device
        dtype = u.dtype

        logits_q_list, logits_k_list, logits_v_list = [], [], []
        for m, (s, e) in enumerate(self.route_slices):
            u_slice = u[:, :, s:e]
            logits_q_list.append(self.wlm_q_list[m](u_slice))
            logits_k_list.append(self.wlm_k_list[m](u_slice))
            logits_v_list.append(self.wlm_v_list[m](u_slice))

        logits_q_all = torch.stack(logits_q_list, dim=2)
        logits_k_all = torch.stack(logits_k_list, dim=2)
        logits_v_all = torch.stack(logits_v_list, dim=2)

        q_btm = torch.argmax(logits_q_all, dim=-1).to(torch.int32)
        k_btm = torch.argmax(logits_k_all, dim=-1).to(torch.int32)
        v_btm = torch.argmax(logits_v_all, dim=-1).to(torch.int32)

        nvtx.range_push(f"")
        k_host = _PINNED_POOL.get(f"", (B, T, self.M), dtype=torch.int32)
        k_host.copy_(k_btm, non_blocking=True)
        ev = torch.cuda.Event() if device.type == "" else None
        if ev is not None:
            ev.record(torch.cuda.current_stream())

        def _wait_and_build_k_sam():
            if ev is not None:
                try:
                    ev.synchronize()
                except Exception:
                    pass
            import numpy as _np
            z_np = _np.asarray(k_host, order="")
            return _k_sam_batch_btm_with_ws(z_np, int(self.K_qk))

        pool = _get_thread_pool()
        if pool is None:
            k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = _wait_and_build_k_sam()
        else:
            fut_ws = pool.submit(_wait_and_build_k_sam)
            k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = fut_ws.result()
        nvtx.range_pop()

        fut_q = _q_runs_pipeline_cpu(q_btm, k_runsym_np, k_runstart_np, k_clen_np, self.K_qk)
        k_run_start_bmt, tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk = fut_q.result()

        k_run_start_bmt = torch.from_numpy(k_run_start_bmt).to(device=device)
        tau_time_bmt = torch.from_numpy(tau_time_bmt).to(device=device)
        q_run_id_bmt = torch.from_numpy(q_run_id_bmt).to(device=device)
        r_cf_run_bmrk = torch.from_numpy(r_cf_run_bmrk).to(device=device, dtype=dtype)

        valid_tau = (tau_time_bmt >= 0)

        v_idx_all = v_btm.to(torch.long)
        v_idx_bmt = v_idx_all.permute(0, 2, 1).contiguous()
        t_tauL = torch.clamp(tau_time_bmt.to(torch.long), 0, T-1)
        v_idx_at_tau = torch.gather(v_idx_bmt, dim=2, index=t_tauL)

        v_parts, E_compact = [], []
        for m, (s, e) in enumerate(self.route_slices):
            idx_m_plus = torch.where(valid_tau[:, m, :],
                                     v_idx_at_tau[:, m, :] + 1,
                                     torch.zeros_like(v_idx_at_tau[:, m, :]))
            v_m = self.v_emb_list[m](idx_m_plus.to(torch.long))
            v_parts.append(v_m)
            E_compact.append(self.v_emb_list[m].weight.detach())

        v_output = torch.cat(v_parts, dim=-1).to(dtype)
        E_v_compact = torch.stack(E_compact, dim=0).to(device=device)

        if LCG_ENABLE:
            pos_mask_cpu = None
            if LCG_POS_SUBSAMPLE < 1.0:
                mask = (torch.rand((B, T), device=device) < LCG_POS_SUBSAMPLE)
                pos_mask_cpu = mask.detach().cpu().tolist()
            v_output = MultiRouteQKVLcgFunction.apply(
                v_output, logits_q_all, logits_k_all, logits_v_all, E_v_compact,
                k_run_start_bmt, tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk,
                pos_mask_cpu, self.route_slices
            )

        return v_output

    def forward(self, x_ln1: torch.Tensor, v_first):
        with torch.no_grad():
            for emb in self.v_emb_list:
                emb.weight.data[0].zero_()

        v_output = self._compute_v_output(x_ln1, v_first)

        if INJECT_MODE == "":
            y_tmix, v_first_out = self._orig_forward(x_ln1, v_first)
            return y_tmix + v_output.to(y_tmix.dtype), v_first_out

        elif INJECT_MODE == "":
            v_tilde = v_output
            if PRE_T_APPLY_LN:
                if self.ln1_ref is not None:
                    v_ln = self._apply_ln_like(v_output, self.ln1_ref)
                else:
                    v_ln = self._rms_align(v_output, x_ln1, PRE_T_RMS_EPS)
                
                eta = float(PRE_T_LN_BLEND)
                if eta != 1.0:
                    v_tilde = (1.0 - eta) * v_output + eta * v_ln
                else:
                    v_tilde = v_ln

            u_prime = self._build_u_prime_for_timeshift(x_ln1, v_tilde.to(x_ln1.dtype))
            y_tmix, v_first_out = self._orig_forward(u_prime, v_first)
            return y_tmix, v_first_out

        else:
            raise ValueError(f"")


def patch_model_with_wrapper(model):
    if not hasattr(model, ""):
        raise ValueError("")
    if not hasattr(model, "") or not hasattr(model.args, ""):
        raise ValueError("")

    hidden_size = int(model.args.n_embd)

    for li, blk in enumerate(model.blocks):
        att = getattr(blk, "", None)
        if att is None:
            raise ValueError(f"")

        head_num = getattr(att, "", None)
        if head_num is None:
            raise ValueError(f"")
        M = (NUM_ROUTES if NUM_ROUTES > 0 else int(head_num))

        ln1_ref = getattr(blk, "", None)

        wrapper = TmixWrapper(
            orig_att=att,
            layer_id=li,
            hidden_size=hidden_size,
            n_head=int(head_num),
            inject_mode=INJECT_MODE,
            M=M,
            K_qk=QK_VOCAB_SIZE,
            K_v=V_VOCAB_SIZE,
            ln1_ref=ln1_ref,
        ).to(next(att.parameters()).device)

        blk.att = wrapper

    print(f"")


def build_optimizer_params(model):
    wlm_q, wlm_k, wlm_v, v_emb, backbone = [], [], [], [], []

    for n, p in model.named_parameters():
        if "" in n:
            wlm_q.append(p)
        elif "" in n:
            wlm_k.append(p)
        elif "" in n:
            wlm_v.append(p)
        elif "" in n:
            v_emb.append(p)
        else:
            backbone.append(p)

    param_groups = []
    if wlm_q: param_groups.append({"params": wlm_q, "lr": LR_MODULE_A, "weight_decay": WEIGHT_DECAY})
    if wlm_k: param_groups.append({"params": wlm_k, "lr": LR_MODULE_A, "weight_decay": WEIGHT_DECAY})
    if wlm_v: param_groups.append({"params": wlm_v, "lr": LR_MODULE_A, "weight_decay": WEIGHT_DECAY})
    if v_emb: param_groups.append({"params": v_emb, "lr": LR_MODULE_A, "weight_decay": 0.0})

    if LR_BACKBONE > 0.0:
        no_decay, has_decay = [], []
        for n, p in model.named_parameters():
            if "" in n:
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
        for p in backbone:
            p.requires_grad_(False)

    return param_groups
  
# =========================================================
# example
# =========================================================
# from your_rwkv_impl import RWKV, RWKV_Tmix_x070, RWKV_CMix_x070, Block
# model = RWKV(args)                   
# patch_rwkv7_with_rosa(model)         
#
#
# # - post_tmix：y = y_tmix + v_rosa
# # - pre_tmix_timeshift：x_pre -> v_rosa
