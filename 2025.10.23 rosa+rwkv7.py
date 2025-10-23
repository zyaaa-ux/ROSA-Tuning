import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

ROSA_RWKV7_NUM_ROUTES: int = int(os.environ.get("ROSA_NUM_ROUTES", ""))
ROSA_RWKV7_QK_VOCAB: int = int(os.environ.get("ROSA_QK_VOCAB_SIZE", ""))
ROSA_RWKV7_V_VOCAB: int = int(os.environ.get("ROSA_V_VOCAB_SIZE", ""))

ROSA_RWKV7_INJECT_MODE: str = os.environ.get("ROSA_RWKV7_INJECT_MODE", "").lower()
assert ROSA_RWKV7_INJECT_MODE in ("post_tmix", "pre_tmix")

ROSA_RWKV7_APPLY_LAYERS_FROM: int = int(os.environ.get("ROSA_RWKV7_APPLY_LAYERS_FROM", ""))

LCG_ENABLED: bool = os.environ.get("LCG_ENABLE", "").lower() not in ("0", "false")
LCG_POS_SUBSAMPLE: float = float(os.environ.get("LCG_POS_SUBSAMPLE", ""))

RWKV7_ROSA_LR: float = float(os.environ.get("RWKV7_ROSA_LR", ""))
RWKV7_BACKBONE_LR: float = float(os.environ.get("RWKV7_BACKBONE_LR", ""))
RWKV7_ROSA_WEIGHT_DECAY: float = float(os.environ.get("RWKV7_ROSA_WEIGHT_DECAY", ""))

ROSA_RWKV7_OUTPUT_DIR: str = os.environ.get("ROSA_RWKV7_OUTPUT_DIR", "")
ROSA_RWKV7_STATE_NAME: str = os.environ.get("ROSA_RWKV7_STATE_NAME", "")
os.makedirs(ROSA_RWKV7_OUTPUT_DIR, exist_ok=True)

try:
    _get_rosa_thread_pool
    _PINNED_POOL
    nvtx
    _k_sam_batch_btm_with_ws
    _q_runs_pipeline_cpu
    MultiRouteQKVLcgFunction
except NameError as _e:
    raise RuntimeError(
        f"[RWKV7-ROSA] Missing ROSA dependencies: {_e}. Please ensure ROSA implementation is imported."
    )


def _init_rosa_heads_for_att(att_mod: nn.Module, hidden_size: int):
    M = int(ROSA_RWKV7_NUM_ROUTES)
    K_qk = int(ROSA_RWKV7_QK_VOCAB)
    K_v = int(ROSA_RWKV7_V_VOCAB)

    assert hidden_size % M == 0, f"hidden_size ({hidden_size}) must be divisible by M={M}"
    d = hidden_size // M
    route_slices = [(i * d, (i + 1) * d) for i in range(M)]

    base_param = next(att_mod.parameters())
    base_dtype = base_param.dtype
    base_device = base_param.device

    att_mod.rosa_wlm_q_list = nn.ModuleList(
        [nn.Linear(d, K_qk, bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)]
    )
    att_mod.rosa_wlm_k_list = nn.ModuleList(
        [nn.Linear(d, K_qk, bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)]
    )
    att_mod.rosa_wlm_v_list = nn.ModuleList(
        [nn.Linear(d, K_v, bias=False).to(dtype=base_dtype, device=base_device) for _ in range(M)]
    )
    for lst in (att_mod.rosa_wlm_q_list, att_mod.rosa_wlm_k_list, att_mod.rosa_wlm_v_list):
        for w in lst:
            nn.init.xavier_uniform_(w.weight)

    att_mod.rosa_v_emb_list = nn.ModuleList(
        [nn.Embedding(K_v + 1, d, padding_idx=0).to(dtype=base_dtype, device=base_device) for _ in range(M)]
    )
    for emb in att_mod.rosa_v_emb_list:
        nn.init.normal_(emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            emb.weight.data[0].zero_()

    att_mod.rosa_num_routes = M
    att_mod.rosa_qk_vocab_size = K_qk
    att_mod.rosa_v_vocab_size = K_v
    att_mod.rosa_route_slices = route_slices
    att_mod.rosa_inject_mode = ROSA_RWKV7_INJECT_MODE


@torch.no_grad()
def _zero_pad_row_every_step(model: nn.Module):
    for block in getattr(model, "blocks", []):
        att = getattr(block, "att", None)
        if att is None:
            continue
        if hasattr(att, "rosa_v_emb_list"):
            for emb in att.rosa_v_emb_list:
                emb.weight.data[0].zero_()


def _compute_rosa_v_for_block(att_mod: nn.Module, u: torch.Tensor):
    device = u.device
    dtype = u.dtype
    B, T, C = u.shape
    M = att_mod.rosa_num_routes
    K_qk = att_mod.rosa_qk_vocab_size
    K_v = att_mod.rosa_v_vocab_size
    slices = att_mod.rosa_route_slices

    logits_q_list, logits_k_list, logits_v_list = [], [], []
    for m, (s, e) in enumerate(slices):
        u_slice = u[:, :, s:e]
        logits_q_list.append(att_mod.rosa_wlm_q_list[m](u_slice))
        logits_k_list.append(att_mod.rosa_wlm_k_list[m](u_slice))
        logits_v_list.append(att_mod.rosa_wlm_v_list[m](u_slice))
    logits_q_all = torch.stack(logits_q_list, dim=2)
    logits_k_all = torch.stack(logits_k_list, dim=2)
    logits_v_all = torch.stack(logits_v_list, dim=2)

    q_btm = torch.argmax(logits_q_all, dim=-1).to(torch.int32)
    k_btm = torch.argmax(logits_k_all, dim=-1).to(torch.int32)
    v_btm = torch.argmax(logits_v_all, dim=-1).to(torch.int32)

    nvtx.range_push("RWKV7_ROSA.K_SAM_async")
    k_host = _PINNED_POOL.get("k_host_rwkv7", (B, T, M), dtype=torch.int32)
    k_host.copy_(k_btm, non_blocking=True)
    ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())

    def _build_k_sam():
        try:
            ev.synchronize()
        except Exception:
            pass
        import numpy as _np
        z_np = _np.asarray(k_host, order="C")
        return _k_sam_batch_btm_with_ws(z_np, int(K_qk))

    pool = _get_rosa_thread_pool()
    if pool is None:
        k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = _build_k_sam()
    else:
        fut_ws = pool.submit(_build_k_sam)
        k_dfa_np, k_e_np, k_runstart_np, k_clen_np, k_runsym_np = fut_ws.result()
    nvtx.range_pop()

    fut_q = _q_runs_pipeline_cpu(q_btm, k_runsym_np, k_runstart_np, k_clen_np, int(K_qk))
    tau_time_np, q_run_id_np, r_cf_run_np = fut_q.result()

    k_run_start_bmt = torch.from_numpy(k_runstart_np).to(device=device, dtype=torch.int32)
    tau_time_bmt = torch.from_numpy(tau_time_np).to(device=device, dtype=torch.int32)
    q_run_id_bmt = torch.from_numpy(q_run_id_np).to(device=device, dtype=torch.int32)
    r_cf_run_bmrk = torch.from_numpy(r_cf_run_np).to(device=device, dtype=torch.int32)

    valid_tau = (tau_time_bmt >= 0)
    v_idx_all = torch.argmax(logits_v_all, dim=-1).to(torch.long)
    v_idx_bmt = v_idx_all.permute(0, 2, 1).contiguous()
    t_tauL = torch.clamp(tau_time_bmt.to(torch.long), 0, T - 1)
    v_idx_at_tau = torch.gather(v_idx_bmt, dim=2, index=t_tauL)

    v_parts, E_compact = [], []
    for m, (s, e) in enumerate(slices):
        idx_m_plus = torch.where(valid_tau[:, m, :],
                                 v_idx_at_tau[:, m, :] + 1,
                                 torch.zeros_like(v_idx_at_tau[:, m, :]))
        v_m = att_mod.rosa_v_emb_list[m](idx_m_plus.to(torch.long))
        v_parts.append(v_m)
        E_compact.append(att_mod.rosa_v_emb_list[m].weight.detach())
    v = torch.cat(v_parts, dim=-1).to(dtype=dtype)
    E_v_compact = torch.stack(E_compact, dim=0)

    if LCG_ENABLED:
        pos_mask_cpu = None
        if LCG_POS_SUBSAMPLE < 1.0:
            mask = (torch.rand((B, T), device=device) < LCG_POS_SUBSAMPLE)
            pos_mask_cpu = mask.detach().cpu().tolist()
        v = MultiRouteQKVLcgFunction.apply(
            v, logits_q_all, logits_k_all, logits_v_all, E_v_compact,
            k_run_start_bmt, tau_time_bmt, q_run_id_bmt, r_cf_run_bmrk,
            pos_mask_cpu, att_mod.rosa_route_slices
        )

    aux = {
        "logits_q_all": logits_q_all,
        "logits_k_all": logits_k_all,
        "logits_v_all": logits_v_all,
        "k_run_start_bmt": k_run_start_bmt,
        "tau_time_bmt": tau_time_bmt,
        "q_run_id_bmt": q_run_id_bmt,
        "r_cf_run_bmrk": r_cf_run_bmrk,
    }
    return v, aux


def _build_pre_injected_input(u_norm: torch.Tensor, v_norm: torch.Tensor) -> torch.Tensor:
    diff = v_norm - u_norm
    x_inj = - torch.cumsum(diff, dim=1)
    return x_inj


def patch_rwkv7_with_rosa(model: nn.Module):
    assert hasattr(model, "blocks"), "Model does not contain blocks attribute."
    hidden_size = int(model.args.n_embd)

    for li, block in enumerate(model.blocks):
        if li < ROSA_RWKV7_APPLY_LAYERS_FROM:
            continue

        att = block.att
        _init_rosa_heads_for_att(att, hidden_size)

        _ln0 = getattr(block, "ln0", None)
        _ln1 = block.ln1
        _ln2 = block.ln2
        _att = block.att
        _ffn = block.ffn
        _layer_id = block.layer_id

        def _forward_with_rosa(_self, x, v_first):
            if _layer_id == 0 and _ln0 is not None:
                x = _ln0(x)

            u = _ln1(x)
            v_rosa, _aux = _compute_rosa_v_for_block(_att, u)

            if ROSA_RWKV7_INJECT_MODE == "post_tmix":
                x_attn, v_first = _att(u, v_first)
                x_attn = x_attn + v_rosa
            else:
                v_norm = _ln1(v_rosa)
                x_inj = _build_pre_injected_input(u_norm=u, v_norm=v_norm)
                x_attn, v_first = _att(x_inj, v_first)

            x = x + x_attn
            x = x + _ffn(_ln2(x))
            return x, v_first

        block.forward = _forward_with_rosa.__get__(block, block.__class__)

    meta = {
        "inject_mode": ROSA_RWKV7_INJECT_MODE,
        "apply_layers_from": ROSA_RWKV7_APPLY_LAYERS_FROM,
        "num_routes_per_layer": ROSA_RWKV7_NUM_ROUTES,
        "qk_vocab_size": ROSA_RWKV7_QK_VOCAB,
        "v_vocab_size": ROSA_RWKV7_V_VOCAB,
    }
    with open(os.path.join(ROSA_RWKV7_OUTPUT_DIR, "rosa_rwkv7_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_rwkv7_rosa_only(model: nn.Module, out_dir: str = None):
    out_dir = out_dir or ROSA_RWKV7_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    state = {}

    for li, block in enumerate(getattr(model, "blocks", [])):
        att = getattr(block, "att", None)
        if att is None:
            continue

        if hasattr(att, "rosa_wlm_q_list"):
            for m, head in enumerate(att.rosa_wlm_q_list):
                state[f"blocks.{li}.att.rosa_wlm_q_list.{m}.weight"] = head.weight.detach().cpu()
        if hasattr(att, "rosa_wlm_k_list"):
            for m, head in enumerate(att.rosa_wlm_k_list):
                state[f"blocks.{li}.att.rosa_wlm_k_list.{m}.weight"] = head.weight.detach().cpu()
        if hasattr(att, "rosa_wlm_v_list"):
            for m, head in enumerate(att.rosa_wlm_v_list):
                state[f"blocks.{li}.att.rosa_wlm_v_list.{m}.weight"] = head.weight.detach().cpu()

        if hasattr(att, "rosa_v_emb_list"):
            for m, emb in enumerate(att.rosa_v_emb_list):
                state[f"blocks.{li}.att.rosa_v_emb_list.{m}.weight"] = emb.weight.detach().cpu()

    path = os.path.join(out_dir, ROSA_RWKV7_STATE_NAME)
    torch.save(state, path)
    print(f"[save] saved ROSA-only params for RWKV7 to: {path}")


def build_rwkv7_optimizer_params(model: nn.Module):
    wlm_q, wlm_k, wlm_v, v_emb, backbone = [], [], [], [], []

    for n, p in model.named_parameters():
        if "rosa_wlm_q_list" in n:
            wlm_q.append(p)
        elif "rosa_wlm_k_list" in n:
            wlm_k.append(p)
        elif "rosa_wlm_v_list" in n:
            wlm_v.append(p)
        elif "rosa_v_emb_list" in n:
            v_emb.append(p)
        else:
            backbone.append(p)

    param_groups = []
    if wlm_q: param_groups.append({"params": wlm_q, "lr": RWKV7_ROSA_LR, "weight_decay": RWKV7_ROSA_WEIGHT_DECAY})
    if wlm_k: param_groups.append({"params": wlm_k, "lr": RWKV7_ROSA_LR, "weight_decay": RWKV7_ROSA_WEIGHT_DECAY})
    if wlm_v: param_groups.append({"params": wlm_v, "lr": RWKV7_ROSA_LR, "weight_decay": RWKV7_ROSA_WEIGHT_DECAY})
    if v_emb: param_groups.append({"params": v_emb, "lr": RWKV7_ROSA_LR, "weight_decay": 0.0})

    if RWKV7_BACKBONE_LR and RWKV7_BACKBONE_LR > 0.0:
        param_groups.append({"params": backbone, "lr": RWKV7_BACKBONE_LR, "weight_decay": RWKV7_ROSA_WEIGHT_DECAY})
    else:
        for p in backbone:
            p.requires_grad_(False)

    return param_groups


# model = RWKV(args) 

# patch_rwkv7_with_rosa(model)

# This code is just a conceptual example, you'll need to make some modifications if you want to run it.
