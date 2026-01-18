from typing import Dict
import math
import torch


def nowmp_reference_attention(
    attn_weight: torch.Tensor,  # [B, H, Q, K]
    value: torch.Tensor,        # [B, H, K, D]
    percentile: float,
    base_cutoff: int = 128,
    base_step: int = 4,
    lr_a: float = 0.15,
    lr_b: float = 0.015,
    anl_a: float = 0.98,
    anl_b: float = 0.995,
    a_min: float = 1e-4,
    a_max: float = 30.0,
    b_min: float = -3.0,
    b_max: float = -1e-4,
    return_traces: bool = True,
    logits: torch.Tensor = None,  # [B, H, Q, K] pre-softmax
    emulate_kernel: bool = False,
) -> Dict[str, torch.Tensor]:
    use_kernel_softmax = emulate_kernel and logits is not None
    logits_f = None
    logits_h = None
    if logits is not None:
        logits_f = logits.to(torch.float32)
        if use_kernel_softmax:
            logits_h = logits_f.to(torch.float16)
        else:
            attn_weight = torch.softmax(logits_f, dim=-1)
    elif emulate_kernel:
        attn_weight = attn_weight.to(torch.float16).to(torch.float32)

    B, H, Q, K = attn_weight.shape
    D = value.size(-1)
    device = attn_weight.device
    value_f = value.to(torch.float32 if use_kernel_softmax else attn_weight.dtype)

    alpha = torch.zeros((B, H), dtype=torch.float32, device=device)
    b = torch.full((B, H), -1.0, dtype=torch.float32, device=device)
    kept_cum = torch.zeros((B, H), dtype=torch.float32, device=device)
    total_cum = torch.zeros((B, H), dtype=torch.float32, device=device)
    r_target = torch.tensor(1.0 - percentile, dtype=torch.float32, device=device)
    lr_a_t = torch.tensor(lr_a, dtype=torch.float32, device=device)
    lr_b_t = torch.tensor(lr_b, dtype=torch.float32, device=device)

    log_a_min = math.log(a_min)
    log_a_max = math.log(a_max)

    out = torch.empty((B, H, Q, D), dtype=attn_weight.dtype, device=device)
    retain_sum = torch.zeros((), dtype=torch.float32, device=device)

    if return_traces:
        alpha_trace = torch.empty((B, H, Q), dtype=torch.float32, device=device)
        b_trace = torch.empty((B, H, Q), dtype=torch.float32, device=device)
        theta_trace = torch.empty((B, H, Q), dtype=torch.float32, device=device)
        retain_trace = torch.empty((B, H, Q), dtype=torch.float32, device=device)

    key_idx = torch.arange(K, device=device)
    q0 = 0
    while q0 < Q:
        t = q0 + 1
        if t < base_cutoff:
            step = base_step
            region_end = base_cutoff
        else:
            ratio = t // base_cutoff
            k = ratio.bit_length() - 1
            step = base_step << (k + 1)
            region_end = base_cutoff << (k + 1)

        q1 = min(q0 + step, region_end, Q)
        m_count = q1 - q0

        t_vec = torch.arange(q0 + 1, q1 + 1, device=device, dtype=torch.float32)
        log_t_vec = torch.log(t_vec)
        x = alpha.unsqueeze(-1) + b.unsqueeze(-1) * log_t_vec
        x = x.clamp(min=-20.0, max=0.0)
        theta = torch.exp(x)

        q_idx = torch.arange(q0, q1, device=device)
        causal_valid = key_idx[None, :] <= q_idx[:, None]

        if use_kernel_softmax:
            logit_chunk_f = logits_f[:, :, q0:q1, :]
            logit_chunk_h = logits_h[:, :, q0:q1, :]
            m = logit_chunk_f.max(dim=-1, keepdim=True).values
            l = torch.exp(logit_chunk_f - m).sum(dim=-1, keepdim=True)
            w = torch.exp(logit_chunk_h - m) / l
        else:
            w = attn_weight[:, :, q0:q1, :]
        keep = (w >= theta[..., None]) & causal_valid[None, None, :, :]
        keep[:, :, torch.arange(m_count, device=device), q_idx] = True

        keep_counts = keep.sum(dim=-1)
        kept_cum += keep_counts.to(torch.float32).sum(dim=-1)
        total_cum += t_vec.sum()
        retain_ratio_chunk = keep_counts.to(torch.float32) / t_vec.view(1, 1, m_count)
        retain_sum += retain_ratio_chunk.mean(dim=(0, 1)).sum()

        w_thresh = w * keep.to(w.dtype)
        if emulate_kernel:
            w_thresh = w_thresh.to(torch.float16).to(torch.float32)
        out[:, :, q0:q1, :] = w_thresh @ value_f

        if return_traces:
            alpha_trace[:, :, q0:q1] = alpha.unsqueeze(-1)
            b_trace[:, :, q0:q1] = b.unsqueeze(-1)
            theta_trace[:, :, q0:q1] = theta
            retain_trace[:, :, q0:q1] = retain_ratio_chunk

        r_cum = kept_cum / (total_cum + 1e-6)
        e = r_cum - r_target
        alpha = alpha + lr_a_t * e
        b = b + lr_b_t * e
        alpha = alpha.clamp(min=log_a_min, max=log_a_max)
        b = b.clamp(min=b_min, max=b_max)
        if q1 >= base_cutoff:
            lr_a_t = lr_a_t * anl_a
            lr_b_t = lr_b_t * anl_b

        q0 = q1

    retain_mean = retain_sum / Q
    result = {"output": out, "retain_mean": retain_mean}
    if return_traces:
        result["traces"] = {
            "alpha": alpha_trace,
            "b": b_trace,
            "theta": theta_trace,
            "retain": retain_trace,
        }
    return result
