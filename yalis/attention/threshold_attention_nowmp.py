from typing import Optional, Dict, Tuple
import math
import os
import re
import torch
from torch.utils.cpp_extension import load

_BASE_DIR = os.path.dirname(__file__)
_KERNEL_ENV = "YALIS_NOWMP_KERNEL"
_DEFAULT_KERNEL = os.path.join(_BASE_DIR, "kernels", "thresh_attn_nowmp_cuda.cu")
_NOWMP_EXT = None

BASE_CUTOFF = 128
BASE_STEP = 4
LR_A = 0.15
LR_B = 0.015
ANL_A = 0.98
ANL_B = 0.995
A_MIN = 1e-4
A_MAX = 30.0
B_MIN = -3.0
B_MAX = -1e-4


def _resolve_kernel_source() -> str:
    kernel_source = os.getenv(_KERNEL_ENV, _DEFAULT_KERNEL)
    if not os.path.isabs(kernel_source):
        kernel_source = os.path.join(_BASE_DIR, kernel_source)
    return os.path.normpath(kernel_source)


def _ext_name_from_kernel(kernel_source: str) -> str:
    stem = os.path.splitext(os.path.basename(kernel_source))[0]
    safe_stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem)
    return f"nowmp_attn_cuda_{safe_stem}"


def _load_nowmp_ext():
    global _NOWMP_EXT
    if _NOWMP_EXT is not None:
        return _NOWMP_EXT
    kernel_source = _resolve_kernel_source()
    _NOWMP_EXT = load(
        name=_ext_name_from_kernel(kernel_source),
        sources=[os.path.join(_BASE_DIR, "thresh_attn_nowmp_c.cpp"), kernel_source],
        verbose=False,
        extra_cuda_cflags=["-O3"],
    )
    return _NOWMP_EXT


@torch.library.custom_op("yalis::nowmp_attn", mutates_args=("alpha", "b", "kept_cum", "total_cum", "lr_a", "lr_b"))
def nowmp_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor,
    token_counter: torch.Tensor,
    alpha: torch.Tensor,
    b: torch.Tensor,
    kept_cum: torch.Tensor,
    total_cum: torch.Tensor,
    lr_a: torch.Tensor,
    lr_b: torch.Tensor,
    r_target: float,
    base_cutoff: int,
    base_step: int,
    anl_a: float,
    anl_b: float,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
    scale_factor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ext = _load_nowmp_ext()
    return ext.nowmp_attn_fwd(
        query,
        key,
        value,
        attn_bias,
        token_counter,
        alpha,
        b,
        kept_cum,
        total_cum,
        lr_a,
        lr_b,
        r_target,
        base_cutoff,
        base_step,
        anl_a,
        anl_b,
        a_min,
        a_max,
        b_min,
        b_max,
        scale_factor,
    )


@nowmp_attn.register_fake
def _(
    query,
    key,
    value,
    attn_bias,
    token_counter,
    alpha,
    b,
    kept_cum,
    total_cum,
    lr_a,
    lr_b,
    r_target,
    base_cutoff,
    base_step,
    anl_a,
    anl_b,
    a_min,
    a_max,
    b_min,
    b_max,
    scale_factor,
):
    B, H, _, _ = query.shape
    D = value.size(-1)
    out = torch.empty((B, H, 1, D), device=query.device, dtype=value.dtype)
    keep_counts = torch.empty((B, H), device=query.device, dtype=torch.float32)
    return out, keep_counts


def init_nowmp_state(
    batch: int,
    heads: int,
    device: torch.device,
    lr_a: float = LR_A,
    lr_b: float = LR_B,
) -> Dict[str, torch.Tensor]:
    return {
        "alpha": torch.zeros((batch, heads), dtype=torch.float32, device=device),
        "b": torch.full((batch, heads), -1.0, dtype=torch.float32, device=device),
        "kept_cum": torch.zeros((batch, heads), dtype=torch.float32, device=device),
        "total_cum": torch.zeros((batch, heads), dtype=torch.float32, device=device),
        "lr_a": torch.full((batch, heads), lr_a, dtype=torch.float32, device=device),
        "lr_b": torch.full((batch, heads), lr_b, dtype=torch.float32, device=device),
    }


def reset_nowmp_state(
    state: Dict[str, torch.Tensor],
    lr_a: float = LR_A,
    lr_b: float = LR_B,
) -> None:
    state["alpha"].zero_()
    state["b"].fill_(-1.0)
    state["kept_cum"].zero_()
    state["total_cum"].zero_()
    state["lr_a"].fill_(lr_a)
    state["lr_b"].fill_(lr_b)


def _build_attn_bias(
    query: torch.Tensor, key: torch.Tensor, attn_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    attn_bias = torch.zeros(B, H, L, S, dtype=torch.float32, device=query.device)
    if attn_mask is not None:
        if attn_mask.device != attn_bias.device:
            attn_mask = attn_mask.to(attn_bias.device)
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask.to(torch.float32) + attn_bias
    return attn_bias.contiguous()


def nowmp_attn_forward(
    query: torch.Tensor,        # [B, H, 1, D], fp16
    key: torch.Tensor,          # [B, H, T, D], fp16
    value: torch.Tensor,        # [B, H, T, D], fp16
    attn_bias: torch.Tensor,    # [B, H, 1, T], fp32
    token_counter: torch.Tensor,  # [B], int32
    state: Dict[str, torch.Tensor],
    percentile: float,
    scale_factor: float,
    base_cutoff: int = BASE_CUTOFF,
    base_step: int = BASE_STEP,
    anl_a: float = ANL_A,
    anl_b: float = ANL_B,
    a_min: float = A_MIN,
    a_max: float = A_MAX,
    b_min: float = B_MIN,
    b_max: float = B_MAX,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r_target = 1.0 - percentile
    return nowmp_attn(
        query,
        key,
        value,
        attn_bias,
        token_counter,
        state["alpha"],
        state["b"],
        state["kept_cum"],
        state["total_cum"],
        state["lr_a"],
        state["lr_b"],
        r_target,
        base_cutoff,
        base_step,
        anl_a,
        anl_b,
        a_min,
        a_max,
        b_min,
        b_max,
        scale_factor,
    )


def nowmp_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_counter: torch.Tensor,
    state: Dict[str, torch.Tensor],
    percentile: float,
    attn_mask: Optional[torch.Tensor] = None,
    enable_gqa: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if query.dtype != torch.float16 or key.dtype != torch.float16 or value.dtype != torch.float16:
        raise RuntimeError("nowmp attention expects fp16 query/key/value")

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if query.size(-2) != 1:
        raise RuntimeError("nowmp attention only supports decode with a single query")

    attn_bias = _build_attn_bias(query, key, attn_mask)
    scale_factor = 1.0 / math.sqrt(query.size(-1))

    return nowmp_attn_forward(
        query=query,
        key=key,
        value=value,
        attn_bias=attn_bias,
        token_counter=token_counter,
        state=state,
        percentile=percentile,
        scale_factor=scale_factor,
    )
