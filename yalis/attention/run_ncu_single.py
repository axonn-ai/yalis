import argparse
import math
import os
import re

import torch
from torch.utils.cpp_extension import load

DEVICE = "cuda"
_BASE_DIR = os.path.dirname(__file__)


def get_threshold(query, key, value, percentile=0.5, attn_mask=None, enable_gqa=False):
    L, S = query.size(-2), key.size(-2)
    assert L == 1, "Only decodes for one query at a time"
    scale_factor = 1 / math.sqrt(query.size(-1))
    B, H = query.shape[0], query.shape[1]

    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1).to(torch.float32)
    attn_weight += attn_bias

    attn_weight = torch.where(attn_weight == float("-inf"), torch.nan, attn_weight)
    threshold = torch.nanquantile(attn_weight, percentile, dim=-1)
    return threshold


def _resolve_kernel(kernel_arg):
    if not kernel_arg:
        return os.path.join(_BASE_DIR, "thresh_attn_cuda.cu")
    kernel_source = kernel_arg
    if not os.path.isabs(kernel_source):
        kernel_source = os.path.join(_BASE_DIR, kernel_source)
    return os.path.normpath(kernel_source)


def _ext_name_from_kernel(kernel_source):
    stem = os.path.splitext(os.path.basename(kernel_source))[0]
    safe_stem = re.sub(r"[^0-9A-Za-z_]+", "_", stem)
    return f"decode_attn_cuda_{safe_stem}"


def main():
    parser = argparse.ArgumentParser(description="Single-config runner for NCU profiling.")
    parser.add_argument("--kernel", type=str, default=None, help="Path to .cu kernel source.")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length T.")
    parser.add_argument("--percentile", type=float, default=0.875, help="Threshold percentile.")
    parser.add_argument("--B", type=int, default=32, help="Batch size.")
    parser.add_argument("--H", type=int, default=16, help="Number of heads.")
    parser.add_argument("--D", type=int, default=128, help="Head dimension.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=1, help="Profile iterations.")
    args = parser.parse_args()

    kernel_source = _resolve_kernel(args.kernel)
    use_gmem_kernel = (
        "tiled_fused_v1" in os.path.basename(kernel_source)
        or "gmem" in os.path.basename(kernel_source)
    )

    decode_attn_cuda = load(
        name=_ext_name_from_kernel(kernel_source),
        sources=[os.path.join(_BASE_DIR, "thresh_attn_c.cpp"), kernel_source],
        verbose=True,
        extra_cuda_cflags=["-arch=sm_90", "-O3", "-lineinfo"],
    )

    B, H, D, T = args.B, args.H, args.D, args.seq_len
    query = torch.randn((B, H, 1, D), device=DEVICE, dtype=torch.float16)
    key = torch.randn((B, H, T, D), device=DEVICE, dtype=torch.float16)
    value = torch.randn((B, H, T, D), device=DEVICE, dtype=torch.float16)
    scale_factor = 1.0 / math.sqrt(D)

    threshold = get_threshold(
        query,
        key,
        value,
        percentile=args.percentile,
        attn_mask=None,
        enable_gqa=False,
    ).reshape((B, H))
    threshold = threshold.to(dtype=torch.float16)
    threshold_f32 = threshold.to(dtype=torch.float32).contiguous()

    attn_bias = torch.zeros((B, H, 1, T), device=DEVICE, dtype=torch.float16)
    attn_bias = attn_bias.to(dtype=torch.float32).contiguous()

    def run_once():
        if use_gmem_kernel:
            return decode_attn_cuda.decode_attn_fwd_gmem(
                query, key, value, attn_bias, threshold_f32, scale_factor
            )
        return decode_attn_cuda.decode_attn_fwd(
            query, key, value, attn_bias, threshold_f32, scale_factor
        )

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()

    for _ in range(args.iters):
        run_once()
    torch.cuda.synchronize()

    print(
        "PID=%d kernel=%s gmem=%s B=%d H=%d D=%d seq_len=%d percentile=%.3f"
        % (
            os.getpid(),
            os.path.basename(kernel_source),
            use_gmem_kernel,
            B,
            H,
            D,
            T,
            args.percentile,
        )
    )


if __name__ == "__main__":
    main()

# Examples:
# python run_ncu_single.py --kernel thresh_attn_cuda_fused.cu --seq_len 16384 --percentile 0.875 --iters 1
# python run_ncu_single.py --kernel tiled_bitmask_v2.cu --seq_len 32768 --percentile 0.875 --iters 1
