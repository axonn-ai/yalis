"""
Offline MoE kernel tuning script.

Usage:
    # Uses $YALIS_CACHE/configs as default save directory
    python benchmarks/benchmark_moe.py \
        --num-experts 8 64 128 \
        --hidden-size 4096 \
        --intermediate-size 14336 \
        --dtype bf16

    # Or specify custom save directory
    python benchmarks/benchmark_moe.py \
        --num-experts 8 \
        --save-dir /path/to/configs

This script benchmarks different Triton kernel configurations for the
fused MoE kernel and saves the optimal configuration for each batch size
to a JSON file. These configs are loaded at runtime by the MoE kernel
from $YALIS_CACHE/configs/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import triton

# Add parent directory to path so we can import from yalis
sys.path.insert(0, str(Path(__file__).parent.parent))

from yalis.external.fused_moe import (  # noqa: E402
    fused_moe_kernel,
    moe_align_block_size,
    fused_topk,
    get_config_file_name,
    get_moe_config_dir,
    get_config_dtype_str,
)


# Configurations to benchmark
BLOCK_CONFIGS = [
    {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
    },
    {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 4,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    },
    {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    },
    {
        "BLOCK_SIZE_M": 128,
        "BLOCK_SIZE_N": 256,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
    },
]

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def benchmark_config(
    M: int,
    E: int,
    N: int,
    K: int,
    top_k: int,
    dtype: torch.dtype,
    config: dict,
    warmup: int = 10,
    rep: int = 100,
) -> float:
    """Benchmark a single configuration and return average time in ms."""
    import triton.language as tl

    # Setup tensors
    hidden_states = torch.randn(M, K, dtype=dtype, device="cuda")
    w1 = torch.randn(E, 2 * N, K, dtype=dtype, device="cuda")
    gating = torch.randn(M, E, dtype=dtype, device="cuda")

    topk_weights, topk_ids = fused_topk(
        hidden_states, gating, top_k, renormalize=True
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)
    )

    # Output tensor
    C = torch.empty((M, top_k, 2 * N), dtype=dtype, device="cuda")

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    def grid(META):
        return (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(w1.shape[1], META["BLOCK_SIZE_N"]),
        )

    # Warmup
    for _ in range(warmup):
        fused_moe_kernel[grid](
            hidden_states,
            w1,
            C,
            None,  # A_scale
            None,  # B_scale
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            w1.shape[1],
            w1.shape[2],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(2),
            w1.stride(1),
            C.stride(1),
            C.stride(2),
            0,  # B_scale stride
            0,  # B_scale stride
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            **config,
        )

    torch.cuda.synchronize()

    # Benchmark
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

    for i in range(rep):
        start_events[i].record()
        fused_moe_kernel[grid](
            hidden_states,
            w1,
            C,
            None,
            None,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            w1.shape[1],
            w1.shape[2],
            sorted_token_ids.shape[0],
            topk_ids.numel(),
            hidden_states.stride(0),
            hidden_states.stride(1),
            w1.stride(0),
            w1.stride(2),
            w1.stride(1),
            C.stride(1),
            C.stride(2),
            0,
            0,
            MUL_ROUTED_WEIGHT=False,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            **config,
        )
        end_events[i].record()

    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(rep)]
    return sum(times) / len(times)


def tune_for_shape(
    E: int,
    N: int,
    K: int,
    top_k: int,
    dtype: torch.dtype,
    save_dir: str,
    batch_sizes: list = None,
) -> dict:
    """Tune for all batch sizes for a given shape."""
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES

    results = {}
    dtype_str = get_config_dtype_str(dtype)

    print(f"\nTuning for E={E}, N={N}, K={K}, dtype={dtype_str}")
    print("=" * 60)

    for M in batch_sizes:
        best_config = None
        best_time = float("inf")

        print(f"\n  M={M}:")
        for config in BLOCK_CONFIGS:
            try:
                time_ms = benchmark_config(M, E, N, K, top_k, dtype, config)
                config_str = (
                    f"M={config['BLOCK_SIZE_M']}, "
                    f"N={config['BLOCK_SIZE_N']}, "
                    f"K={config['BLOCK_SIZE_K']}, "
                    f"G={config['GROUP_SIZE_M']}"
                )
                print(f"    {config_str}: {time_ms:.4f} ms")
                if time_ms < best_time:
                    best_time = time_ms
                    best_config = config.copy()
            except Exception as e:
                print(f"    Config failed: {e}")
                continue

        if best_config:
            results[M] = best_config
            print(f"  -> Best: {best_config} ({best_time:.4f} ms)")

    # Save to JSON
    os.makedirs(save_dir, exist_ok=True)
    filename = get_config_file_name(E, N, dtype_str)
    filepath = Path(save_dir) / filename

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: {filepath}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Offline MoE kernel tuning script"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[8],
        help="Number of experts to tune for",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Hidden size (K dimension)",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=14336,
        help="Intermediate size (N dimension)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Top-k experts per token",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16"],
        default="bf16",
        help="Data type for benchmarking",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help=(
            "Directory to save config files "
            "(default: $YALIS_CACHE/configs or ~/.cache/yalis/configs)"
        ),
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to tune for (default: standard set)",
    )
    args = parser.parse_args()

    # Resolve save directory - uses YALIS_CACHE or falls back to ~/.cache/yalis
    save_dir = (
        args.save_dir if args.save_dir else get_moe_config_dir(create=True)
    )

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print("MoE Kernel Tuning")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Save directory: {save_dir}")
    print(f"Hidden size (K): {args.hidden_size}")
    print(f"Intermediate size (N): {args.intermediate_size}")
    print(f"Top-k: {args.top_k}")
    print(f"Dtype: {args.dtype}")
    print(f"Experts to tune: {args.num_experts}")

    for E in args.num_experts:
        tune_for_shape(
            E=E,
            N=args.intermediate_size,
            K=args.hidden_size,
            top_k=args.top_k,
            dtype=dtype,
            save_dir=save_dir,
            batch_sizes=args.batch_sizes,
        )

    print("\nTuning complete!")


if __name__ == "__main__":
    main()
