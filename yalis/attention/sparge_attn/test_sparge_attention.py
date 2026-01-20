import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from yalis.attention.sparge import sparge_attention_forward


def _parse_args():
    parser = argparse.ArgumentParser(description="SpargeAttn non-quant correctness check")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    parser.add_argument("--simthreshd1", type=float, default=0.3)
    parser.add_argument("--cdfthreshd", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    B, H, T, D = args.batch, args.heads, args.seq_len, args.head_dim
    q = torch.randn((B, H, T, D), device=device, dtype=dtype)
    k = torch.randn((B, H, T, D), device=device, dtype=dtype)
    v = torch.randn((B, H, T, D), device=device, dtype=dtype)

    out = sparge_attention_forward(
        q=q,
        k=k,
        v=v,
        is_causal=True,
        simthreshd1=args.simthreshd1,
        cdfthreshd=args.cdfthreshd,
    )

    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True
    )

    diff = (out.float() - ref.float()).abs()
    print(f"max_abs_diff={diff.max().item():.6f}")
    print(f"mean_abs_diff={diff.mean().item():.6f}")
    print(f"out_nan={torch.isnan(out).any().item()}")


if __name__ == "__main__":
    main()
