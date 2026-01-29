import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from yalis.attention.double_sparse import get_label_tensor, fwd_sparse_no_mask


def _parse_args():
    parser = argparse.ArgumentParser(description="DoubleSparse decode correctness check")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--heavy-channel-num", type=int, default=None)
    parser.add_argument("--heavy-const", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _torch_sparse_reference(q, k, v, heavy_list):
    B, H, D = q.shape
    N_CTX = k.shape[0] // B
    k = k.view(B, N_CTX, H, D)
    v = v.view(B, N_CTX, H, D)
    out = torch.empty((B, H, D), device=q.device, dtype=q.dtype)
    scale = 1.0 / math.sqrt(D)
    for b in range(B):
        for h in range(H):
            idx = heavy_list[b, h]
            k_sel = k[b, idx, h]
            v_sel = v[b, idx, h]
            scores = (q[b, h] * k_sel).sum(dim=-1) * scale
            weights = torch.softmax(scores.to(torch.float32), dim=0).to(q.dtype)
            out[b, h] = (weights[:, None] * v_sel).sum(dim=0)
    return out


def main():
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    if args.head_dim not in {16, 32, 64, 128}:
        raise ValueError("head_dim must be one of {16, 32, 64, 128}.")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16

    B, H, N_CTX, D = args.batch, args.heads, args.seq_len, args.head_dim
    heavy_channel_num = args.heavy_channel_num or max(1, D // 16)
    heavy_const = args.heavy_const or max(1, N_CTX // 16)
    if heavy_const > N_CTX:
        raise ValueError("heavy_const must be <= seq_len.")

    q = torch.randn((B, H, D), device=device, dtype=dtype)
    k_full = torch.randn((B, N_CTX, H, D), device=device, dtype=dtype)
    v_full = torch.randn((B, N_CTX, H, D), device=device, dtype=dtype)
    k = k_full.contiguous().view(B * N_CTX, H, D)
    v = v_full.contiguous().view(B * N_CTX, H, D)

    channel = torch.zeros((H, heavy_channel_num), dtype=torch.int64, device=device)
    for h in range(H):
        channel[h] = torch.randperm(D, device=device)[:heavy_channel_num]

    q_label = torch.empty((B, H, heavy_channel_num), device=device, dtype=dtype)
    k_label = torch.empty((B * N_CTX, H, heavy_channel_num), device=device, dtype=dtype)
    get_label_tensor(q, channel, q_label, heavy_channel_num)
    get_label_tensor(k, channel, k_label, heavy_channel_num)

    label_scores = torch.matmul(
        q_label.view(B, 1, H, heavy_channel_num).transpose(1, 2),
        k_label.view(B, N_CTX, H, heavy_channel_num).transpose(1, 2).transpose(2, 3),
    ).view(B, H, 1, N_CTX)
    _, label_index = torch.topk(label_scores, heavy_const, dim=-1)
    heavy_list = label_index.view(B, H, heavy_const)

    out = torch.empty((B, H, D), device=device, dtype=dtype)
    fwd_sparse_no_mask(q, k, v, out, heavy_list)

    ref = _torch_sparse_reference(q, k, v, heavy_list)
    diff = (out.float() - ref.float()).abs()
    print(f"max_abs_diff={diff.max().item():.6f}")
    print(f"mean_abs_diff={diff.mean().item():.6f}")
    print(f"out_nan={torch.isnan(out).any().item()}")


if __name__ == "__main__":
    main()
