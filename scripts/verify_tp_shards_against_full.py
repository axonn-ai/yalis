#!/usr/bin/env python3
"""
Verify TP shards by reconstructing full tensors from per-rank safetensors
and comparing shape (and optionally values) against original unsharded checkpoint.

Usage:
  python scripts/verify_tp_shards_against_full.py \
    --tp-root yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints_tp \
    --full-root yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints \
    --world-size 2 \
    [--check-values] \
    [--sample 50]

Notes:
- If the full unsharded checkpoint directory is not available, run without --full-root
  to only validate per-rank shapes reconstruct into sensible full shapes.
- Use --check-values to compare numeric equality (may be slow / memory heavy).
"""
import argparse
from pathlib import Path
from safetensors import safe_open, load_file
import torch
from collections import defaultdict

def get_shard_axis(key: str, ndim: int):
    if any(x in key for x in ["embed", "norm", "router", "lm_head"]):
        return None
    if "mlp" in key and ndim == 3:
        if "gate_up_proj" in key:
            return 2
        elif "proj" in key:
            return 1
        else:
            return None
    if ndim == 2:
        return 0
    return None

def collect_rank_keys(rank_dir: Path):
    keys = set()
    for p in sorted(rank_dir.glob("*.safetensors")):
        with safe_open(p, framework="pt", device="cpu") as f:
            keys.update(f.keys())
    return keys

def load_tensor_from_rank(rank_dir: Path, key: str):
    # searches each file in rank_dir for key, returns tensor if found
    for p in sorted(rank_dir.glob("*.safetensors")):
        with safe_open(p, framework="pt", device="cpu") as f:
            if key in f.keys():
                return f.get_tensor(key)
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp-root", type=Path, required=True, help="TP root with rank_{i}/yalis_checkpoints")
    p.add_argument("--full-root", type=Path, default=None, help="Original unsharded checkpoint directory (optional)")
    p.add_argument("--world-size", type=int, default=2)
    p.add_argument("--check-values", action="store_true", help="Compare tensor values with original (slow)")
    p.add_argument("--sample", type=int, default=0, help="Optional sample limit (0 = all keys)")
    args = p.parse_args()

    rank_dirs = [args.tp_root / f"rank_{r}" / "yalis_checkpoints" for r in range(args.world_size)]
    for rd in rank_dirs:
        if not rd.exists():
            raise SystemExit(f"Missing rank dir: {rd}")

    # union of all keys present in shards
    all_keys = set()
    for rd in rank_dirs:
        all_keys.update(collect_rank_keys(rd))
    all_keys = sorted(all_keys)
    if args.sample and args.sample > 0:
        all_keys = all_keys[:args.sample]

    failures = []
    for key in all_keys:
        # gather per-rank tensors and shapes
        tensors = []
        shapes = []
        for rd in rank_dirs:
            t = load_tensor_from_rank(rd, key)
            if t is None:
                tensors.append(None)
                shapes.append(None)
            else:
                tensors.append(t)
                shapes.append(tuple(t.shape))
        # determine ndim from any non-None
        ndim = None
        for s in shapes:
            if s is not None:
                ndim = len(s)
                break
        shard_axis = get_shard_axis(key, ndim) if ndim is not None else None

        # shape-only checks: missing?
        if any(s is None for s in shapes):
            failures.append((key, "missing_in_some_ranks", shapes))
            continue

        if shard_axis is None:
            # expected replicated -> check all shapes equal
            if not all(s == shapes[0] for s in shapes):
                failures.append((key, "replicated_shapes_mismatch", shapes))
            else:
                # optional value check: if full-root present check it matches
                if args.full_root:
                    # try load from full-root
                    full_t = load_tensor_from_rank(args.full_root, key)
                    if full_t is not None and args.check_values:
                        # compare values to one rank (they should be identical)
                        if not torch.allclose(full_t, tensors[0], rtol=1e-4, atol=1e-6):
                            failures.append((key, "value_mismatch_replicated", None))
            continue

        # expected to be sharded along shard_axis
        # verify that other dims are equal
        other_dims_equal = True
        for d in range(len(shapes[0])):
            if d == shard_axis:
                continue
            for s in shapes[1:]:
                if s[d] != shapes[0][d]:
                    other_dims_equal = False
                    break
            if not other_dims_equal:
                break
        if not other_dims_equal:
            failures.append((key, "other_dims_mismatch", shapes))
            continue

        # compute reconstructed full shape along shard axis
        total = sum(s[shard_axis] for s in shapes)
        reconstructed_shape = list(shapes[0])
        reconstructed_shape[shard_axis] = total
        reconstructed_shape = tuple(reconstructed_shape)

        # if full-root available, compare reconstructed shape to full
        if args.full_root:
            full_t = load_tensor_from_rank(args.full_root, key)
            if full_t is None:
                failures.append((key, "full_missing", reconstructed_shape))
                continue
            full_shape = tuple(full_t.shape)
            if full_shape != reconstructed_shape:
                failures.append((key, "shape_mismatch_with_full", {"reconstructed": reconstructed_shape, "full": full_shape}))
                continue
            # optional value check
            if args.check_values:
                # concat along axis and compare
                concatenated = torch.cat([t for t in tensors], dim=shard_axis)
                if concatenated.shape != full_t.shape:
                    failures.append((key, "concat_shape_mismatch", {"concat": tuple(concatenated.shape), "full": tuple(full_t.shape)}))
                else:
                    if not torch.allclose(concatenated, full_t, rtol=1e-4, atol=1e-6):
                        failures.append((key, "value_mismatch_after_concat", None))
        else:
            # no full-root; we only report reconstructed shape (for user info)
            # If you want, we can print first few reconstructed shapes
            pass

    # Report
    if not failures:
        print("Verification passed: no issues found for inspected keys.")
        print(f"Inspected keys: {len(all_keys)}")
    else:
        print(f"Verification found {len(failures)} issues:")
        for f in failures[:200]:
            print(f"- {f[0]} : {f[1]} -> {f[2]}")
        print("...")

if __name__ == '__main__':
    main()