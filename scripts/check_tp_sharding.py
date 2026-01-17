#!/usr/bin/env python3
"""
Check which keys in a TP checkpoint were NOT sharded as expected.

Usage:
  python scripts/check_tp_sharding.py \
    --tp-root yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints_tp \
    --world-size 2

Default world-size is 2 and it expects subdirs `rank_0` and `rank_1`
each containing a nested `yalis_checkpoints` directory with .safetensors files.
"""
import argparse
from pathlib import Path
from safetensors import safe_open
from collections import defaultdict

def get_shard_axis(key: str, ndim: int):
    # mirror the sharding logic from create_tp_checkpoint.py
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

def collect_keys(rank_dir: Path):
    keys = set()
    for p in sorted(rank_dir.glob("*.safetensors")):
        try:
            with safe_open(p, framework="pt", device="cpu") as f:
                keys.update(f.keys())
        except Exception as e:
            print(f"Warning: failed to open {p}: {e}")
    return keys

def find_shape(rank_dir: Path, key: str):
    for p in sorted(rank_dir.glob("*.safetensors")):
        try:
            with safe_open(p, framework="pt", device="cpu") as f:
                if key in f.keys():
                    t = f.get_tensor(key)
                    return tuple(t.shape)
        except Exception as e:
            print(f"Warning reading {p}: {e}")
    return None

def pretty_shape(s):
    return str(s) if s is not None else "MISSING"

def analyze(tp_root: Path, world_size: int):
    rank_dirs = [tp_root / f"rank_{r}" / "yalis_checkpoints" for r in range(world_size)]
    for rd in rank_dirs:
        if not rd.exists():
            raise SystemExit(f"Rank dir missing: {rd}")

    # union of keys across ranks
    all_keys = set()
    for rd in rank_dirs:
        all_keys.update(collect_keys(rd))
    all_keys = sorted(all_keys)

    not_sharded = []
    missing_in_some = []
    ok_sharded = []
    ambiguous = []

    for key in all_keys:
        shapes = [find_shape(rd, key) for rd in rank_dirs]
        # basic info
        ndim = None
        for s in shapes:
            if s is not None:
                ndim = len(s)
                break
        shard_axis = get_shard_axis(key, ndim) if ndim is not None else None

        # detect missing
        if any(s is None for s in shapes):
            missing_in_some.append((key, shapes, shard_axis))
            continue

        # all present
        if shard_axis is None:
            # expected replicated
            all_equal = all(s == shapes[0] for s in shapes)
            if all_equal:
                ok_sharded.append((key, shapes, shard_axis))
            else:
                ambiguous.append((key, shapes, shard_axis))
            continue

        # expected to be sharded along shard_axis
        # check if shapes appear to be shards that reconstruct a larger tensor
        # simple heuristics for world_size==2 (generalizes for any world_size)
        try:
            # check other dims equal
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
                ambiguous.append((key, shapes, shard_axis))
                continue

            # compute sum along shard axis
            total = sum(s[shard_axis] for s in shapes)
            # if every rank has same size (i.e., shapes[0][shard]==shapes[1][shard]==total),
            # then they are replicated/full (bad).
            all_same_as_total = all(s[shard_axis] == total for s in shapes)
            if all_same_as_total:
                # not sharded: both ranks have full-size tensor
                expected_per_rank = list(shapes[0])
                expected_per_rank[shard_axis] = total // world_size if total % world_size == 0 else None
                not_sharded.append((key, shapes, shard_axis, tuple(expected_per_rank)))
                continue

            # otherwise check that each shard size equals total // world_size (or sums to total)
            if total % world_size != 0:
                # weird shapes that don't divide evenly
                ambiguous.append((key, shapes, shard_axis))
                continue

            per_rank_expected = total // world_size
            sizes_ok = all(s[shard_axis] <= per_rank_expected for s in shapes) and \
                       sum(s[shard_axis] for s in shapes) == total
            if sizes_ok:
                ok_sharded.append((key, shapes, shard_axis))
            else:
                ambiguous.append((key, shapes, shard_axis))

        except Exception as e:
            ambiguous.append((key, shapes, shard_axis))

    # Print summary
    print("\nSharding check summary:")
    print(f"  total keys scanned: {len(all_keys)}")
    print(f"  OK sharded (or correctly replicated): {len(ok_sharded)}")
    print(f"  NOT sharded (replicated but expected to shard): {len(not_sharded)}")
    print(f"  Missing in some ranks: {len(missing_in_some)}")
    print(f"  Ambiguous / unexpected layout: {len(ambiguous)}\n")

    if not_sharded:
        print("Keys that appear NOT sharded (replicated full tensors on every rank):")
        for key, shapes, axis, expected in not_sharded:
            print(f"- {key}")
            for r, s in enumerate(shapes):
                print(f"    rank_{r} shape: {pretty_shape(s)}")
            print(f"    expected per-rank along dim {axis}: {expected}")
            print()
    if missing_in_some:
        print("Keys missing in at least one rank:")
        for key, shapes, axis in missing_in_some:
            print(f"- {key}")
            for r, s in enumerate(shapes):
                print(f"    rank_{r} shape: {pretty_shape(s)}")
            print()
    if ambiguous:
        print("Ambiguous keys (unexpected shapes):")
        for key, shapes, axis in ambiguous:
            print(f"- {key}  shard_axis={axis}")
            for r, s in enumerate(shapes):
                print(f"    rank_{r} shape: {pretty_shape(s)}")
            print()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp-root", type=Path, required=True,
                   help="Path to TP checkpoint root (contains rank_0, rank_1, ...)")
    p.add_argument("--world-size", type=int, default=2)
    args = p.parse_args()
    analyze(args.tp_root, args.world_size)

if __name__ == "__main__":
    main()