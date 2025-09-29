#!/usr/bin/env python3

"""
Tune NVSHMEM allreduce_preallocated kernel parameters for one or more message sizes.

Quick start (single size, bytes):
  torchrun --nproc_per_node=4 benchmark_tune_allreduce_preallocated.py \
    --size 1048576 --dtype float32 \
    --num-blocks 4,8,16,32 --threads-per-block 128,256,512 --chunk-bytes 16384,32768,65536,131072,262144 \
    --iterations 50 --warmup 10 --topk 10

Multiple sizes at once (bytes with suffixes supported):
  torchrun --nproc_per_node=4 benchmark_tune_allreduce_preallocated.py \
    --sizes 64KiB,1MiB,8MiB --dtype float32 \
    --output tune_results.json --best-output best_by_size.json

Notes:
- Uses torch.distributed (NCCL) to broadcast NVSHMEM UID and to reduce timings across ranks.
- Initializes NVSHMEMCommWrapper via UID-based constructor (no mpi4py).
- Explores a grid of (num_blocks, threads_per_block, chunk_bytes) combinations.
- Measures per-rank GPU time with CUDA events and reports the global max latency (critical path) across ranks.
- Validates correctness (tensor values == world_size) for each parameter combination.
"""

try:
    from mpi4py import MPI
    print("✓ Successfully imported MPI4Py")
except ImportError as e:
    print(f"✗ Failed to import MPI4Py: {e}")
    print("Please install MPI4Py: pip install mpi4py")
    sys.exit(1)
import os
import sys
import json
import time
import math
import argparse
from itertools import product

import torch
import torch.distributed as dist
import numpy as np

# Add the build directory to the Python path so we can import the extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    from yalis_nvshmem_collectives import nvshmem_comm_cuda
except ImportError as e:
    print(f"Failed to import nvshmem_comm_cuda extension: {e}")
    sys.exit(1)


DTYPE_MAP = {
    'int32': torch.int32,
    'float32': torch.float32,
    'half': torch.float16,
    'bfloat16': torch.bfloat16,
    'int': torch.int32,  # alias
    'float': torch.float32,  # alias
}


def parse_int_list(csv: str):
    return [int(x.strip()) for x in csv.split(',') if x.strip()]


def parse_size_to_bytes(token: str) -> int:
    """Parse human-friendly size tokens like '64K', '64KiB', '1M', '1MiB', '2GB', case-insensitive.
    Returns the size in bytes as an integer. If the token is a bare integer, interpret as bytes.
    Accepts suffixes: K, M, G, KiB, MiB, GiB, KB, MB, GB.
    """
    s = token.strip().replace('_', '').lower()
    if not s:
        raise ValueError("empty size token")
    # If purely numeric, treat as bytes
    if s.isdigit():
        return int(s)
    # Split numeric prefix and suffix
    i = 0
    while i < len(s) and (s[i].isdigit() or s[i] == '.'):  # allow decimal, will floor later
        i += 1
    num_str = s[:i]
    unit = s[i:]
    value = float(num_str)
    # Binary prefixes default; accept decimal too
    if unit in ('k', 'kb'):
        return int(value * (1000 ** 1))
    if unit in ('m', 'mb'):
        return int(value * (1000 ** 2))
    if unit in ('g', 'gb'):
        return int(value * (1000 ** 3))
    if unit in ('kib', 'ki'):
        return int(value * (1024 ** 1))
    if unit in ('mib', 'mi'):
        return int(value * (1024 ** 2))
    if unit in ('gib', 'gi'):
        return int(value * (1024 ** 3))
    if unit in ('b',):
        return int(value)
    # Fallback: unknown unit → try int
    try:
        return int(float(s))
    except Exception as e:
        raise ValueError(f"Unrecognized size token: {token}") from e


def parse_size_list(csv: str) -> list:
    return [x for x in (t.strip() for t in csv.split(',')) if x]


def detect_local_device(rank: int) -> int:
    # Prefer LOCAL_RANK (torchrun). Fallback to common MPI env vars if present.
    for key in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID", "MV2_COMM_WORLD_LOCAL_RANK"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    # Fallback: rank modulo device count
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("No CUDA devices available")
    return rank % device_count


def main():
    parser = argparse.ArgumentParser(description="Tune kernel params for allreduce_preallocated (sizes in BYTES)")
    # Either --size or --sizes must be provided (bytes only). --sizes supports 64KiB, 1MiB, etc.
    parser.add_argument("--size", type=int, required=False, help="Single message size in bytes")
    parser.add_argument("--sizes", type=str, default=None, help="CSV of message sizes in bytes (supports 64KiB, 1MiB, 2GB)")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="int32", help="Tensor dtype")
    parser.add_argument("--num-blocks", type=str, default="4,8,16,32", help="CSV of grid values for num_blocks")
    parser.add_argument("--threads-per-block", type=str, default="128,256,512", help="CSV of grid values for threads per block (<=1024)")
    parser.add_argument("--chunk-bytes", type=str, default="16384,32768,65536,131072,262144", help="CSV of grid values for chunk size in bytes")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations per configuration")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per configuration")
    parser.add_argument("--topk", type=int, default=10, help="Show top-K configurations")
    parser.add_argument("--output", type=str, default="tune_results.json", help="Path to save detailed JSON results (rank 0)")
    parser.add_argument("--best-output", type=str, default=None, help="Optional path to save message_size -> best_params JSON (rank 0)")
    parser.add_argument("--quiet", action="store_true", help="Reduce non-rank0 logging")

    args = parser.parse_args()

    # Initialize torch.distributed (NCCL)
    if not dist.is_available():
        print("torch.distributed not available")
        sys.exit(1)
    if not dist.is_initialized():
        backend = "nccl"
        dist.init_process_group(backend=backend, init_method=os.environ.get("DIST_INIT_METHOD", "env://"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Device selection (auto-detect from local rank)
    local_device = detect_local_device(rank)
    torch.cuda.set_device(local_device)

    if rank == 0:
        print("NVSHMEM allreduce_preallocated parameter tuner")
        print(f"World size: {world_size}")

    dtype = DTYPE_MAP[args.dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    # Resolve list of message sizes in elements (inputs are bytes)
    sizes_elements = []
    if args.sizes is not None and args.sizes.strip():
        tokens = parse_size_list(args.sizes)
        bytes_list = [parse_size_to_bytes(t) for t in tokens]
        for b in bytes_list:
            if b % dtype_size != 0 and rank == 0:
                print(f"Warning: size {b} bytes not divisible by dtype size {dtype_size}. Rounding down.")
            sizes_elements.append(b // dtype_size)
    elif args.size is not None:
        if args.size % dtype_size != 0 and rank == 0:
            print(f"Warning: --size {args.size} bytes is not divisible by dtype size {dtype_size}. Rounding down.")
        sizes_elements = [args.size // dtype_size]
    else:
        if rank == 0:
            print("Must provide either --size or --sizes")
        sys.exit(1)

    # Filter non-positive sizes
    sizes_elements = [n for n in sizes_elements if n > 0]
    if not sizes_elements:
        if rank == 0:
            print("No valid message sizes to tune. Ensure sizes > 0.")
        sys.exit(1)

    # Broadcast NVSHMEM UID and initialize communicator via UID constructor
    uid_bytes = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()
    uid_gpu = uid_bytes.to(f"cuda:{local_device}")
    dist.broadcast(uid_gpu, src=0)
    dist.barrier()
    uid_cpu = uid_gpu.to("cpu")

    comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_device, uid_cpu)

    # Parameter grid
    grid_num_blocks = parse_int_list(args.num_blocks)
    grid_tpb = parse_int_list(args.threads_per_block)
    grid_chunk_bytes = parse_int_list(args.chunk_bytes)

    # Validate TPB values
    grid_tpb = [v for v in grid_tpb if 1 <= v <= 1024]
    if rank == 0 and len(grid_tpb) == 0:
        print("No valid threads-per-block values (must be 1..1024)")
        sys.exit(1)

    # Helper stream used across sizes
    stream = torch.cuda.Stream(device=local_device)
    stream_ptr = stream.cuda_stream

    def tune_one_size(num_elems: int):
        # Allocate preallocated tensor and id for this size
        tensor, tensor_id = comm_wrapper.allocate_tensor(num_elems, dtype, torch.device(f"cuda:{local_device}"), nvshmem_comm_cuda.Protocol.LL8)

        def valid_combo(nb: int, tpb: int, chunk_b: int) -> bool:
            # Guard constraints from kernels: partitioning uses integer division per block.
            if num_elems % nb != 0:
                return False
            elems_per_block = num_elems // nb
            # Recursive algorithm chunk sizing
            chunk_elems = max(32, chunk_b // dtype_size)
            if chunk_elems == 0:
                return False
            if elems_per_block % chunk_elems != 0:
                return False
            return True

        search_space = [(nb, tpb, cb) for nb, tpb, cb in product(grid_num_blocks, grid_tpb, grid_chunk_bytes) if valid_combo(nb, tpb, cb)]

        if rank == 0 and not args.quiet:
            print(f"\nBenchmarking message size: {num_elems} elements ({num_elems * dtype_size / (1024**2):.2f} MiB), dtype={args.dtype}")
            print(f"Search space size: {len(search_space)} combinations")

        # Ensure all ranks are ready
        dist.barrier()

        results_local = []

        for nb, tpb, cb in search_space:
            # Sync ranks before changing params
            dist.barrier()
            comm_wrapper.set_kernel_params(nvshmem_comm_cuda.Protocol.LL8, nb, tpb, cb)

            # Warmup
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    tensor.fill_(1)
                    comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "recursive")

            torch.cuda.synchronize(device=local_device)
            dist.barrier()

            # Measure
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            with torch.cuda.stream(stream):
                start_event.record()
                for _ in range(args.iterations):
                    tensor.fill_(0.01)
                    comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "recursive")
                end_event.record()

            torch.cuda.synchronize(device=local_device)

            local_total_ms = float(start_event.elapsed_time(end_event))
            # Use the max across ranks as the collective latency
            time_tensor = torch.tensor([local_total_ms], device=f"cuda:{local_device}", dtype=torch.float64)
            dist.all_reduce(time_tensor, op=dist.ReduceOp.MAX)
            global_total_ms = float(time_tensor.item())
            global_avg_ms = global_total_ms / max(1, args.iterations)

            # Verify correctness once for this configuration
            with torch.cuda.stream(stream):
                tensor.fill_(1)
                comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, "recursive")
            torch.cuda.synchronize(device=local_device)

            expected = torch.ones(num_elems, dtype=dtype, device=f"cuda:{local_device}") * world_size * 1
            if tensor.dtype in (torch.float16, torch.bfloat16):
                ok = bool(torch.allclose(tensor, expected, rtol=1e-2, atol=1e-2))
            elif tensor.dtype == torch.float32:
                ok = bool(torch.allclose(tensor, expected, rtol=1e-4, atol=1e-5))
            else:
                ok = bool(torch.equal(tensor, expected))

            # Reduce correctness across ranks (all must pass)
            ok_tensor = torch.tensor([1 if ok else 0], device=f"cuda:{local_device}", dtype=torch.int)
            dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
            all_ok = (int(ok_tensor.item()) == 1)

            if rank == 0 and not args.quiet:
                status = "OK" if all_ok else "FAIL"
                print(f"nb={nb:>3}, tpb={tpb:>4}, chunk_bytes={cb:>7} -> {global_avg_ms:8.4f} ms [{status}]")

            results_local.append({
                'num_blocks': nb,
                'threads_per_block': tpb,
                'chunk_bytes': cb,
                'avg_time_ms': global_avg_ms,
                'valid': all_ok,
            })

            dist.barrier()

        # Sort and summarize
        valid_results = [r for r in results_local if r['valid']]
        valid_results.sort(key=lambda r: r['avg_time_ms'])
        topk = valid_results[: max(1, args.topk)] if valid_results else []

        if rank == 0 and not args.quiet:
            print("\nTop configurations:")
            if topk:
                for r in topk:
                    print(f"  nb={r['num_blocks']}, tpb={r['threads_per_block']}, chunk_bytes={r['chunk_bytes']} -> {r['avg_time_ms']:.4f} ms")
            else:
                print("  No valid configurations found.")

        # Free resources for this size
        comm_wrapper.free_tensor(tensor_id)

        return results_local, topk

    # Containers for multi-size outputs (rank 0 will write)
    per_size = {}
    best_map = {}

    for idx, num_elems in enumerate(sizes_elements):
        # Show progress per size on rank 0
        if rank == 0 and not args.quiet:
            total = len(sizes_elements)
            print(f"\n=== Size {idx+1}/{total} ===")

        results_this, topk_this = tune_one_size(num_elems)

        # Aggregate on rank 0 only (data already reduced across ranks)
        if rank == 0:
            key_elems = str(num_elems)
            key_bytes = str(num_elems * dtype_size)
            per_size[key_elems] = {
                'elements': num_elems,
                'bytes': num_elems * dtype_size,
                'results': results_this,
                'topk': topk_this,
            }
            if topk_this:
                best = topk_this[0]
                best_entry = {
                    'num_blocks': best['num_blocks'],
                    'threads_per_block': best['threads_per_block'],
                    'chunk_bytes': best['chunk_bytes'],
                    'avg_time_ms': best['avg_time_ms'],
                    'algorithm': 'recursive',
                    'dtype': args.dtype,
                }
                best_map[key_bytes] = best_entry

        # Ensure sync between sizes
        dist.barrier()

    # Rank 0 aggregates/sorts and saves
    if rank == 0:
        meta = {
            'world_size': world_size,
            'dtype': args.dtype,
            'algorithm': 'recursive',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_name': torch.cuda.get_device_name(local_device) if torch.cuda.is_available() else 'Unknown',
            'cuda_version': torch.version.cuda,
            'grid': {
                'num_blocks': grid_num_blocks,
                'threads_per_block': grid_tpb,
                'chunk_bytes': grid_chunk_bytes,
            },
        }

        out = {
            'meta': meta,
            'per_size': per_size,
            'best_map': best_map,
        }

        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        if not args.quiet:
            print(f"\nSaved detailed results to {args.output}")

        if args.best_output:
            with open(args.best_output, 'w') as f:
                json.dump(best_map, f, indent=2)
            if not args.quiet:
                print(f"Saved best mapping to {args.best_output}")

    dist.barrier()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    MPI.Finalize()


if __name__ == "__main__":
    main()

