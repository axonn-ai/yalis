#!/usr/bin/env python3
"""
Tune NVSHMEM allreduce_preallocated kernel parameters for a given message size.

Usage examples (UID-based init, no mpi4py):
  torchrun --nproc_per_node=4 benchmark_tune_allreduce_preallocated.py \
    --size 1048576 --units elements --dtype float32 --alg recursive \
    --num-blocks 4,8,16,32 --threads-per-block 128,256,512 --chunk-bytes 16384,32768,65536,131072,262144 \
    --iterations 50 --warmup 10 --topk 10

Notes:
- Uses torch.distributed (NCCL) to broadcast NVSHMEM UID and to reduce timings across ranks.
- Initializes NVSHMEMCommWrapper via UID-based constructor (no mpi4py).
- Explores a grid of (num_blocks, threads_per_block, chunk_bytes) combinations.
- Measures per-rank GPU time with CUDA events and reports the global max latency (critical path) across ranks.
- Validates correctness (tensor values == world_size) for each parameter combination.
"""

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
    import nvshmem_comm_cuda
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
    parser = argparse.ArgumentParser(description="Tune kernel params for allreduce_preallocated")
    parser.add_argument("--size", type=int, required=True, help="Message size (elements or bytes depending on --units)")
    parser.add_argument("--units", choices=["elements", "bytes"], default="elements", help="Units for --size")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP.keys()), default="int32", help="Tensor dtype")
    parser.add_argument("--alg", choices=["recursive", "ring"], default="recursive", help="Allreduce algorithm")
    parser.add_argument("--num-blocks", type=str, default="4,8,16,32", help="CSV of grid values for num_blocks")
    parser.add_argument("--threads-per-block", type=str, default="128,256,512", help="CSV of grid values for threads per block (<=1024)")
    parser.add_argument("--chunk-bytes", type=str, default="16384,32768,65536,131072,262144", help="CSV of grid values for chunk size in bytes")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations per configuration")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations per configuration")
    parser.add_argument("--topk", type=int, default=10, help="Show top-K configurations")
    parser.add_argument("--output", type=str, default="tune_results.json", help="Path to save JSON results (rank 0)")
    parser.add_argument("--device", type=int, default=None, help="CUDA device id to use for this rank (default: auto-detect local rank)")

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

    # Device selection
    local_device = args.device if args.device is not None else detect_local_device(rank)
    torch.cuda.set_device(local_device)

    if rank == 0:
        print("NVSHMEM allreduce_preallocated parameter tuner")
        print(f"World size: {world_size}")
        print(f"Target algorithm: {args.alg}")

    dtype = DTYPE_MAP[args.dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    # Compute num elements from size and units
    if args.units == "elements":
        num_elems = args.size
    else:
        if args.size % dtype_size != 0:
            if rank == 0:
                print(f"Warning: --size {args.size} bytes is not divisible by dtype size {dtype_size}. Rounding down.")
        num_elems = args.size // dtype_size

    if num_elems <= 0:
        if rank == 0:
            print("Message size must be > 0 elements")
        sys.exit(1)

    # Broadcast NVSHMEM UID and initialize communicator via UID constructor
    uid_bytes = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()
    uid_gpu = uid_bytes.to(f"cuda:{local_device}")
    dist.broadcast(uid_gpu, src=0)
    dist.barrier()
    uid_cpu = uid_gpu.to("cpu")

    comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_device, uid_cpu)

    # Allocate preallocated tensor and id
    tensor, tensor_id = comm_wrapper.allocate_tensor(num_elems, dtype, torch.device(f"cuda:{local_device}"), nvshmem_comm_cuda.Protocol.SIMPLE)

    # Helper CUDA stream and events
    stream = torch.cuda.Stream(device=local_device)
    stream_ptr = stream.cuda_stream

    # Parameter grid
    grid_num_blocks = parse_int_list(args.num_blocks)
    grid_tpb = parse_int_list(args.threads_per_block)
    grid_chunk_bytes = parse_int_list(args.chunk_bytes)

    # Validate TPB values
    grid_tpb = [v for v in grid_tpb if 1 <= v <= 1024]
    if rank == 0 and len(grid_tpb) == 0:
        print("No valid threads-per-block values (must be 1..1024)")
        sys.exit(1)

    def valid_combo(nb: int, tpb: int, chunk_b: int) -> bool:
        # Guard constraints from kernels: partitioning uses integer division per block.
        # To avoid dropped tail elements, require exact divisibility by num_blocks.
        if num_elems % nb != 0:
            return False
        elems_per_block = num_elems // nb
        if args.alg == "ring":
            # Ring kernel computes chunk_elems = chunk_bytes/sizeof(T)
            chunk_elems = max(1, chunk_b // dtype_size)
            if chunk_elems == 0:
                return False
            if elems_per_block % chunk_elems != 0:
                return False
        else:  # recursive
            # Recursive kernel uses chunk_elems = max(32, chunk_bytes/sizeof(T))
            chunk_elems = max(32, chunk_b // dtype_size)
            if chunk_elems == 0:
                return False
            if elems_per_block % chunk_elems != 0:
                return False
        return True

    search_space = [(nb, tpb, cb) for nb, tpb, cb in product(grid_num_blocks, grid_tpb, grid_chunk_bytes) if valid_combo(nb, tpb, cb)]

    if rank == 0:
        print(f"Benchmarking message size: {num_elems} elements ({num_elems * dtype_size / (1024**2):.2f} MiB), dtype={args.dtype}")
        print(f"Search space size: {len(search_space)} combinations")

    # Ensure all ranks are ready
    dist.barrier()

    results = []

    for nb, tpb, cb in search_space:
        # Sync ranks before changing params
        dist.barrier()
        comm_wrapper.set_kernel_params(nvshmem_comm_cuda.Protocol.SIMPLE, nb, tpb, cb)

        # Warmup
        with torch.cuda.stream(stream):
            for _ in range(args.warmup):
                tensor.fill_(1)
                comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, args.alg)

        torch.cuda.synchronize(device=local_device)
        dist.barrier()

        # Measure
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(stream):
            start_event.record()
            for _ in range(args.iterations):
                tensor.fill_(0.01)
                comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, args.alg)
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
            tensor.fill_(0.01)
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, stream_ptr, args.alg)
        torch.cuda.synchronize(device=local_device)

        expected = torch.ones(num_elems, dtype=dtype, device=f"cuda:{local_device}") * world_size * 0.01
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

        if rank == 0:
            status = "OK" if all_ok else "FAIL"
            print(f"nb={nb:>3}, tpb={tpb:>4}, chunk_bytes={cb:>7} -> {global_avg_ms:8.4f} ms [{status}]")

        results.append({
            'num_blocks': nb,
            'threads_per_block': tpb,
            'chunk_bytes': cb,
            'avg_time_ms': global_avg_ms,
            'valid': all_ok,
        })

        dist.barrier()

    # Rank 0 aggregates/sorts and saves
    if rank == 0:
        valid_results = [r for r in results if r['valid']]
        valid_results.sort(key=lambda r: r['avg_time_ms'])

        topk = valid_results[: max(1, args.topk)] if valid_results else []

        print("\nTop configurations:")
        if topk:
            for r in topk:
                print(f"  nb={r['num_blocks']}, tpb={r['threads_per_block']}, chunk_bytes={r['chunk_bytes']} -> {r['avg_time_ms']:.4f} ms")
        else:
            print("  No valid configurations found.")

        meta = {
            'world_size': world_size,
            'dtype': args.dtype,
            'elements': num_elems,
            'bytes': num_elems * dtype_size,
            'algorithm': args.alg,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device_name': torch.cuda.get_device_name(local_device) if torch.cuda.is_available() else 'Unknown',
            'cuda_version': torch.version.cuda,
        }

        out = {
            'meta': meta,
            'results': results,
            'topk': topk,
        }

        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved results to {args.output}")

    dist.barrier()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

