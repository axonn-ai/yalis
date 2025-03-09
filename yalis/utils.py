import torch
import torch.distributed as dist
import time
import numpy as np
from typing import Optional, List, Tuple


def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")

def get_gpu_memory_info():
    BYTES_TO_GB = 1 / (1024 ** 3)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() * BYTES_TO_GB
        reserved = torch.cuda.memory_reserved() * BYTES_TO_GB
        return allocated, reserved
    return 0, 0

def test_allreduce_bandwidth(
    process_group: Optional[dist.ProcessGroup] = None,
    sizes: List[int] = [2**i for i in range(10, 28)],  # From 1KB to 256MB
    dtype: torch.dtype = torch.bfloat16,
    warmup_iterations: int = 5,
    benchmark_iterations: int = 10,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> Tuple[List[int], List[float], List[float]]:
    """
    Test the all-reduce bandwidth between processes in a distributed setting.

    Args:
        process_group: The process group to test. If None, uses the default process group.
        sizes: List of tensor sizes in number of elements to test.
        dtype: Data type of the tensor.
        warmup_iterations: Number of warmup iterations before benchmarking.
        benchmark_iterations: Number of iterations to average for benchmarking.
        device: Device to allocate tensors on.

    Returns:
        Tuple containing:
            - List of tensor sizes in bytes
            - List of bandwidths in GB/s
            - List of latencies in ms
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed module not initialized. Call dist.init_process_group() first.")

    if process_group is None:
        process_group = dist.group.WORLD

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    # Calculate element size in bytes
    element_size = torch.tensor([], dtype=dtype).element_size()

    # Prepare results containers
    size_bytes = []
    bandwidths = []
    latencies = []

    for size in sizes:
        # Convert size to bytes for reporting
        bytes_per_rank = size * element_size
        total_bytes = bytes_per_rank * world_size
        size_bytes.append(bytes_per_rank)

        # Create tensor on the specified device
        tensor = torch.ones(size, dtype=dtype, device=device) * (rank + 1)

        # Warmup
        for _ in range(warmup_iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)

        # Synchronize before timing
        torch.cuda.synchronize() if device.type == 'cuda' else None

        # Benchmark
        iteration_times = []
        for _ in range(benchmark_iterations):
            # Reset tensor
            tensor.fill_(rank + 1)

            # Synchronize before starting timer
            torch.cuda.synchronize() if device.type == 'cuda' else None

            start_time = time.time()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=process_group)

            # Synchronize before stopping timer
            torch.cuda.synchronize() if device.type == 'cuda' else None

            end_time = time.time()
            iteration_times.append(end_time - start_time)

        # Calculate average time and bandwidth
        avg_time_s = np.mean(iteration_times)
        latency_ms = avg_time_s * 1000
        #bandwidth_gbps = (total_bytes / avg_time_s) / (1024**3)  # GB/s

        bandwidth_gbps = (2 * ((world_size - 1) / world_size) * bytes_per_rank / avg_time_s) / (1024**3)

        bandwidths.append(bandwidth_gbps)
        latencies.append(latency_ms)

        # Print results from rank 0
        if rank == 0:
            print(f"Size: {bytes_per_rank/1024**2:.2f} MB/rank, "
                  f"Total: {total_bytes/1024**2:.2f} MB, "
                  f"Latency: {latency_ms:.3f} ms, "
                  f"Bandwidth: {bandwidth_gbps:.3f} GB/s")

    return size_bytes, bandwidths, latencies

