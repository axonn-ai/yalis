import torch
import torch.distributed as dist
import socket


def print_rank0(msg):
    if not dist.is_initialized():
        raise RuntimeError(
            "Distributed process group is not initialized. "
            "Make sure init_distributed() is called before using this."
        )
    if dist.get_rank() == 0:
        print(f"{msg}")


def get_gpu_memory_info():
    BYTES_TO_GB = 1 / (1024**3)
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() * BYTES_TO_GB
        reserved = torch.cuda.memory_reserved() * BYTES_TO_GB
        return allocated, reserved
    return 0, 0


def get_max_gpu_memory_info():
    BYTES_TO_GB = 1 / (1024**3)
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() * BYTES_TO_GB
        reserved = torch.cuda.max_memory_reserved() * BYTES_TO_GB
        return allocated, reserved
    return 0, 0


def get_nvtx_funcs(enabled: bool):
    if enabled:

        def nvtx_range_push(msg):
            torch.cuda.nvtx.range_push(msg)

        def nvtx_range_pop():
            torch.cuda.nvtx.range_pop()

    else:

        def nvtx_range_push(msg):
            pass

        def nvtx_range_pop():
            pass

    return nvtx_range_push, nvtx_range_pop


def get_platform():
    if torch.cuda.is_available():
        if torch.version.cuda is not None:
            return "cuda"
        elif torch.version.hip is not None:
            return "rocm"
    return "cpu"


def is_process_group_within_node(group=None):
    local_hostname = socket.gethostname()

    # Convert hostname to bytes and pad to fixed length
    max_len = 128
    local_bytes = local_hostname.encode("utf-8")
    if len(local_bytes) > max_len:
        local_bytes = local_bytes[:max_len]
    local_bytes += b" " * (max_len - len(local_bytes))

    # Gather hostnames from all processes
    all_hostnames = [
        torch.empty(max_len, dtype=torch.uint8, device="cuda")
        for _ in range(torch.distributed.get_world_size(group))
    ]
    local_tensor = torch.tensor(
        list(local_bytes), dtype=torch.uint8, device="cuda"
    )
    torch.distributed.all_gather(all_hostnames, local_tensor, group=group)

    # Decode hostnames
    decoded_hostnames = [
        bytes(t.cpu().tolist()).decode("utf-8").strip() for t in all_hostnames
    ]

    return all(h == local_hostname for h in decoded_hostnames)
