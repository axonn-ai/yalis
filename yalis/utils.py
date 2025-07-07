import torch
import torch.distributed as dist


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
