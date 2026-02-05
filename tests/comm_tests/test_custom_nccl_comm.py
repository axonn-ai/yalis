import torch
import torch.distributed as dist
from yalis.external.nccl_comm import CommHandler

DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
SIZES = [1, 4, 16, 128, 512, 1024, 2048, 4096]


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        return
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank
    torch.cuda.set_device(device)

    comm_idx = CommHandler.create_communicator_from_process_group(dist.group.WORLD)
    comm = CommHandler.get_communicator_from_idx(comm_idx)

    for dtype in DTYPES:
        for size in SIZES:
            tensor = torch.ones(size, device=device, dtype=dtype) * (rank + 1)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                comm.all_reduce(tensor, stream=stream)
            stream.synchronize()
            expected = world_size * (world_size + 1) / 2
            atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-6
            if not torch.allclose(tensor, torch.full_like(tensor, expected), atol=atol):
                print(
                    f"[Rank {rank}] FAILED: dtype={dtype}, size={size},"
                    f" tensor={tensor}, expected={expected}"
                )
            else:
                print(f"[Rank {rank}] PASSED: dtype={dtype}, size={size}")
    comm.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
