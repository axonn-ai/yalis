import torch
import torch.distributed as dist
from yalis.external.nccl_comm import NCCLCommunicator, ncclUniqueId, CommHandler

CommHandler.load_commlib()

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank
    torch.cuda.set_device(device)

    comm_idx = CommHandler.create_communicator_from_process_group(dist.group.WORLD)

    comm = CommHandler.get_communicator_from_idx(comm_idx)

    # Test allreduce for all float types
    for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        tensor = torch.ones(4, device=device, dtype=dtype) * (rank + 1)
        # Print tensor
        print(f"Rank {rank}, dtype {dtype}: Tensor: {tensor}")
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            comm.all_reduce(tensor, stream=stream)
        stream.synchronize()
        expected = world_size * (world_size + 1) / 2
        print(f"Rank {rank}, dtype {dtype}: Expected: {expected}")
        atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-6
        if not torch.allclose(tensor, torch.full_like(tensor, expected), atol=atol):
            print(f"Rank {rank}, dtype {dtype}: Test failed! Tensor: {tensor}")
        else:
            print(f"Rank {rank}, dtype {dtype}: Test passed.")

    comm.destroy()

if __name__ == "__main__":
    main() 
