import torch.distributed as dist

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")


