
try:
    from mpi4py import MPI
except ImportError:
    pass
import time
from typing import Optional
from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from yalis.model import get_model
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")

def init_everything(dtype: Optional[str] ="bf16-mixed"):
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
        strategy = AxonnStrategy(
                G_intra_r=world_size,
                G_intra_c=1,
                G_intra_d=1,
                overlap_communication=True,
                enable_timers=False,
            )
        fabric = Fabric(
            accelerator="gpu",
            devices=torch.cuda.device_count(),
            num_nodes=world_size // torch.cuda.device_count(),
            precision=dtype,
            strategy=strategy,
        )
    else:
        fabric = Fabric(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            precision=dtype,
        )
    fabric.launch()
    # this is very important to ensure that the same token is sampled on each TP rank!
    seed_everything(1234)
    return fabric

