from lightning.fabric import seed_everything
import torch
import torch.distributed as dist
from axonn import axonn as ax
import os
import datetime


def init_distributed(
    tp_dims = None
):
    dist.init_process_group(backend="nccl") 
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    print (f"[{dist.get_rank()}] Current Device - {torch.cuda.get_device_properties(torch.cuda.current_device())}")
    if tp_dims is None:
        tp_dims = (dist.get_world_size(), 1, 1)
    ax.init(G_intra_r=tp_dims[0], G_intra_c=tp_dims[1], G_intra_d=tp_dims[2])
    # this is very important to ensure that the same token is sampled on each TP rank!
    # todo: seed should be set from InferenceConfig
    seed_everything(1234)
    
