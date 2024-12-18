from lightning.fabric import seed_everything
import torch
import torch.distributed as dist
from axonn import axonn as ax


def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    ax.init(G_intra_r=dist.get_world_size(), G_intra_c=1, G_intra_d=1)
    # this is very important to ensure that the same token is sampled on each TP rank!
    # todo: seed should be set from InferenceConfig
    seed_everything(1234)
    
