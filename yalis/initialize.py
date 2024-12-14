from typing import Optional
from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
import torch
import torch.distributed as dist

yalis_fabric = None


def init_distributed():
    global yalis_fabric
    if yalis_fabric is not None:
        return yalis_fabric
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
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
            num_nodes=(world_size + torch.cuda.device_count() -1)  // torch.cuda.device_count(),
            strategy=strategy,
        )
    else:
        fabric = Fabric(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
        )
    fabric.launch()
    # this is very important to ensure that the same token is sampled on each TP rank!
    # todo: seed should be set from InferenceConfig
    seed_everything(1234)
    yalis_fabric = fabric
    return fabric
