from lightning.fabric import seed_everything
import torch
import torch.distributed as dist
from axonn import axonn as ax
import warnings


def init_distributed(tp_dims=None):
    # Passing device_id=local_rank will results in slighlty higher memory usage
    # - around 1GB This causes OOMs for some runs. For now, we are not passing
    # device_id which leads to a warning but it works fine. Ideally, we need to
    # find a way to do this to avoid the warning without OOMs.
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    else:
        current_backend = dist.get_backend()
        if str(current_backend) != "nccl":
            warnings.warn(
                f"Existing distributed process group backend '{current_backend}' does not "
                "match expected backend 'nccl'. Proceeding with existing configuration.",
                UserWarning,
            )
        else:
            print(f"Reusing existing distributed process group with backend '{current_backend}'.")
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    print(
        f"[{dist.get_rank()}] Current Device - {torch.cuda.get_device_properties(torch.cuda.current_device())}"  # noqa: E501
    )
    if tp_dims is None:
        tp_dims = (dist.get_world_size(), 1, 1)
    ax.init(G_intra_r=tp_dims[0], G_intra_c=tp_dims[1], G_intra_d=tp_dims[2])
    # this is very imp. to ensure the same token is sampled on each TP rank!
    # todo: seed should be set from InferenceConfig

    if dist.get_rank() != 0:
        # Suppress warnings from all non-zero ranks
        warnings.filterwarnings("ignore", category=UserWarning)
    seed_everything(1234)
