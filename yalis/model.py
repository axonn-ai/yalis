import torch
from litgpt.model import Config
from pathlib import Path
import sys
from litgpt.utils import load_checkpoint
from yalis.external.model import GPT
import torch.distributed as dist


def get_model(model_id, fabric, litgpt_checkpoint_directory, random_init=False):
    tensor_parallel = dist.get_world_size() > 1
    if tensor_parallel and dist.get_rank() == 0:
        print(f"Using Tensor parallelism on {dist.get_world_size()} GPUs")
    checkpoint_dir = Path(litgpt_checkpoint_directory)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    config.tensor_parallel = tensor_parallel
    model = GPT(config).to(torch.bfloat16)
    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(fabric, model, checkpoint_path)
    model.eval()
    return model
