import torch
from litgpt.model import Config
from pathlib import Path
import sys
from yalis.external.litgpt_utils import load_checkpoint, _EmptyInit
from yalis.external.model import GPT
import torch.distributed as dist
import time


def get_model(
    model_config,
    model_dtype,
    max_sequence_length=None,
    random_init=False,
    device="cuda",
):
    litgpt_checkpoint_directory = model_config.model_path
    tensor_parallel = dist.get_world_size() > 1
    if model_config.disable_tp:
        tensor_parallel = False
    if tensor_parallel and dist.get_rank() == 0:
        print(f"Using Tensor parallelism on {dist.get_world_size()} GPUs")
    checkpoint_dir = Path(litgpt_checkpoint_directory)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    if max_sequence_length is not None:
        assert (
            max_sequence_length <= config.block_size
        ), f"Maximum sequence length for this model is {config.block_size}"
        config.block_size = max_sequence_length
    config.tensor_parallel = tensor_parallel
    config.tp_dims = model_config.tp_dims

    with _EmptyInit(enabled=(not random_init)):
        model = GPT(config).to(model_dtype)

    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(model, checkpoint_path)
    model.eval()
    return model
