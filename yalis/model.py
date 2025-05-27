import torch
from yalis.external.config import Config
from pathlib import Path
import sys
from yalis.external.litgpt_utils import load_checkpoint, _EmptyInit
from yalis.external.model import GPT
import torch.distributed as dist
import time
from yalis.attention.backends import AttentionBackend


def get_model(
    litgpt_checkpoint_directory,
    model_dtype,
    max_sequence_length=None,
    random_init=False,
    device="cuda",
    use_intra_head_parallelism=False,
    attention_backend=AttentionBackend.FLASH,
    use_paged_kv_caching=False,
    prestore_kv_cache=True
):
    tensor_parallel = dist.get_world_size() > 1
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
    config.use_intra_head_parallelism = use_intra_head_parallelism
    config.attention_backend = attention_backend
    config.use_paged_kv_caching = use_paged_kv_caching
    config.prestore_kv_cache = prestore_kv_cache
    config.init_device = device if random_init else "meta"

    with _EmptyInit(enabled=(not random_init)):
        model = GPT(config).to(model_dtype)

    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(model, checkpoint_path)
    model.eval()
    return model
