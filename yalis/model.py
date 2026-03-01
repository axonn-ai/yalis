from yalis.external.config import Config
import os
from pathlib import Path
from yalis.external.litgpt_utils import _EmptyInit, load_litgpt_checkpoint
from yalis.model_loading import load_checkpoint_safetensors
from yalis.external.model import GPT
import torch.distributed as dist
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
    prestore_kv_cache=True,
    disable_tp=False,
    default_vector_prefetching=False,
    default_vector_path=None,
):
    tensor_parallel = dist.get_world_size() > 1
    if disable_tp:
        print(
            f"Disabling tensor parallelism for {litgpt_checkpoint_directory}"
        )
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
    config.use_intra_head_parallelism = use_intra_head_parallelism
    config.attention_backend = attention_backend
    config.use_paged_kv_caching = use_paged_kv_caching
    config.prestore_kv_cache = prestore_kv_cache
    config.init_device = device if random_init else "meta"
    config.use_prefetched = default_vector_prefetching
    config.prefetch_default_vect_path = default_vector_path

    with _EmptyInit(enabled=(not random_init)):
        model = GPT(config).to(model_dtype)

    if not random_init:
        checkpoint_path = checkpoint_dir / "yalis_checkpoints"
        if os.path.exists(checkpoint_path):
            load_checkpoint_safetensors(model, checkpoint_path)
        else:
            # Deprecated loading path as a fallback
            checkpoint_path = checkpoint_dir / "lit_model.pth"
            load_litgpt_checkpoint(model, checkpoint_path)

    model.eval()
    return model
