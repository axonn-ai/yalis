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
):
    tensor_parallel = dist.get_world_size() > 1
    if disable_tp:
        print(
            f"Disabling tensor parallelism for {litgpt_checkpoint_directory}"
        )
        tensor_parallel = False
    if tensor_parallel and dist.get_rank() == 0:
        print(f"Using Tensor parallelism on {dist.get_world_size()} GPUs")

    # Point to the checkpoint directory and log what we're using
    print(f"Using {litgpt_checkpoint_directory} as litgpt checkpoint directory with dtype {model_dtype}")
    checkpoint_dir = Path(litgpt_checkpoint_directory)

    # For TP inference, load config from rank-specific directory if available
    config_path = checkpoint_dir / "model_config.yaml"
    checkpoint_path = checkpoint_dir / "yalis_checkpoints"
    
    if tensor_parallel:
        tp_rank_dir = checkpoint_dir / "yalis_checkpoints_tp" / f"rank_{dist.get_rank()}"
        tp_checkpoint_path = tp_rank_dir / "yalis_checkpoints"
        tp_rank_config = tp_rank_dir / "model_config.yaml"
        tp_root_config = checkpoint_dir / "yalis_checkpoints_tp" / "model_config.yaml"
        tp_cp_exists = os.path.exists(tp_checkpoint_path)
        tp_rank_cfg_exists = os.path.exists(tp_rank_config)
        tp_root_cfg_exists = os.path.exists(tp_root_config)
        if dist.get_rank() == 0:
            print(f"[TP DEBUG] tp_checkpoint_path={tp_checkpoint_path} exists={tp_cp_exists}")
            print(f"[TP DEBUG] tp_rank_config={tp_rank_config} exists={tp_rank_cfg_exists}")
            print(f"[TP DEBUG] tp_root_config={tp_root_config} exists={tp_root_cfg_exists}")
        # Prefer rank-local config when available, otherwise fall back to root TP config.
        if tp_cp_exists and tp_rank_cfg_exists:
            config_path = tp_rank_config
            checkpoint_path = tp_checkpoint_path
            if dist.get_rank() == 0:
                print(f"[TP DEBUG] Using rank-local TP config and checkpoint")
        elif tp_cp_exists and tp_root_cfg_exists:
            config_path = tp_root_config
            checkpoint_path = tp_checkpoint_path
            if dist.get_rank() == 0:
                print(f"[TP DEBUG] Using root TP config and per-rank checkpoint")

    if dist.get_rank() == 0:
        print(f"[CONFIG DEBUG] Loading config from: {config_path}")
    config = Config.from_file(config_path)
    if dist.get_rank() == 0:
        print(f"[CONFIG DEBUG] Loaded config: n_expert={config.n_expert}, padded_vocab_size={config.padded_vocab_size}, vocab_size={config.vocab_size}, n_head={config.n_head}")
    
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

    if dist.get_rank() == 0:
        print(f"[CONFIG DEBUG] Model will be initialized with: n_expert={config.n_expert}, padded_vocab_size={config.padded_vocab_size}")
    with _EmptyInit(enabled=(not random_init)):
        model = GPT(config).to(model_dtype)

    if not random_init:
        if os.path.exists(checkpoint_path):
            load_checkpoint_safetensors(model, checkpoint_path)
        else:
            # Deprecated loading path as a fallback
            checkpoint_path = checkpoint_dir / "lit_model.pth"
            load_litgpt_checkpoint(model, checkpoint_path)

    model.eval()
    return model
