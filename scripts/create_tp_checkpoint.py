#!/usr/bin/env python3
"""
Creates a properly sharded checkpoint that can be loaded with disable_tp=False.

How it works:
1. Rank 0 loads the full unshard checkpoint from yalis_checkpoints
2. We manually compute shards for each rank based on TP dimensions
3. Rank 0 broadcasts each rank's shard to that rank via P2P communications
4. Each rank saves its local shard to disk
5. The result: a checkpoint where each rank has its subset of weights

Usage:
    torchrun --nproc_per_node=2 scripts/create_tp_checkpoint.py
"""
import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from yalis.initialize import init_distributed
from yalis.external.safetensor_saver import incremental_save
from safetensors.torch import load_file as load_safetensors
import warnings
warnings.filterwarnings("ignore")


class SafePrinter:
    @staticmethod
    def print(msg: str):
        try:
            if dist.is_initialized() and dist.get_rank() == 0:
                print(msg)
            elif not dist.is_initialized():
                print(msg)
        except Exception:
            print(msg)


def load_full_checkpoint(checkpoint_dir: Path) -> Dict[str, torch.Tensor]:
    """Load full unshard checkpoint on rank 0 only."""
    if dist.get_rank() != 0:
        return {}
    
    SafePrinter.print(f"[Rank 0] Loading full checkpoint...")
    index_path = checkpoint_dir / "model.safetensors.index.json"
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    weight_map = index.get("weight_map", {})
    state_dict = {}
    
    for key, shard_file in weight_map.items():
        shard_path = checkpoint_dir / shard_file
        tensors = load_safetensors(str(shard_path))
        state_dict.update(tensors)
    
    SafePrinter.print(f"[Rank 0] Loaded {len(state_dict)} tensors")
    return state_dict


def get_shard_indices(
    key: str,
    weight_shape: torch.Size,
    weight_ndim: int,
    rank: int,
    world_size: int,
) -> Optional[Tuple[int, int, int]]:
    """
    For a given weight, determine if it should be sharded and return (dim, start, end).
    
    Args:
        key: weight key name
        weight_shape: shape of the weight tensor
        weight_ndim: number of dimensions
        rank: target rank
        world_size: total number of ranks
    
    Returns:
        (shard_dim, start_idx, end_idx) if sharded, None if replicated
    """
    
    # Don't shard: embeddings, norms, routers, lm_head
    if any(x in key for x in ["embed", "norm", "router", "lm_head"]):
        return None
    
    # MoE weights
    if "mlp" in key and weight_ndim == 3:  # [n_experts, d1, d2]
        if "gate_up_proj" in key:
            # [n_experts, 2*intermediate, hidden] -> shard hidden (dim 2)
            d = 2
        elif "proj" in key:
            # [n_experts, hidden, intermediate] -> shard hidden (dim 1)
            d = 1
        else:
            return None
        
        size = weight_shape[d]
        shard_size = size // world_size
        if size % world_size != 0:
            raise ValueError(f"Cannot evenly shard {key} dim {d} (size {size}) across {world_size} ranks")
        return (d, rank * shard_size, (rank + 1) * shard_size)
    
    # Linear weights [out, in] -> shard out (dim 0)
    if weight_ndim == 2:
        d = 0  # out_features
        size = weight_shape[d]
        shard_size = size // world_size
        if size % world_size != 0:
            raise ValueError(f"Cannot evenly shard {key} dim {d} (size {size}) across {world_size} ranks")
        return (d, rank * shard_size, (rank + 1) * shard_size)
    
    return None


def extract_shard(
    weight: torch.Tensor,
    dim: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Extract shard from full weight."""
    if dim == 0:
        return weight[start:end, ...].contiguous()
    elif dim == 1:
        return weight[:, start:end, ...].contiguous()
    elif dim == 2:
        return weight[:, :, start:end, ...].contiguous()
    else:
        raise ValueError(f"Unsupported shard dim: {dim}")


def send_shard_to_rank(
    key: str,
    full_weight: Optional[torch.Tensor],
    weight_shape: torch.Size,
    weight_dtype: torch.dtype,
    weight_ndim: int,
    target_rank: int,
    world_size: int,
):
    """
    Rank 0 extracts shard for target_rank and sends it via P2P broadcast.
    target_rank receives the shard.
    """
    rank = dist.get_rank()
    
    # Determine shard info based on key and shape (all ranks can compute this)
    shard_info = get_shard_indices(key, weight_shape, weight_ndim, target_rank, world_size)
    
    if shard_info is None:
        # Replicate: broadcast full weight
        if rank == 0:
            shard = full_weight.clone()
        else:
            shard = torch.zeros(weight_shape, dtype=weight_dtype)
        
        shard = shard.to("cuda")
        dist.broadcast(shard, src=0)
        return shard.cpu()
    
    # Shard: extract and send only this rank's portion
    dim, start, end = shard_info
    
    if rank == 0:
        shard = extract_shard(full_weight, dim, start, end)
    else:
        # Create empty tensor of correct shape
        shape = list(weight_shape)
        shape[dim] = (end - start)
        shard = torch.zeros(shape, dtype=weight_dtype)
    
    shard = shard.to("cuda")
    dist.broadcast(shard, src=0)
    return shard.cpu()


def create_tp_checkpoint(checkpoint_dir: Path, output_dir: Path):
    """Main conversion logic."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Rank 0 loads checkpoint
    full_state_dict = load_full_checkpoint(checkpoint_dir) if rank == 0 else {}
    dist.barrier()
    
    # All ranks iterate through keys and receive their shards
    if rank == 0:
        all_keys = list(full_state_dict.keys())
        key_list = [all_keys]
    else:
        key_list = [None]
    
    dist.broadcast_object_list(key_list, src=0)
    all_keys = key_list[0]
    
    SafePrinter.print(f"\n{'='*80}")
    SafePrinter.print(f"[Rank {rank}] Creating TP shards for {len(all_keys)} tensors")
    SafePrinter.print(f"{'='*80}\n")
    
    rank_state_dict = {}
    
    for i, key in enumerate(all_keys):
        if i % max(1, len(all_keys) // 10) == 0 and rank == 0:
            SafePrinter.print(f"[Rank 0] Processing {i+1}/{len(all_keys)} ...")
        
        # Rank 0 has the full weight; send this rank's shard to all ranks
        if rank == 0:
            full_weight = full_state_dict[key]
            weight_shape = full_weight.shape
            weight_dtype = full_weight.dtype
            weight_ndim = full_weight.ndim
        else:
            full_weight = None
            weight_shape = None
            weight_dtype = None
            weight_ndim = None
        
        # Broadcast shape/dtype info to all ranks
        shape_list = [weight_shape]
        dtype_list = [weight_dtype]
        ndim_list = [weight_ndim]
        dist.broadcast_object_list(shape_list, src=0)
        dist.broadcast_object_list(dtype_list, src=0)
        dist.broadcast_object_list(ndim_list, src=0)
        
        weight_shape = shape_list[0]
        weight_dtype = dtype_list[0]
        weight_ndim = ndim_list[0]
        
        # P2P: each rank receives its shard
        local_shard = send_shard_to_rank(key, full_weight, weight_shape, weight_dtype, weight_ndim, rank, world_size)
        rank_state_dict[key] = local_shard
    
    dist.barrier()
    
    # Each rank saves its local checkpoint
    SafePrinter.print(f"\n[Rank {rank}] Saving local checkpoint to {output_dir} ...")

    output_dir.mkdir(parents=True, exist_ok=True)
    rank_output_dir = output_dir / f"rank_{rank}"
    rank_output_dir.mkdir(parents=True, exist_ok=True)

    # Create a nested `yalis_checkpoints` directory inside each rank dir so
    # the existing loader (which looks for `<rank_dir>/yalis_checkpoints`) can
    # find a safetensors index and shards without falling back to .pth files.
    safetensors_dir = rank_output_dir / "yalis_checkpoints"
    safetensors_dir.mkdir(parents=True, exist_ok=True)

    with incremental_save(str(safetensors_dir), max_shard_size_bytes=4*(1024**3)) as saver:
        for name, param in rank_state_dict.items():
            saver.store_early(name, param)
    
    SafePrinter.print(f"[Rank {rank}] Saved {len(rank_state_dict)} tensors")

    # Copy metadata into each rank directory so loaders can find config/tokenizer
    metadata_files = [
        "model_config.yaml",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ]
    for fname in metadata_files:
        src = checkpoint_dir / fname
        if src.exists():
            try:
                shutil.copy2(src, rank_output_dir / fname)
            except Exception:
                SafePrinter.print(f"[Rank {rank}] Warning: failed to copy metadata file {fname} to {rank_output_dir}")

    # Only rank 0 copies metadata to the root and creates tp_index.json
    if rank == 0:
        for fname in metadata_files:
            src = checkpoint_dir / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)

        # Create index file that points to rank subdirs
        index = {
            "world_size": world_size,
            "rank_dirs": {f"rank_{r}": f"rank_{r}" for r in range(world_size)},
        }
        with open(output_dir / "tp_index.json", "w") as f:
            json.dump(index, f, indent=2)
    
    dist.barrier()
    SafePrinter.print(f"\n{'='*80}")
    SafePrinter.print(f"[Rank {rank}] TP Checkpoint Created")
    SafePrinter.print(f"{'='*80}\n")


def main():
    checkpoint_dir = Path("yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints")
    output_dir = Path("yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints_tp")
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    
    if world_size > 1 and not dist.is_initialized():
        init_distributed(tp_dims=(world_size, 1, 1))
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    SafePrinter.print(f"\n{'='*80}")
    SafePrinter.print(f"TP Checkpoint Conversion - Full Solution")
    SafePrinter.print(f"Input:  {checkpoint_dir}")
    SafePrinter.print(f"Output: {output_dir}")
    SafePrinter.print(f"World size: {world_size}, Rank: {rank}")
    SafePrinter.print(f"{'='*80}\n")
    
    create_tp_checkpoint(checkpoint_dir, output_dir)
    
    if dist.is_initialized():
        dist.barrier()
    
    SafePrinter.print(f"\n{'='*80}")
    SafePrinter.print("Done! Checkpoint ready for TP inference")
    SafePrinter.print(f"Use: model_id = '{output_dir}'")
    SafePrinter.print(f"Set: disable_tp=False")
    SafePrinter.print(f"{'='*80}\n")
    
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
