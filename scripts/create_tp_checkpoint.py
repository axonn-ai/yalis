#!/usr/bin/env python3
"""
Creates a TP-compatible checkpoint with canonically sharded tensors.

How it works:
1. Rank 0 loads the full unsharded checkpoint from yalis_checkpoints
2. Rank 0 broadcasts each tensor to all ranks via collective broadcast
3. Each rank extracts its local shard (local_out_features, local_in_features) for 2D linear weights
4. Each rank saves its sharded tensors to its own directory
5. The loader detects these as sharded and uses them directly

Usage:
    torchrun --nproc_per_node=2 scripts/create_tp_checkpoint.py
"""
import argparse
import os
import sys
import json
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Optional, Tuple
import shutil

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


def _shard_range(size: int, groups: int, index: int) -> Tuple[int, int]:
    if groups <= 0:
        raise ValueError("Tensor parallel group size must be > 0")
    if size % groups != 0:
        raise ValueError(
            f"Cannot evenly shard dimension of size {size} across {groups} groups"
        )
    chunk = size // groups
    start = index * chunk
    return start, start + chunk


def compute_rank_coords(rank: int, inner_size: int, outer_size: int, depth_size: int):
    if inner_size <= 0 or outer_size <= 0 or depth_size <= 0:
        raise ValueError("Tensor parallel mesh sizes must be positive")
    ranks_per_plane = inner_size * outer_size
    if rank >= ranks_per_plane * depth_size:
        raise ValueError("Rank exceeds mesh capacity")
    depth_rank = rank // ranks_per_plane
    remainder = rank % ranks_per_plane
    inner_rank = remainder % inner_size
    outer_rank = remainder // inner_size
    return inner_rank, outer_rank, depth_rank


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
    orig_ndim: int,
    rank: int,
    inner_rank: int,
    outer_rank: int,
    inner_size: int,
    outer_size: int,
    world_size: int,
) -> Optional[Tuple]:
    """
    For a given weight, determine if it should be sharded and return sharding info.
    
    Args:
        key: weight key name
        weight_shape: shape of the weight tensor
        weight_ndim: number of dimensions (actual tensor ndim, not reference)
        rank: target rank
        world_size: total number of ranks
    
    Returns:
        For 3D (MoE): (dim, start, end)
        For 2D (linear): (dim0, start0, end0, dim1, start1, end1)
        For 1D (bias): (dim, start, end)
        None if replicated
    """
    
    # Don't shard: embeddings, norms, routers, gate
    # Note: `lm_head` must be sharded across the vocab (out) dimension to
    # produce TP-consistent checkpoints. Previously lm_head was excluded here
    # which led to full, unsharded lm_head tensors ending up in shards.
    # Conversely, the gating linear (`*.mlp.gate.weight`) should be replicated
    # across ranks because the model constructs a full gate linear layer
    # (`nn.Linear(..., n_expert)`) on each rank. Replicating gate ensures the
    # model and checkpoint shapes match.
    if any(x in key for x in ["embed", "norm", "router", "gate"]):
        return None
    
    # MoE biases MUST be checked BEFORE generic 1D bias handling
    # because MoE biases are 2D: (n_experts, intermediate)
    if "mlp" in key and key.endswith("bias") and weight_ndim == 2:
        # mlp1_bias: (n_experts, 2*intermediate) -> shard dim 1
        # mlp2_bias: (n_experts, hidden) -> replicated (don't shard)
        if "mlp1_bias" in key:
            size = weight_shape[1]
            shard_size = size // world_size
            if size % world_size != 0:
                raise ValueError(f"Cannot evenly shard {key} dim 1 (size {size}) across {world_size} ranks")
            return (1, rank * shard_size, (rank + 1) * shard_size)
        else:
            # mlp2_bias, gate.weight, or other MoE 2D tensors are replicated
            return None
    
    # Special handling for 1D biases (NOT MoE biases which are 2D)
    # Use orig_ndim (actual tensor dim) not weight_ndim (reference shape dim),
    # since biases are often matched to 2D weight references but are themselves 1D.
    if key.endswith(".bias") and orig_ndim == 1:
        out_size = weight_shape[0]
        # If this bias corresponds to a transposed proj (e.g. '*.proj.bias') the
        # runtime expects the sharding to follow the swapped inner/outer groups
        # (same rule we apply for the corresponding weight). Detect that case
        # and shard using the inner mesh axis instead of the outer one.
        transpose_like_bias = ".proj.bias" in key or key.endswith(".proj.bias")
        if transpose_like_bias:
            start, end = _shard_range(out_size, inner_size, inner_rank)
        else:
            start, end = _shard_range(out_size, outer_size, outer_rank)
        return (0, start, end)
    
    # Sinks tensor: (n_head, 1, 1) -> shard along head dimension (dim 0)
    # Used by GPT-OSS attention for sliding-window sink tokens
    if key.endswith(".sinks") and weight_ndim == 3:
        n_head = weight_shape[0]
        shard_size = n_head // world_size
        if n_head % world_size != 0:
            raise ValueError(f"Cannot evenly shard {key} dim 0 (n_head={n_head}) across {world_size} ranks")
        return (0, rank * shard_size, (rank + 1) * shard_size)
    
    # MoE weights (GPT-OSS MoE) - 3D tensors
    if "mlp" in key and weight_ndim == 3:  # [n_experts, d1, d2]
        # GPT-OSS MoE:
        # mlp1_weight: (n_experts, 2*intermediate, hidden) -> shard intermediate (dim 1)
        # mlp2_weight: (n_experts, hidden, intermediate) -> shard intermediate (dim 2)
        if "mlp1_weight" in key:
            d = 1  # shard the 2*intermediate dimension
        elif "mlp2_weight" in key:
            d = 2  # shard the intermediate dimension
        else:
            # Unknown 3D MoE weight, don't shard
            return None
        
        size = weight_shape[d]
        shard_size = size // world_size
        if size % world_size != 0:
            raise ValueError(f"Cannot evenly shard {key} dim {d} (size {size}) across {world_size} ranks")
        return (d, rank * shard_size, (rank + 1) * shard_size)
    
    # Linear weights [out, in] -> shard out (dim 0) and in (dim 1)
    if weight_ndim == 2:
        out_size = weight_shape[0]
        in_size = weight_shape[1]
        # Some TPLinear instances are created with `transpose=True` in the
        # model (for example attention/MMLP `*.proj` layers). When
        # `transpose=True` the runtime swaps inner/outer groups which means
        # the expected on-disk sharding for that weight is the opposite of
        # the default assumption. Detect those keys and swap the shard
        # computation accordingly so the produced shards match the loader
        # expectation.
        transpose_like = ".proj.weight" in key or key.endswith(".proj.weight")
        if transpose_like:
            # Swap the roles of inner/outer when computing shards:
            # - out dimension is sharded using `inner_size` / `inner_rank`
            # - in dimension is sharded using `outer_size` / `outer_rank`
            out_start, out_end = _shard_range(out_size, inner_size, inner_rank)
            in_start, in_end = _shard_range(in_size, outer_size, outer_rank)
        else:
            out_start, out_end = _shard_range(out_size, outer_size, outer_rank)
            in_start, in_end = _shard_range(in_size, inner_size, inner_rank)

        return (0, out_start, out_end, 1, in_start, in_end)
    
    return None


def extract_shard(
    weight: torch.Tensor,
    dim: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Extract shard from full weight along a single dimension."""
    if dim == 0:
        return weight[start:end, ...].contiguous()
    elif dim == 1:
        return weight[:, start:end, ...].contiguous()
    elif dim == 2:
        return weight[:, :, start:end, ...].contiguous()
    else:
        raise ValueError(f"Unsupported shard dim: {dim}")


def extract_shard_2d(
    weight: torch.Tensor,
    dim0: int,
    start0: int,
    end0: int,
    dim1: int,
    start1: int,
    end1: int,
) -> torch.Tensor:
    """Extract shard from full weight along two dimensions (for 2D tensors)."""
    # For 2D weights [out, in], extract [out_start:out_end, in_start:in_end]
    if dim0 == 0 and dim1 == 1:
        return weight[start0:end0, start1:end1].contiguous()
    else:
        raise ValueError(f"Unsupported 2D shard dims: {dim0}, {dim1}")


def compute_local_shard(
    key: str,
    full_weight: Optional[torch.Tensor],
    orig_shape,
    orig_dtype,
    orig_ndim,
    ref_shape,
    ref_dtype,
    ref_ndim,
    shard_info,
    rank: int,
    world_size: int,
):
    """
    Compute and extract the per-rank shard for this tensor.
    
    Strategy:
    1. Rank 0 broadcasts the full tensor to all ranks
    2. All ranks then extract their local shard based on shard_info
    3. If shard_info is None, tensor is replicated across all ranks
    
    Result: produces canonically sharded tensors matching loader expectations.
    """
    # Broadcast the full tensor from rank 0 to all ranks
    if rank == 0 and full_weight is not None:
        weight = full_weight.clone().to("cuda")
    else:
        weight = torch.zeros(orig_shape, dtype=orig_dtype, device="cuda")

    dist.broadcast(weight, src=0)
    
    # Extract local shard if needed
    if shard_info is not None:
        if len(shard_info) == 3:
            # 3D tensor (MoE) or 1D tensor (bias): (dim, start, end)
            dim, start, end = shard_info
            weight = extract_shard(weight, dim, start, end)
        elif len(shard_info) == 6 and weight.ndim == 2:
            # 2D tensor (linear): (dim0, start0, end0, dim1, start1, end1)
            dim0, start0, end0, dim1, start1, end1 = shard_info
            weight = extract_shard_2d(weight, dim0, start0, end0, dim1, start1, end1)
        elif len(shard_info) == 6 and weight.ndim == 1:
            # Fallback: if 6-element shard_info but 1D tensor, just use first 3 elements
            dim, start, end = shard_info[0], shard_info[1], shard_info[2]
            weight = extract_shard(weight, dim, start, end)
    
    return weight.cpu()


def create_tp_checkpoint(
    checkpoint_dir: Path,
    output_dir: Path,
    inner_size: int,
    outer_size: int,
    depth_size: int,
):
    """Main conversion logic."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    inner_rank, outer_rank, depth_rank = compute_rank_coords(
        rank, inner_size, outer_size, depth_size
    )
    
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
    SafePrinter.print(
        f"[Rank {rank}] Creating TP shards for {len(all_keys)} tensors"
    )
    SafePrinter.print(
        f"[Rank {rank}] Mesh coords (inner={inner_rank}, outer={outer_rank}, depth={depth_rank})"
    )
    SafePrinter.print(f"{'='*80}\n")
    
    rank_state_dict = {}
    
    for i, key in enumerate(all_keys):
        if i % max(1, len(all_keys) // 10) == 0 and rank == 0:
            SafePrinter.print(f"[Rank 0] Processing {i+1}/{len(all_keys)} ...")
        
        # Rank 0 has the full weight; prepare shape/dtype info for sharding.
        # For certain 1-D bias tensors we want to compute shard indices
        # based on the corresponding weight tensor (so biases are sliced
        # consistently along the same dimension as their weight).
        if rank == 0:
            full_weight = full_state_dict[key]
            # By default, use the tensor itself as the reference for sharding
            ref_shape = full_weight.shape
            ref_dtype = full_weight.dtype
            ref_ndim = full_weight.ndim

            # For 1D biases (not MoE), try to find a matching weight tensor
            # and use that tensor's shape for computing shard indices so
            # the bias is sliced to match the weight shards.
            # Skip this for MoE biases which are 2D and should use their own shape.
            if key.endswith(".bias") and full_weight.ndim == 1:
                weight_key = key[:-5] + ".weight"
                if weight_key in full_state_dict:
                    ref = full_state_dict[weight_key]
                    ref_shape = ref.shape
                    ref_dtype = ref.dtype
                    ref_ndim = ref.ndim

            weight_shape = ref_shape
            weight_dtype = ref_dtype
            weight_ndim = ref_ndim
        else:
            full_weight = None
            weight_shape = None
            weight_dtype = None
            weight_ndim = None
        
        # Broadcast original tensor shape/dtype info to all ranks
        if rank == 0 and full_weight is not None:
            orig_shape = full_weight.shape
            orig_dtype = full_weight.dtype
            orig_ndim = full_weight.ndim
        else:
            orig_shape = None
            orig_dtype = None
            orig_ndim = None

        orig_shape_list = [orig_shape]
        orig_dtype_list = [orig_dtype]
        orig_ndim_list = [orig_ndim]
        dist.broadcast_object_list(orig_shape_list, src=0)
        dist.broadcast_object_list(orig_dtype_list, src=0)
        dist.broadcast_object_list(orig_ndim_list, src=0)

        orig_shape = orig_shape_list[0]
        orig_dtype = orig_dtype_list[0]
        orig_ndim = orig_ndim_list[0]

        # Broadcast reference shape/dtype info to all ranks (for shard index computation)
        shape_list = [weight_shape]
        dtype_list = [weight_dtype]
        ndim_list = [weight_ndim]
        dist.broadcast_object_list(shape_list, src=0)
        dist.broadcast_object_list(dtype_list, src=0)
        dist.broadcast_object_list(ndim_list, src=0)
        
        ref_shape = shape_list[0]
        ref_dtype = dtype_list[0]
        ref_ndim = ndim_list[0]

        shard_info = get_shard_indices(
            key,
            ref_shape,
            ref_ndim,
            orig_ndim,
            rank,
            inner_rank,
            outer_rank,
            inner_size,
            outer_size,
            world_size,
        )

        # All ranks compute and extract their own shard
        local_shard = compute_local_shard(
            key,
            full_weight,
            orig_shape,
            orig_dtype,
            orig_ndim,
            ref_shape,
            ref_dtype,
            ref_ndim,
            shard_info,
            rank,
            world_size,
        )
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

    # Post-process model_config.yaml inside each rank so that:
    # 1. padded_vocab_size reflects the per-rank padded vocab (not combined)
    # 2. n_head, head_size, and n_query_groups are divided by world_size for TP
    # IMPORTANT: vocab_size must remain the FULL unpadded vocabulary size;
    # only padded_vocab_size should be divided by world_size.
    cfg_path = rank_output_dir / "model_config.yaml"
    if cfg_path.exists():
        try:
            # Read and parse YAML
            import yaml
            with open(cfg_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update padded_vocab_size
            if 'padded_vocab_size' in config:
                combined_padded = config['padded_vocab_size']
                if combined_padded % world_size != 0:
                    SafePrinter.print(f"[Rank {rank}] Warning: combined padded_vocab_size {combined_padded} not divisible by world_size {world_size}")
                config['padded_vocab_size'] = combined_padded // world_size
            
            # Update TP-specific fields
            if 'n_head' in config and config['n_head'] % world_size == 0:
                config['n_head'] = config['n_head'] // world_size
            if 'n_query_groups' in config and config['n_query_groups'] % world_size == 0:
                config['n_query_groups'] = config['n_query_groups'] // world_size
            if 'head_size' in config and 'n_embd' in config:
                # Recompute head_size based on new n_head
                config['head_size'] = config['n_embd'] // config['n_head']
            
            # Write back
            with open(cfg_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            SafePrinter.print(f"[Rank {rank}] Updated model_config.yaml: n_head={config.get('n_head')}, head_size={config.get('head_size')}, n_query_groups={config.get('n_query_groups')}, padded_vocab_size={config.get('padded_vocab_size')}")
        except Exception as e:
            SafePrinter.print(f"[Rank {rank}] Warning: failed to update model_config.yaml: {e}")

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


def parse_args() -> argparse.Namespace:
    default_checkpoint_dir = Path(
        "yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints"
    )
    default_output_dir = Path(
        "yalis/external/checkpoints/openai/gpt-oss-20b/yalis_checkpoints_tp"
    )
    parser = argparse.ArgumentParser(
        description="Stream and shard a full checkpoint into a TP mesh"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=default_checkpoint_dir,
        help="Path to the full (unsharded) checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where per-rank TP shards will be written",
    )
    parser.add_argument(
        "--tp-inner-size",
        type=int,
        default=1,
        help="Number of ranks in the inner (in_features) dimension",
    )
    parser.add_argument(
        "--tp-outer-size",
        type=int,
        default=None,
        help="Number of ranks in the outer (out_features) dimension",
    )
    parser.add_argument(
        "--tp-depth-size",
        type=int,
        default=1,
        help="Number of ranks in the depth (depth parallelism) dimension",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.tp_inner_size <= 0:
        raise ValueError("--tp-inner-size must be > 0")
    if args.tp_depth_size <= 0:
        raise ValueError("--tp-depth-size must be > 0")

    inner_size = args.tp_inner_size
    depth_size = args.tp_depth_size
    outer_size = args.tp_outer_size

    if outer_size is None:
        if inner_size * depth_size == 0:
            raise ValueError("Tensor parallel mesh sizes cannot be zero")
        if env_world_size % (inner_size * depth_size) != 0:
            raise ValueError(
                f"WORLD_SIZE ({env_world_size}) is not divisible by inner*depth ({inner_size * depth_size})"
            )
        outer_size = env_world_size // (inner_size * depth_size)
    if outer_size <= 0:
        raise ValueError("--tp-outer-size must be > 0")

    if inner_size * outer_size * depth_size != env_world_size:
        raise ValueError(
            f"Tensor parallel mesh ({inner_size}x{outer_size}x{depth_size}) "
            f"must equal WORLD_SIZE ({env_world_size})"
        )

    if env_world_size > 1 and not dist.is_initialized():
        init_distributed(tp_dims=(inner_size, outer_size, depth_size))

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if world_size != inner_size * outer_size * depth_size:
        raise RuntimeError(
            f"Distributed world size ({world_size}) does not match configured mesh "
            f"({inner_size}x{outer_size}x{depth_size})"
        )

    SafePrinter.print(f"\n{'='*80}")
    SafePrinter.print(f"TP Checkpoint Conversion")
    SafePrinter.print(f"Input:  {checkpoint_dir}")
    SafePrinter.print(f"Output: {output_dir}")
    SafePrinter.print(
        f"Mesh dims: inner={inner_size}, outer={outer_size}, depth={depth_size}"
    )
    SafePrinter.print(f"World size: {world_size}, Rank: {rank}")
    SafePrinter.print(f"{'='*80}\n")

    create_tp_checkpoint(
        checkpoint_dir,
        output_dir,
        inner_size,
        outer_size,
        depth_size,
    )
    
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
