#!/usr/bin/env python3
"""
Shard existing yalis_checkpoints to TP format for multi-GPU inference.

Usage:
    torchrun --nproc_per_node=2 examples/shard_checkpoint.py

This script:
1. Loads the unshard checkpoint from yalis_checkpoints/
2. Initializes the model with TP enabled across all ranks
3. Each rank gets its sharded portion of weights automatically
4. Rank 0 saves the sharded state_dict to yalis_checkpoints_tp_sharded/
"""
import os
import sys
import shutil
import torch
import torch.distributed as dist
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from yalis.model import get_model
from yalis.initialize import init_distributed
from yalis.attention.backends import AttentionBackend
from yalis.external.safetensor_saver import incremental_save
from yalis.utils import print_rank0
import warnings
warnings.filterwarnings("ignore")

def main():
    # Configuration
    # Point to the existing yalis_checkpoints directory (already in LitGPT format)
    checkpoint_dir = Path("yalis/external/checkpoints/openai/gpt-oss-20b")
    input_checkpoint = checkpoint_dir / "yalis_checkpoints"
    output_dir = checkpoint_dir / "yalis_checkpoints_tp_sharded"
    
    # Verify input checkpoint exists
    if not input_checkpoint.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_checkpoint}")
    
    # Initialize distributed (2 ranks for 2 GPUs)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    
    if world_size > 1 and not dist.is_initialized():
        init_distributed(tp_dims=(world_size, 1, 1))
    
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    print_rank0(f"\n{'='*80}")
    print_rank0(f"Sharding checkpoint to {world_size} ranks")
    print_rank0(f"Input checkpoint:  {input_checkpoint}")
    print_rank0(f"Output directory:  {output_dir}")
    print_rank0(f"{'='*80}\n")
    
    # Load model with TP ENABLED (this will shard the weights automatically)
    # Use the parent directory so get_model can find model_config.yaml
    yalis_model = get_model(
        str(checkpoint_dir),
        model_dtype=torch.bfloat16,
        attention_backend=AttentionBackend.SDPA,
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
        disable_tp=False,  # Enable TP sharding
    ).to("cuda")
    
    # Save sharded checkpoint from rank 0
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print_rank0(f"[Rank {rank}] Saving sharded checkpoint with incremental_save...")
        
        # Use incremental_save for efficient sharding across multiple safetensors files
        # This will automatically handle the large checkpoint by splitting into shards
        with incremental_save(str(output_dir), max_shard_size_bytes=4*(1024**3)) as saver:
            # Save all model parameters
            total_params = sum(p.numel() for p in yalis_model.parameters())
            saved_params = 0
            for name, param in yalis_model.named_parameters():
                saver.store_early(name, param)
                saved_params += param.numel()
                if saved_params % (100 * 1024 * 1024) < param.numel():  # Log every ~100M params
                    print_rank0(f"  Saved {saved_params / (1024**3):.2f}GB / {total_params / (1024**3):.2f}GB")
            
            # Save all buffers (e.g., learned sinks)
            for name, buf in yalis_model.named_buffers():
                saver.store_early(name, buf)
        
        print_rank0(f"\nCheckpoint saved to {output_dir}")
        print_rank0(f"Total parameters: {total_params / (1024**3):.2f}GB")
        print_rank0(f"Sharded across {world_size} GPU ranks")
        
        # Copy metadata files from input checkpoint
        metadata_files = [
            "model_config.yaml",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "chat_template.jinja",
        ]
        for fname in metadata_files:
            src = input_checkpoint / fname
            if src.exists():
                shutil.copy2(src, output_dir / fname)
                print_rank0(f"  Copied {fname}")
    
    # Synchronize all ranks
    if dist.is_initialized():
        dist.barrier()
    
    print_rank0(f"\n{'='*80}")
    print_rank0("Sharding complete! Ready for TP-enabled inference.")
    print_rank0(f"Update your script to use: {output_dir}")
    print_rank0(f"And set: disable_tp=False")
    print_rank0(f"{'='*80}\n")

if __name__ == "__main__":
    main()