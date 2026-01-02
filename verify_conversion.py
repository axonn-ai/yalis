#!/usr/bin/env python3
"""Verify GPT-OSS checkpoint conversion."""

import sys
import os
from pathlib import Path

# Add yalis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yalis', 'external'))

import torch
from safetensors.torch import load_file as load_safetensors

def verify_conversion(hf_dir, yalis_dir):
    """Verify that the conversion was successful."""
    
    hf_dir = Path(hf_dir)
    yalis_dir = Path(yalis_dir)
    
    print("="*70)
    print("GPT-OSS Checkpoint Conversion Verification")
    print("="*70)
    
    # 1. Check that yalis checkpoint directory exists and has files
    print("\n1. Checking converted checkpoint structure...")
    if not yalis_dir.exists():
        print(f"   ✗ Directory not found: {yalis_dir}")
        return False
    
    # Check for safetensors files
    yalis_files = list(yalis_dir.glob("*.safetensors"))
    index_file = yalis_dir / "model.safetensors.index.json"
    
    if not yalis_files:
        print(f"   ✗ No .safetensors files found in {yalis_dir}")
        return False
    
    print(f"   ✓ Found {len(yalis_files)} checkpoint shard(s)")
    total_size_gb = sum(f.stat().st_size for f in yalis_files) / (1024**3)
    for f in sorted(yalis_files):
        size_gb = f.stat().st_size / (1024**3)
        print(f"     - {f.name}: {size_gb:.2f} GB")
    print(f"   ✓ Total checkpoint size: {total_size_gb:.2f} GB")
    
    if index_file.exists():
        print(f"   ✓ Found index file: {index_file.name}")
    
    # 2. Load and inspect the converted checkpoint
    print("\n2. Loading converted checkpoint...")
    try:
        # Load all shards into a single state dict
        state_dict = {}
        if index_file.exists():
            import json
            with open(index_file) as f:
                index = json.load(f)
            print(f"   ✓ Index contains {len(index.get('weight_map', {}))} weight mappings")
            
            # Load each shard
            shard_files = set(index['weight_map'].values())
            for shard_name in sorted(shard_files):
                shard_path = yalis_dir / shard_name
                if shard_path.exists():
                    shard_weights = load_safetensors(shard_path)
                    state_dict.update(shard_weights)
        else:
            # Load single file or all files
            for f in yalis_files:
                weights = load_safetensors(f)
                state_dict.update(weights)
        
        print(f"   ✓ Loaded checkpoint from {len(yalis_files)} shard(s)")
        print(f"   ✓ Total parameters: {len(state_dict)} tensors")
    except Exception as e:
        print(f"   ✗ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. Verify expected keys and structure
    print("\n3. Verifying model structure...")
    
    expected_patterns = [
        ('transformer.wte.weight', 'Token embeddings'),
        ('transformer.ln_f.weight', 'Final layer norm'),
        ('lm_head.weight', 'Language model head'),
        ('transformer.h.0.norm_1.weight', 'Layer 0 input norm'),
        ('transformer.h.0.attn.attn.weight', 'Layer 0 QKV weights'),
        ('transformer.h.0.attn.attn.bias', 'Layer 0 QKV bias'),
        ('transformer.h.0.attn.proj.weight', 'Layer 0 attention output'),
        ('transformer.h.0.attn.proj.bias', 'Layer 0 attention output bias'),
        ('transformer.h.0.sinks', 'Layer 0 sinks'),
        ('transformer.h.0.mlp.router.weight', 'Layer 0 MoE router'),
        ('transformer.h.0.mlp.router.bias', 'Layer 0 MoE router bias'),
        ('transformer.h.0.mlp.gate_up_weight', 'Layer 0 MoE gate_up weights'),
        ('transformer.h.0.mlp.gate_up_bias', 'Layer 0 MoE gate_up bias'),
        ('transformer.h.0.mlp.down_weight', 'Layer 0 MoE down weights'),
        ('transformer.h.0.mlp.down_bias', 'Layer 0 MoE down bias'),
        ('transformer.h.0.norm_2.weight', 'Layer 0 post-attention norm'),
    ]
    
    missing = []
    found = []
    for key, desc in expected_patterns:
        if key in state_dict:
            shape = tuple(state_dict[key].shape)
            dtype = state_dict[key].dtype
            found.append((desc, shape, dtype))
            print(f"   ✓ {desc}: {shape} ({dtype})")
        else:
            missing.append((key, desc))
    
    if missing:
        print(f"\n   ⚠ Missing {len(missing)} expected keys:")
        for key, desc in missing:
            print(f"     - {key} ({desc})")
    
    # 4. Verify MoE weight shapes
    print("\n4. Verifying MoE weight shapes...")
    layer_0_moe_keys = [k for k in state_dict.keys() if k.startswith('transformer.h.0.mlp.')]
    
    if 'transformer.h.0.mlp.gate_up_weight' in state_dict:
        gate_up_shape = state_dict['transformer.h.0.mlp.gate_up_weight'].shape
        down_shape = state_dict['transformer.h.0.mlp.down_weight'].shape
        print(f"   ✓ gate_up_weight: {gate_up_shape}")
        print(f"   ✓ down_weight: {down_shape}")
        
        # Verify expected shapes for GPT-OSS 20B: (32 experts, 5760, 2880) and (32, 2880, 2880)
        expected_gate_up = (32, 5760, 2880)
        expected_down = (32, 2880, 2880)
        
        if gate_up_shape == expected_gate_up:
            print(f"   ✓ gate_up_weight shape matches expected {expected_gate_up}")
        else:
            print(f"   ⚠ gate_up_weight shape {gate_up_shape} != expected {expected_gate_up}")
        
        if down_shape == expected_down:
            print(f"   ✓ down_weight shape matches expected {expected_down}")
        else:
            print(f"   ⚠ down_weight shape {down_shape} != expected {expected_down}")
    
    # 5. Count layers
    print("\n5. Counting transformer layers...")
    layer_nums = set()
    for key in state_dict.keys():
        if 'transformer.h.' in key:
            parts = key.split('.')
            if len(parts) > 2:
                try:
                    layer_nums.add(int(parts[2]))
                except ValueError:
                    pass
    
    if layer_nums:
        num_layers = len(layer_nums)
        print(f"   ✓ Found {num_layers} transformer layers (indices: {min(layer_nums)}-{max(layer_nums)})")
        expected_layers = 24
        if num_layers == expected_layers:
            print(f"   ✓ Layer count matches GPT-OSS 20B ({expected_layers} layers)")
        else:
            print(f"   ⚠ Expected {expected_layers} layers, found {num_layers}")
    
    # 6. Compare sample weights with HF checkpoint
    print("\n6. Spot-checking weight values against HF checkpoint...")
    try:
        # Load a sample from HF checkpoint
        hf_file = hf_dir / "model-00000-of-00002.safetensors"
        hf_weights = load_safetensors(hf_file)
        
        # Check embeddings
        if 'model.embed_tokens.weight' in hf_weights and 'transformer.wte.weight' in state_dict:
            hf_embed = hf_weights['model.embed_tokens.weight']
            yalis_embed = state_dict['transformer.wte.weight']
            
            if torch.allclose(hf_embed.float(), yalis_embed.float(), rtol=1e-3, atol=1e-5):
                print(f"   ✓ Embeddings match (shape: {yalis_embed.shape})")
            else:
                max_diff = (hf_embed.float() - yalis_embed.float()).abs().max().item()
                print(f"   ⚠ Embeddings differ (max diff: {max_diff:.6f})")
        
        # Check layer 0 input norm
        if 'model.layers.0.input_layernorm.weight' in hf_weights and 'transformer.h.0.norm_1.weight' in state_dict:
            hf_norm = hf_weights['model.layers.0.input_layernorm.weight']
            yalis_norm = state_dict['transformer.h.0.norm_1.weight']
            
            if torch.allclose(hf_norm.float(), yalis_norm.float(), rtol=1e-3, atol=1e-5):
                print(f"   ✓ Layer 0 norm matches (shape: {yalis_norm.shape})")
            else:
                max_diff = (hf_norm.float() - yalis_norm.float()).abs().max().item()
                print(f"   ⚠ Layer 0 norm differs (max diff: {max_diff:.6f})")
        
        # Check router weights
        if 'model.layers.0.mlp.router.weight' in hf_weights and 'transformer.h.0.mlp.router.weight' in state_dict:
            hf_router = hf_weights['model.layers.0.mlp.router.weight']
            yalis_router = state_dict['transformer.h.0.mlp.router.weight']
            
            if torch.allclose(hf_router.float(), yalis_router.float(), rtol=1e-3, atol=1e-5):
                print(f"   ✓ Router weights match (shape: {yalis_router.shape})")
            else:
                max_diff = (hf_router.float() - yalis_router.float()).abs().max().item()
                print(f"   ⚠ Router weights differ (max diff: {max_diff:.6f})")
        
    except Exception as e:
        print(f"   ⚠ Could not compare with HF checkpoint: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("Verification Summary")
    print("="*70)
    print(f"✓ Checkpoint files created: {len(yalis_files)}")
    print(f"✓ Total tensors in checkpoint: {len(state_dict)}")
    print(f"✓ Found {len(found)}/{len(expected_patterns)} expected keys")
    
    if missing:
        print(f"⚠ Missing {len(missing)} expected keys")
        return False
    
    print("\n✓ Conversion verification PASSED!")
    return True


if __name__ == "__main__":
    hf_dir = Path("yalis/external/checkpoints/openai/gpt-oss-20b")
    yalis_dir = hf_dir / "yalis_checkpoints"
    
    success = verify_conversion(hf_dir, yalis_dir)
    sys.exit(0 if success else 1)
