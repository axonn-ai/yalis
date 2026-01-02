#!/usr/bin/env python3
"""Test GPT-OSS checkpoint conversion."""

import sys
import os

# Add yalis to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yalis', 'external'))

from convert_hf_checkpoint import convert_hf_checkpoint
from pathlib import Path

# Path to downloaded GPT-OSS checkpoint
checkpoint_dir = Path("yalis/external/checkpoints/openai/gpt-oss-20b")

if not checkpoint_dir.exists():
    print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
    print("Please ensure you've downloaded the GPT-OSS 20B checkpoint.")
    sys.exit(1)

print(f"Converting checkpoint from: {checkpoint_dir}")
print("This may take a few minutes...")

# Run conversion
try:
    convert_hf_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model_name="gpt-oss-20b",
        dtype="float16",
        debug_mode=False
    )
    print("\n✓ Conversion completed successfully!")
    print(f"Converted checkpoint saved to: {checkpoint_dir / 'yalis_checkpoints'}")
except Exception as e:
    print(f"\n✗ Conversion failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
