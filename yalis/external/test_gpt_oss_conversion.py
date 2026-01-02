#!/usr/bin/env python3
"""Run the GPT-OSS checkpoint conversion into YALIS format."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from convert_hf_checkpoint import convert_hf_checkpoint  # noqa: E402

CHECKPOINT_DIR = REPO_ROOT / "yalis/external/checkpoints/openai/gpt-oss-20b"
OUTPUT_DIR = CHECKPOINT_DIR / "yalis_checkpoints"


def main() -> None:
    if not CHECKPOINT_DIR.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        print("Please download the GPT-OSS 20B checkpoint before running this script.")
        sys.exit(1)

    print(f"Starting GPT-OSS checkpoint conversion from {CHECKPOINT_DIR}")
    print("This process usually takes several minutes on modern hardware.")

    try:
        convert_hf_checkpoint(
            checkpoint_dir=CHECKPOINT_DIR,
            model_name="gpt-oss-20b",
            dtype="float16",
            debug_mode=False,
        )
    except Exception:
        import traceback

        print("Checkpoint conversion failed.")
        traceback.print_exc()
        sys.exit(1)

    print("Checkpoint conversion finished successfully.")
    print(f"Converted checkpoint saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
