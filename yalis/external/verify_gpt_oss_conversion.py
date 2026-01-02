#!/usr/bin/env python3
"""Validate the GPT-OSS checkpoint conversion artifacts."""

import json
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
from safetensors.torch import load_file as load_safetensors

SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    """Locate the repository root by searching for repository markers.

    This helper searches upward for files typically present at the project
    root. If none are found it returns a conservative fallback parent.
    """
    cur = start
    markers = ("setup.py", "LICENSE", "README.md", ".git")
    for _ in range(10):
        for m in markers:
            if (cur / m).exists():
                return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.parents[1] if len(start.parents) > 1 else start


REPO_ROOT = find_repo_root(SCRIPT_DIR)
HF_CHECKPOINT_DIR = REPO_ROOT / "yalis/external/checkpoints/openai/gpt-oss-20b"
YALIS_CHECKPOINT_DIR = HF_CHECKPOINT_DIR / "yalis_checkpoints"

EXPECTED_KEYS = [
    "transformer.wte.weight",
    "transformer.ln_f.weight",
    "lm_head.weight",
    "transformer.h.0.norm_1.weight",
    "transformer.h.0.attn.attn.weight",
    "transformer.h.0.attn.attn.bias",
    "transformer.h.0.attn.proj.weight",
    "transformer.h.0.attn.proj.bias",
    "transformer.h.0.sinks",
    "transformer.h.0.mlp.router.weight",
    "transformer.h.0.mlp.router.bias",
    "transformer.h.0.mlp.gate_up_weight",
    "transformer.h.0.mlp.gate_up_bias",
    "transformer.h.0.mlp.down_weight",
    "transformer.h.0.mlp.down_bias",
    "transformer.h.0.norm_2.weight",
]

EXPECTED_GATE_UP_SHAPE = (32, 5760, 2880)
EXPECTED_DOWN_SHAPE = (32, 2880, 2880)
EXPECTED_LAYER_COUNT = 24


def _print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def _gather_shard_paths(checkpoint_dir: Path) -> Tuple[Sequence[Path], int]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as index_file:
            index_data = json.load(index_file)
        weight_map = index_data.get("weight_map", {})
        shard_names = sorted(set(weight_map.values()))
        shard_paths = [checkpoint_dir / name for name in shard_names]
        return shard_paths, len(weight_map)

    shard_paths = sorted(checkpoint_dir.glob("*.safetensors"))
    return shard_paths, len(shard_paths)


def _load_state_dict(shard_paths: Iterable[Path]) -> dict:
    state_dict: dict = {}
    for shard_path in shard_paths:
        shard_weights = load_safetensors(shard_path)
        state_dict.update(shard_weights)
    return state_dict


def _compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> Tuple[bool, float]:
    reference_value = reference.float()
    candidate_value = candidate.float()
    max_difference = (reference_value - candidate_value).abs().max().item()
    tolerance = 1e-3 * max(reference_value.abs().max().item(), 1.0) + 1e-5
    return max_difference <= tolerance, max_difference


def verify_conversion() -> bool:
    _print_section("Section 1: Directory layout")
    if not YALIS_CHECKPOINT_DIR.exists():
        print(f"Converted checkpoint directory not found at {YALIS_CHECKPOINT_DIR}")
        return False

    shard_paths, _ = _gather_shard_paths(YALIS_CHECKPOINT_DIR)
    if not shard_paths:
        print("No checkpoint shards were located in the target directory.")
        return False

    total_size = sum(path.stat().st_size for path in shard_paths) / (1024**3)
    print(f"Located {len(shard_paths)} shard files with a combined size of {total_size:.2f} GB")

    _print_section("Section 2: Loading checkpoint data")
    try:
        state_dict = _load_state_dict(shard_paths)
    except Exception as error:
        print(f"Unable to load shard tensors: {error}")
        return False

    print(f"Loaded {len(state_dict)} tensors from the converted checkpoint")

    _print_section("Section 3: Key coverage")
    missing_keys = [key for key in EXPECTED_KEYS if key not in state_dict]
    for key in EXPECTED_KEYS:
        if key in state_dict:
            tensor = state_dict[key]
            print(f"Present: {key} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    if missing_keys:
        print(f"Missing {len(missing_keys)} expected tensors:")
        for key in missing_keys:
            print(f"  - {key}")
        return False

    _print_section("Section 4: MoE weight shapes")
    gate_up = state_dict.get("transformer.h.0.mlp.gate_up_weight")
    down = state_dict.get("transformer.h.0.mlp.down_weight")
    if gate_up is None or down is None:
        print("MoE gate_up or down weight tensors are missing.")
        return False

    print(f"Gate up weight shape: {tuple(gate_up.shape)}")
    print(f"Down weight shape: {tuple(down.shape)}")
    if tuple(gate_up.shape) != EXPECTED_GATE_UP_SHAPE or tuple(down.shape) != EXPECTED_DOWN_SHAPE:
        print("MoE weight shapes do not match the GPT-OSS 20B expectations.")
        return False

    _print_section("Section 5: Layer enumeration")
    layer_indices = sorted({int(part)
                            for key in state_dict
                            if key.startswith("transformer.h.")
                            for part in key.split(".")
                            if part.isdigit()})
    if not layer_indices:
        print("No transformer layers were detected in the converted checkpoint.")
        return False

    distinct_layers = len(set(layer_indices))
    if distinct_layers > 6:
        print(f"Layer indices detected: {layer_indices[:3]} ... {layer_indices[-3:]}")
    else:
        print(f"Layer indices detected: {layer_indices}")

    if distinct_layers != EXPECTED_LAYER_COUNT:
        print(f"Layer count mismatch: expected {EXPECTED_LAYER_COUNT}, found {distinct_layers}")
        return False

    _print_section("Section 6: Spot checks against the HF checkpoint")
    try:
        hf_shard = HF_CHECKPOINT_DIR / "model-00000-of-00002.safetensors"
        hf_state = load_safetensors(hf_shard)
    except Exception as error:
        print(f"Unable to load reference HF shard: {error}")
        return False

    comparisons = [
        ("model.embed_tokens.weight", "transformer.wte.weight"),
        ("model.layers.0.input_layernorm.weight", "transformer.h.0.norm_1.weight"),
        ("model.layers.0.mlp.router.weight", "transformer.h.0.mlp.router.weight"),
    ]

    for hf_key, yalis_key in comparisons:
        if hf_key not in hf_state or yalis_key not in state_dict:
            print(f"Skipping comparison for {yalis_key} because the matching HF tensor is missing.")
            continue
        matches, max_diff = _compare_tensors(hf_state[hf_key], state_dict[yalis_key])
        status = "matches" if matches else "diverges from"
        print(f"{yalis_key} {status} the HF tensor (max difference {max_diff:.6f})")
        if not matches:
            return False

    print("All spot checks are within the defined tolerances.")
    return True


if __name__ == "__main__":
    success = verify_conversion()
    sys.exit(0 if success else 1)
