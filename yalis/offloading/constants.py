"""
Constants and utilities for CPU offloading.
"""

import torch
from enum import Enum, auto
from typing import List

# Disable torch.compile for param data assignment
# Critical for CPU offloading to work with torch.compile
compiler_disable = getattr(torch.compiler, 'disable', lambda: lambda f: f)

# Valid component names for offloading
VALID_COMPONENTS = {"mlp", "attn", "norm"}

# Full offload = all components
FULL_OFFLOAD = ["mlp", "attn", "norm"]


class PrefetchMode(Enum):
    """Prefetch mode - kept for backward compatibility."""
    FULL_LAYER = auto()
    MLP_ONLY = auto()
    ATTENTION_ONLY = auto()
    SELECTIVE = auto()
    INLINE = auto()

def get_mode(mode_str: str) -> PrefetchMode:
    """Convert string to PrefetchMode enum."""
    mapping = {
        "all": PrefetchMode.FULL_LAYER,
        "rows": PrefetchMode.MLP_ONLY,
        "inline": PrefetchMode.INLINE,
    }
    return mapping.get(mode_str.lower(), PrefetchMode.FULL_LAYER)

def mode_to_components(mode: PrefetchMode) -> List[str]:
    """Convert legacy PrefetchMode to component list."""
    mapping = {
        PrefetchMode.FULL_LAYER: ["mlp", "attn", "norm"],
        PrefetchMode.MLP_ONLY: ["mlp"],
        PrefetchMode.ATTENTION_ONLY: ["attn"],
        PrefetchMode.SELECTIVE: ["mlp"],
        PrefetchMode.INLINE: ["mlp"],
    }
    return mapping.get(mode, FULL_OFFLOAD.copy())


def get_component_for_param(name: str) -> str:
    """Determine which component a parameter belongs to based on name."""
    if 'attn' in name:
        return 'attn'
    elif 'mlp' in name:
        return 'mlp'
    return 'norm'

