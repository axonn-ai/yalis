"""
Constants and utilities for CPU offloading.
"""

import torch
from enum import Enum, auto

# Disable torch.compile for param data assignment
# Critical for CPU offloading to work with torch.compile
compiler_disable = getattr(torch.compiler, "disable", lambda: lambda f: f)


class PrefetchMode(Enum):
    """How to prefetch offloaded params back to GPU."""

    ALL = auto()  # prefetch full offloaded params of next layer(s)
    SELECTIVE = auto()  # prefetch only selected rows/experts
    NONE = auto()  # no prefetch, synchronous fetch when entered


def get_mode(mode_str: str) -> PrefetchMode:
    """Convert config string to PrefetchMode enum."""
    mapping = {
        "all": PrefetchMode.ALL,
        "selective": PrefetchMode.SELECTIVE,
        "none": PrefetchMode.NONE,
    }
    result = mapping.get(mode_str.lower())
    if result is None:
        raise ValueError(
            f"Invalid prefetch mode '{mode_str}'. "
            f"Valid options: {list(mapping.keys())}"
        )
    return result
