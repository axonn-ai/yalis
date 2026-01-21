"""
Constants and utilities for CPU offloading.
"""

import torch
from enum import Enum, auto
from typing import List, Set

# Disable torch.compile for param data assignment
# Critical for CPU offloading to work with torch.compile
compiler_disable = getattr(torch.compiler, 'disable', lambda: lambda f: f)

# Valid component names for offloading
VALID_COMPONENTS = {"mlp", "attn", "norm"}

# Full offload = all components
FULL_OFFLOAD = ["mlp", "attn", "norm"]

# No component hierarchy - "mlp" means experts-only for MoE, full MLP for dense
COMPONENT_HIERARCHY = {}


class PrefetchMode(Enum):
    """Prefetch mode - kept for backward compatibility."""
    FULL_LAYER = auto()
    MLP_ONLY = auto()
    ATTENTION_ONLY = auto()
    SELECTIVE = auto()
    INLINE = auto()
    EXPERT_ONLY = auto()

def get_mode(mode_str: str) -> PrefetchMode:
    """Convert string to PrefetchMode enum."""
    mapping = {
        "all": None,
        "rows": None,
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
        PrefetchMode.EXPERT_ONLY: ["mlp"],
    }
    return mapping.get(mode, FULL_OFFLOAD.copy())


def get_component_for_param(name: str) -> str:
    """
    Determine which component a parameter belongs to based on name.
    
    For MoE models:
    - 'mlp' for expert weights (gate_up_proj, proj in experts)
    - Router (gate) is NOT considered part of 'mlp' component (always GPU-resident)
    
    For regular MLP:
    - 'mlp' for all MLP layers
    
    Returns:
    - 'mlp' for MLP/expert layers
    - 'attn' for attention layers
    - 'norm' for normalization layers
    """
    if 'attn' in name:
        return 'attn'
    elif 'mlp' in name:
        # For MoE: check if this is the router (gate) - should NOT be offloaded
        # Router path: mlp.gate.weight (the linear layer for routing)
        # Expert path: mlp.experts.gate_up_proj, mlp.experts.proj
        if 'mlp.gate.weight' in name or 'mlp.gate.bias' in name:
            # This is the router - mark it as 'norm' so it won't match 'mlp' component
            # (router stays on GPU, not offloaded as part of 'mlp')
            return 'router'  # Special component that won't match any offload component
        return 'mlp'
    return 'norm'


def expand_components(components: List[str]) -> List[str]:
    """
    Expand parent components to include sub-components for matching.
    
    No expansion needed anymore - 'mlp' directly means:
    - Experts only for MoE models
    - Full MLP for dense models
    
    This is a no-op but kept for backward compatibility.
    """
    return components


def component_matches(param_component: str, offload_components: List[str]) -> bool:
    """
    Check if a parameter's component matches any of the offload components.
    
    Direct matching only - no hierarchy.
    
    Args:
        param_component: The component name for a parameter (e.g., 'mlp', 'attn')
        offload_components: List of components to offload (e.g., ['mlp', 'attn'])
    
    Returns:
        True if param should be offloaded based on component matching
    """
    return param_component in offload_components


def get_unique_components_for_offload(offload_components: List[str]) -> Set[str]:
    """
    Get the set of unique fine-grained components to offload.
    
    No expansion needed - returns components as-is.
    
    Example:
        ["mlp", "attn"] -> {"mlp", "attn"}
    """
    return set(offload_components)

