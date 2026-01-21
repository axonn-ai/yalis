"""
Constants and utilities for CPU offloading.
"""

import torch
from enum import Enum, auto
from typing import List, Set

# Disable torch.compile for param data assignment
# Critical for CPU offloading to work with torch.compile
compiler_disable = getattr(torch.compiler, 'disable', lambda: lambda f: f)

# Valid component names for offloading (including MoE sub-components)
VALID_COMPONENTS = {"mlp", "attn", "norm", "mlp.experts", "mlp.gate"}

# Full offload = all components
FULL_OFFLOAD = ["mlp", "attn", "norm"]

# Component hierarchy: parent -> children
COMPONENT_HIERARCHY = {
    "mlp": ["mlp.gate", "mlp.experts"],
}


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
        "rows": PrefetchMode.EXPERT_ONLY,
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
    
    Returns the most specific component for MoE models:
    - 'mlp.experts' for expert weights (gate_up_proj, proj in TPMoE)
    - 'mlp.gate' for the router (gate linear layer in LLaMAMoE)
    - 'mlp' for regular dense MLP layers
    - 'attn' for attention layers
    - 'norm' for normalization layers
    """
    if 'attn' in name:
        return 'attn'
    elif 'mlp' in name:
        # Check for MoE sub-components (more specific patterns first)
        if 'experts' in name:
            return 'mlp.experts'
        # 'gate' alone is the router, but 'gate_up_proj' is part of experts
        elif 'gate' in name and 'gate_up_proj' not in name:
            return 'mlp.gate'
        return 'mlp'
    return 'norm'


def expand_components(components: List[str]) -> List[str]:
    """
    Expand parent components to include sub-components for matching.
    
    When "mlp" is specified, parameters with 'mlp.gate' or 'mlp.experts' 
    should also be matched. This maintains backward compatibility.
    
    Example:
        ["mlp", "attn"] -> ["mlp", "mlp.gate", "mlp.experts", "attn"]
    """
    expanded = set(components)
    for comp in components:
        if comp in COMPONENT_HIERARCHY:
            expanded.update(COMPONENT_HIERARCHY[comp])
    return list(expanded)


def component_matches(param_component: str, offload_components: List[str]) -> bool:
    """
    Check if a parameter's component matches any of the offload components.
    
    Handles hierarchical matching:
    - Direct match: 'mlp.experts' matches ['mlp.experts']
    - Parent match: 'mlp.experts' matches ['mlp'] (parent includes children)
    
    Args:
        param_component: The component name for a parameter (e.g., 'mlp.experts')
        offload_components: List of components to offload (e.g., ['mlp', 'attn'])
    
    Returns:
        True if param should be offloaded based on component matching
    """
    # Direct match
    if param_component in offload_components:
        return True
    
    # Check if parent is in offload_components
    # e.g., 'mlp.experts' -> parent is 'mlp'
    if '.' in param_component:
        parent = param_component.rsplit('.', 1)[0]
        if parent in offload_components:
            return True
    
    return False


def get_unique_components_for_offload(offload_components: List[str]) -> Set[str]:
    """
    Get the set of unique fine-grained components to offload.
    
    This expands parent components and returns only leaf components
    to avoid double-counting in buffer allocation.
    
    Example:
        ["mlp", "attn"] -> {"mlp.gate", "mlp.experts", "attn"}
        ["mlp.experts", "attn"] -> {"mlp.experts", "attn"}
    """
    result = set()
    for comp in offload_components:
        if comp in COMPONENT_HIERARCHY:
            # Parent component: add children
            result.update(COMPONENT_HIERARCHY[comp])
        else:
            result.add(comp)
    return result

