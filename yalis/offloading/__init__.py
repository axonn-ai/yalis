"""
CPU Offloading for YALIS Inference Server.

This module implements CPU offloading for large language models that don't fit
entirely in GPU memory. The strategy is:
1. Keep model weights on CPU
2. Execute one layer at a time on GPU
3. Prefetch next layer(s) while current layer computes
4. Use CUDA streams to overlap transfer and compute

Features:
- Selective component offloading (MLP only, attention only, etc.)
- Pre-allocated GPU buffers for zero-allocation transfers
- Multi-buffer strategy for any prefetch depth
- torch.compile compatible

Usage:
    from yalis.offloading import CPUOffloadedModel, enable_cpu_offloading
    
    # As a model wrapper
    model = CPUOffloadedModel(
        model,
        use_preallocated_buffers=True,
        offload_components=["mlp"],  # Only offload MLP
    )
    
    # Or as a function
    offloaded = enable_cpu_offloading(model)
    output = offloaded(input_ids, phase)
"""

from .constants import (
    VALID_COMPONENTS,
    FULL_OFFLOAD,
    PrefetchMode,
    get_component_for_param,
)

from .buffer_manager import (
    ComponentBuffers,
    GPUBufferManager,
)

from .manager import CPUOffloadManager

from .forward import OffloadedGPTForward

from .model import (
    CPUOffloadedModel,
    enable_cpu_offloading,
)

__all__ = [
    # Constants
    "VALID_COMPONENTS",
    "FULL_OFFLOAD",
    "PrefetchMode",
    "get_component_for_param",
    # Buffer management
    "ComponentBuffers",
    "GPUBufferManager",
    # Core manager
    "CPUOffloadManager",
    # Forward pass
    "OffloadedGPTForward",
    # Model wrapper
    "CPUOffloadedModel",
    "enable_cpu_offloading",
]

