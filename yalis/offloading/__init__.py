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
    # In engine.py (done automatically when use_cpu_offloading=True):
    from yalis.offloading import CPUOffloadManager
    
    offload_manager = CPUOffloadManager(
        model,
        use_preallocated_buffers=True,
        offload_components=["mlp"],  # Only offload MLP
    )
    offload_manager.prepare_for_offloading(dtype)
    model.offload_manager = offload_manager
    
    # The model's forward() will automatically use offloading via layer_context()
"""

from .constants import (
    VALID_COMPONENTS,
    FULL_OFFLOAD,
    PrefetchMode,
    get_component_for_param,
    get_mode,
)

from .buffer_manager import (
    ComponentBuffers,
    GPUBufferManager,
)

from .manager import CPUOffloadManager

__all__ = [
    # Constants
    "VALID_COMPONENTS",
    "FULL_OFFLOAD",
    "PrefetchMode",
    "get_component_for_param",
    "get_mode",
    # Buffer management
    "ComponentBuffers",
    "GPUBufferManager",
    # Core manager
    "CPUOffloadManager",
]
