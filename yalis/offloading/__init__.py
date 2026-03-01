"""
CPU Offloading for YALIS Inference Server.

This module implements CPU offloading for large language models that don't fit
entirely in GPU memory. The strategy is:
1. Keep model weights on CPU
2. Execute one layer at a time on GPU
3. Prefetch next layer(s) while current layer computes
4. Use CUDA streams to overlap transfer and compute

Configuration axes:
- **What to offload** (cpu_offload_modules): module paths relative to each
  block. None = offload everything. e.g. ["mlp.experts"] for experts only.
- **How to prefetch** (cpu_offload_prefetch_mode): "all", "selective", "none".

Usage:
    from yalis.offloading import CPUOffloadManager, get_mode

    offload_manager = CPUOffloadManager(
        model,
        use_preallocated_buffers=True,
        offload_modules=["mlp.experts"],
        prefetch_mode=get_mode("all"),
    )
    offload_manager.prepare_for_offloading(dtype)
    model.offload_manager = offload_manager
"""

from .constants import PrefetchMode, get_mode

from .buffer_manager import ComponentBuffers, GPUBufferManager

from .manager import CPUOffloadManager

__all__ = [
    "PrefetchMode",
    "get_mode",
    "ComponentBuffers",
    "GPUBufferManager",
    "CPUOffloadManager",
]
