"""
CPU Offloaded Model wrapper and helper functions.
"""

import torch
import torch.nn as nn
from typing import List

from .constants import FULL_OFFLOAD, PrefetchMode
from .forward import OffloadedGPTForward


def enable_cpu_offloading(
    model: nn.Module,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    num_prefetch_layers: int = 1,
    pin_memory: bool = True,
    use_preallocated_buffers: bool = False,
    offload_components: List[str] = None,
) -> OffloadedGPTForward:
    """
    Enable CPU offloading for a GPT model.
    
    Args:
        model: GPT model to offload
        device: Target GPU device
        dtype: Model data type
        num_prefetch_layers: Layers to prefetch ahead (default 1)
        pin_memory: Pin CPU memory for faster transfers
        use_preallocated_buffers: Use fixed GPU buffers with .copy_()
        offload_components: Components to offload ["mlp", "attn", "norm"]
        
    Returns:
        OffloadedGPTForward wrapper
        
    Examples:
        # Full layer offload
        offloaded = enable_cpu_offloading(model)
        
        # MLP only (attention stays on GPU)
        offloaded = enable_cpu_offloading(
            model,
            use_preallocated_buffers=True,
            offload_components=["mlp"]
        )
    """
    return OffloadedGPTForward(
        model=model,
        device=device,
        dtype=dtype,
        num_prefetch_layers=num_prefetch_layers,
        pin_memory=pin_memory,
        use_preallocated_buffers=use_preallocated_buffers,
        offload_components=offload_components,
    )


class CPUOffloadedModel(nn.Module):
    """
    Drop-in replacement for GPT that uses CPU offloading.
    
    Wraps the original model and provides the same interface,
    but uses CPU offloading internally for memory efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
        use_preallocated_buffers: bool = False,
        offload_components: List[str] = None,
        prefetch_mode: PrefetchMode = None,  # Legacy
    ):
        super().__init__()
        self.model = model
        self.offloaded_forward = OffloadedGPTForward(
            model=model,
            device=device,
            dtype=dtype,
            num_prefetch_layers=num_prefetch_layers,
            pin_memory=pin_memory,
            use_preallocated_buffers=use_preallocated_buffers,
            offload_components=offload_components,
            prefetch_mode=prefetch_mode,
        )
        
        # Expose model attributes
        self.config = model.config
        self.transformer = model.transformer
        self.lm_head = model.lm_head
    
    def forward(self, *args, **kwargs):
        return self.offloaded_forward(*args, **kwargs)
    
    def __getattr__(self, name):
        if name in ['model', 'offloaded_forward', 'config', 'transformer', 'lm_head']:
            return super().__getattr__(name)
        return getattr(self.model, name)
    
    @property
    def max_seq_length(self):
        return self.model.max_seq_length
    
    @max_seq_length.setter
    def max_seq_length(self, value):
        self.model.max_seq_length = value
    
    def set_kv_cache(self, *args, **kwargs):
        return self.model.set_kv_cache(*args, **kwargs)
    
    def clear_kv_cache(self):
        return self.model.clear_kv_cache()
    
    def rewind_kv_cache(self, *args, **kwargs):
        return self.model.rewind_kv_cache(*args, **kwargs)
    
    def create_symmetric_memory_pool(self, *args, **kwargs):
        return self.model.create_symmetric_memory_pool(*args, **kwargs)
    
    def cleanup(self):
        """Clean up offloading resources."""
        self.offloaded_forward.cleanup()

