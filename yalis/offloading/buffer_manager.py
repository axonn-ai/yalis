"""
GPU Buffer Manager for CPU offloading.

Manages pre-allocated GPU buffers for efficient CPU→GPU transfers
using .copy_() instead of .to() to avoid allocation overhead.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from yalis.utils import print_rank0
from .constants import compiler_disable, get_component_for_param, FULL_OFFLOAD


@dataclass
class ComponentBuffers:
    """Pre-allocated GPU buffers for a component (MLP, Attention, or Norm)."""
    tensors: Dict[str, torch.Tensor] = field(default_factory=dict)
    
    def total_size_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())


class GPUBufferManager:
    """
    Manages pre-allocated GPU buffers with multi-buffer strategy.
    
    Multi-buffer strategy allocates (num_prefetch_layers + 1) buffer sets
    to allow overlapping compute and transfer without conflicts:
    - Layer N uses buffer (N % num_buffer_sets)
    - Adjacent layers always use different buffers
    """
    
    def __init__(
        self,
        layer_template: nn.Module,
        device: torch.device,
        dtype: torch.dtype,
        offload_components: List[str] = None,
        num_buffer_sets: int = 2,
    ):
        self.device = device
        self.dtype = dtype
        self.offload_components = offload_components or FULL_OFFLOAD
        self.num_buffer_sets = num_buffer_sets
        
        # Allocate multiple buffer sets
        self.buffer_sets: List[Dict[str, ComponentBuffers]] = [
            {comp: ComponentBuffers() for comp in self.offload_components}
            for _ in range(num_buffer_sets)
        ]
        
        self._allocate_buffers(layer_template)
        self._log_allocation()
    
    def _allocate_buffers(self, layer: nn.Module):
        """Allocate GPU buffers for all buffer sets."""
        for name, param in layer.named_parameters():
            component = get_component_for_param(name)
            for buffer_set in self.buffer_sets:
                if component in buffer_set:
                    buffer_set[component].tensors[name] = torch.empty(
                        param.shape, dtype=self.dtype, device=self.device
                    )
    
    def _log_allocation(self):
        """Log buffer allocation info."""
        print_rank0(f"[GPUBufferManager] Allocated {self.num_buffer_sets} buffer sets")
        for comp in self.offload_components:
            buf = self.buffer_sets[0].get(comp)
            if buf and buf.tensors:
                size_mb = buf.total_size_bytes() / 1e6
                total_mb = size_mb * self.num_buffer_sets
                print_rank0(f"  - {comp}: {size_mb:.2f} MB x{self.num_buffer_sets} = {total_mb:.2f} MB")
    
    def get_buffer_set(self, buffer_idx: int) -> Dict[str, ComponentBuffers]:
        """Get buffer set by index."""
        return self.buffer_sets[buffer_idx % self.num_buffer_sets]
    
    def get_buffer_idx_for_layer(self, layer_idx: int) -> int:
        """Get buffer index for a layer."""
        return layer_idx % self.num_buffer_sets
    
    @compiler_disable()
    def copy_components(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
        components: List[str] = None,
        buffer_idx: int = 0,
    ):
        """Copy components from CPU to GPU buffers."""
        components = components or self.offload_components
        buffer_set = self.get_buffer_set(buffer_idx)
        
        with torch.cuda.stream(stream):
            for comp in components:
                if comp in buffer_set:
                    self._copy_to_buffer(block, cpu_state, buffer_set[comp])
    
    @compiler_disable()
    def copy_rows(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        row_indices: torch.Tensor,
        stream: torch.cuda.Stream,
        component: str = "mlp",
        buffer_idx: int = 0,
    ):
        """Copy specific rows of a component (for sparse computation)."""
        buffer_set = self.get_buffer_set(buffer_idx)
        
        if component not in buffer_set:
            return
        
        with torch.cuda.stream(stream):
            for name, gpu_buffer in buffer_set[component].tensors.items():
                if name not in cpu_state:
                    continue
                    
                cpu_tensor = cpu_state[name]
                
                # Copy rows for 2D, full copy for 1D
                if cpu_tensor.dim() == 2 and row_indices is not None:
                    gpu_buffer[row_indices].copy_(cpu_tensor[row_indices], non_blocking=True)
                else:
                    gpu_buffer.copy_(cpu_tensor, non_blocking=True)
                
                # Point param to buffer if needed
                self._maybe_set_param(block, name, gpu_buffer)
    
    @compiler_disable()
    def _copy_to_buffer(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        buffers: ComponentBuffers,
    ):
        """Copy tensors to GPU buffer."""
        for name, gpu_buffer in buffers.tensors.items():
            if name not in cpu_state:
                continue
            
            cpu_tensor = cpu_state[name]
            param_data = self._get_param_data(block, name)
            
            # Copy data
            gpu_buffer.copy_(cpu_tensor, non_blocking=True)
            
            # Point param to buffer if not already
            if param_data is None or param_data.data_ptr() != gpu_buffer.data_ptr():
                self._set_param_data(block, name, gpu_buffer)
    
    def _get_param_data(self, block: nn.Module, name: str) -> Optional[torch.Tensor]:
        """Get parameter data by name."""
        try:
            parts = name.split('.')
            module = block
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            return param.data if isinstance(param, nn.Parameter) else param
        except AttributeError:
            return None
    
    @compiler_disable()
    def _set_param_data(self, block: nn.Module, name: str, tensor: torch.Tensor):
        """Set parameter data to point to buffer."""
        parts = name.split('.')
        module = block
        for part in parts[:-1]:
            module = getattr(module, part)
        
        param = getattr(module, parts[-1])
        if isinstance(param, nn.Parameter):
            param.data = tensor
        else:
            setattr(module, parts[-1], tensor)
    
    def _maybe_set_param(self, block: nn.Module, name: str, gpu_buffer: torch.Tensor):
        """Set param to buffer only if not already pointing there."""
        param_data = self._get_param_data(block, name)
        if param_data is None or param_data.data_ptr() != gpu_buffer.data_ptr():
            self._set_param_data(block, name, gpu_buffer)
    
    def cleanup(self):
        """Free GPU buffers."""
        for buffer_set in self.buffer_sets:
            for comp_buffers in buffer_set.values():
                comp_buffers.tensors.clear()
            buffer_set.clear()
        self.buffer_sets.clear()
        torch.cuda.empty_cache()

