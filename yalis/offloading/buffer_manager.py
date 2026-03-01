"""
GPU Buffer Manager for CPU offloading.

Manages pre-allocated GPU buffers for efficient CPU->GPU transfers
using .copy_() instead of .to() to avoid allocation overhead.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable
from dataclasses import dataclass, field

from yalis.utils import print_rank0
from .constants import compiler_disable


@dataclass
class ComponentBuffers:
    """Pre-allocated GPU buffers for offloaded parameters."""

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
        should_offload: Callable[[str], bool],
        num_buffer_sets: int = 2,
    ):
        self.device = device
        self.dtype = dtype
        self._should_offload = should_offload
        self.num_buffer_sets = num_buffer_sets

        # Single flat buffer set — all offloaded params keyed by name
        self.buffer_sets: List[ComponentBuffers] = [
            ComponentBuffers() for _ in range(num_buffer_sets)
        ]
        self._allocate_buffers(layer_template)
        self._log_allocation()

    def _allocate_buffers(self, layer: nn.Module):
        """Allocate GPU buffers for all buffer sets."""
        for name, param in layer.named_parameters():
            if not self._should_offload(name):
                continue
            for buf_set in self.buffer_sets:
                print_rank0(
                    f"Allocating buffer for parameter: {name}"
                    f" - {param.dtype} - {param.shape}"
                )
                buf_set.tensors[name] = torch.empty(
                    param.shape,
                    dtype=param.dtype,
                    device=self.device,
                ).contiguous()

        for name, buf in layer.named_buffers():
            if "kv_cache" in name:
                continue
            if not self._should_offload(name):
                continue
            for buf_set in self.buffer_sets:
                buf_set.tensors[name] = torch.empty(
                    buf.shape,
                    dtype=buf.dtype,
                    device=self.device,
                ).contiguous()

    def _log_allocation(self):
        """Log buffer allocation info."""
        print_rank0(
            f"[GPUBufferManager] Allocated"
            f" {self.num_buffer_sets} buffer sets"
        )
        buf = self.buffer_sets[0]
        if buf.tensors:
            size_mb = buf.total_size_bytes() / 1e6
            total_mb = size_mb * self.num_buffer_sets
            n_params = len(buf.tensors)
            print_rank0(
                f"  - {n_params} params: {size_mb:.2f} MB"
                f" x{self.num_buffer_sets} = {total_mb:.2f} MB"
            )

    def get_buffer_set(self, buffer_idx: int) -> ComponentBuffers:
        """Get buffer set by index."""
        return self.buffer_sets[buffer_idx % self.num_buffer_sets]

    def get_buffer_idx_for_layer(self, layer_idx: int) -> int:
        """Get buffer index for a layer."""
        return layer_idx % self.num_buffer_sets

    @compiler_disable()
    def copy_all(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        stream: torch.cuda.Stream,
        buffer_idx: int = 0,
        non_blocking: bool = True,
    ):
        """Copy all offloaded params from CPU to GPU buffers."""
        buf_set = self.get_buffer_set(buffer_idx)

        with torch.cuda.stream(stream):
            self._copy_to_buffer(block, cpu_state, buf_set, non_blocking)

    @compiler_disable()
    def copy_rows(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        row_indices: torch.Tensor,
        stream: torch.cuda.Stream,
        buffer_idx: int = 0,
        non_blocking: bool = True,
    ):
        """Copy specific rows of offloaded params (for sparse/expert)."""
        buf_set = self.get_buffer_set(buffer_idx)

        with torch.cuda.stream(stream):
            for name, gpu_buffer in buf_set.tensors.items():
                if name not in cpu_state:
                    continue

                cpu_tensor = cpu_state[name]

                if cpu_tensor.dim() >= 2 and row_indices is not None:
                    for r in row_indices.tolist():
                        assert r < cpu_tensor.size(0), (
                            f"Row index {r} out of bounds for"
                            f" tensor {name} with size"
                            f" {cpu_tensor.size(0)}"
                        )
                        gpu_buffer[r].copy_(
                            cpu_tensor[r],
                            non_blocking=non_blocking,
                        )
                else:
                    gpu_buffer.copy_(cpu_tensor, non_blocking=non_blocking)

                self._maybe_set_param(block, name, gpu_buffer)

    @compiler_disable()
    def _copy_to_buffer(
        self,
        block: nn.Module,
        cpu_state: Dict[str, torch.Tensor],
        buffers: ComponentBuffers,
        non_blocking: bool,
    ):
        """Copy tensors to GPU buffer."""
        for name, gpu_buffer in buffers.tensors.items():
            if name not in cpu_state:
                continue

            cpu_tensor = cpu_state[name]
            param_data = self._get_param_data(block, name)

            gpu_buffer.copy_(cpu_tensor, non_blocking=non_blocking)

            if (
                param_data is None
                or param_data.data_ptr() != gpu_buffer.data_ptr()
            ):
                self._set_param_data(block, name, gpu_buffer)

    def _get_param_data(
        self, block: nn.Module, name: str
    ) -> Optional[torch.Tensor]:
        """Get parameter data by name."""
        try:
            parts = name.split(".")
            module = block
            for part in parts[:-1]:
                module = getattr(module, part)
            param = getattr(module, parts[-1])
            return param.data if isinstance(param, nn.Parameter) else param
        except AttributeError:
            return None

    @compiler_disable()
    def _set_param_data(
        self, block: nn.Module, name: str, tensor: torch.Tensor
    ):
        """Set parameter data to point to buffer."""
        parts = name.split(".")
        module = block
        for part in parts[:-1]:
            module = getattr(module, part)

        param = getattr(module, parts[-1])
        if isinstance(param, nn.Parameter):
            param.data = tensor
        else:
            setattr(module, parts[-1], tensor)

    def _maybe_set_param(
        self,
        block: nn.Module,
        name: str,
        gpu_buffer: torch.Tensor,
    ):
        """Set param to buffer only if not already pointing there."""
        param_data = self._get_param_data(block, name)
        if (
            param_data is None
            or param_data.data_ptr() != gpu_buffer.data_ptr()
        ):
            self._set_param_data(block, name, gpu_buffer)

    def cleanup(self):
        """Free GPU buffers."""
        for buf_set in self.buffer_sets:
            buf_set.tensors.clear()
        self.buffer_sets.clear()
        torch.cuda.empty_cache()
