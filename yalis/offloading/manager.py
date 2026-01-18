"""
CPU Offload Manager - coordinates layer-by-layer execution with prefetching.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set, Callable
from contextlib import contextmanager

from yalis.utils import print_rank0
from .constants import (
    compiler_disable, get_component_for_param,
    VALID_COMPONENTS, FULL_OFFLOAD, PrefetchMode, mode_to_components
)
from .buffer_manager import GPUBufferManager


class CPUOffloadManager:
    """
    Manages CPU<->GPU transfers for layer-by-layer model execution.
    
    Features:
    - Prefetches next layers while current layer executes
    - Uses separate CUDA streams for compute and transfer
    - Supports selective component offloading (MLP only, attention only, etc.)
    - Optional pre-allocated GPU buffers for zero-allocation transfers
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
        use_preallocated_buffers: bool = False,
        offload_components: List[str] = None,
        prefetch_mode: PrefetchMode = None,  # Legacy support
    ):
        self.model = model
        self.device = device
        self.num_prefetch_layers = num_prefetch_layers
        self.pin_memory = pin_memory
        self.use_preallocated_buffers = use_preallocated_buffers
        self.prefetch_mode = prefetch_mode
        
        # Handle legacy prefetch_mode
        if prefetch_mode is not None:
            self.offload_components = mode_to_components(prefetch_mode)
        else:
            self.offload_components = offload_components or FULL_OFFLOAD.copy()
        
        # Validate components
        for comp in self.offload_components:
            if comp not in VALID_COMPONENTS:
                raise ValueError(f"Invalid component '{comp}'. Valid: {VALID_COMPONENTS}")
        
        # CUDA streams
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device)
        
        # State tracking
        self.layers_on_gpu: Set[int] = set()
        self.components_on_gpu: Dict[int, Set[str]] = {}
        self.transfer_events: Dict[int, torch.cuda.Event] = {}
        
        # Model structure
        self.blocks = self._get_transformer_blocks()
        self.num_layers = len(self.blocks)
        
        # CPU storage
        self.cpu_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
        self.layer_buffer_idx: Dict[int, int] = {}
        
        # GPU buffer manager
        self.gpu_buffer_manager: Optional[GPUBufferManager] = None
        
        # Callback for sparse row indices
        self.row_indices_callback: Optional[Callable[[int], Optional[torch.Tensor]]] = None
        
    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Get transformer blocks from model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise ValueError("Expected model.transformer.h")
    
    def prepare_for_offloading(self, dtype: torch.dtype = torch.bfloat16):
        """Prepare model for CPU offloading."""
        self.dtype = dtype
        
        print_rank0(f"[CPUOffloadManager] Offloading: {self.offload_components}")
        resident = [c for c in VALID_COMPONENTS if c not in self.offload_components]
        if resident:
            print_rank0(f"[CPUOffloadManager] GPU resident: {resident}")
        
        # Keep embeddings/output on GPU
        self._move_fixed_components_to_gpu()
        
        # Initialize buffer manager if using preallocated buffers
        if self.use_preallocated_buffers:
            num_buffer_sets = self.num_prefetch_layers + 1
            print_rank0(f"[CPUOffloadManager] Creating {num_buffer_sets} buffer sets")
            self.gpu_buffer_manager = GPUBufferManager(
                layer_template=self.blocks[0],
                device=self.device,
                dtype=dtype,
                offload_components=self.offload_components,
                num_buffer_sets=num_buffer_sets,
            )
        
        # Process each block
        for idx, block in enumerate(self.blocks):
            self._prepare_block(idx, block)
        
        # Load first layer
        self._move_components_to_gpu(0)
        
        print_rank0(f"[CPUOffloadManager] Prepared {self.num_layers} layers")
    
    def _move_fixed_components_to_gpu(self):
        """Move embeddings, final norm, lm_head to GPU."""
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                self.model.transformer.wte.to(self.device)
            if hasattr(self.model.transformer, 'ln_f'):
                self.model.transformer.ln_f.to(self.device)
        
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.to(self.device)
        
        # RoPE cache
        if hasattr(self.model, 'cos'):
            self.model.cos = self.model.cos.to(self.device)
        if hasattr(self.model, 'sin'):
            self.model.sin = self.model.sin.to(self.device)
    
    def _prepare_block(self, idx: int, block: nn.Module):
        """Prepare a single block for offloading."""
        self.cpu_state_dicts[idx] = {}
        self.components_on_gpu[idx] = set()
        
        # Process parameters
        for name, param in block.named_parameters():
            component = get_component_for_param(name)
            
            if component in self.offload_components:
                self._offload_param(idx, name, param, component)
            else:
                self._keep_param_on_gpu(idx, param, component)
        
        # Process buffers (KV cache must stay on GPU)
        for name, buf in block.named_buffers():
            if 'kv_cache' in name:
                if buf.device.type != 'cuda':
                    buf.data = buf.data.to(self.device)
                continue
            
            component = get_component_for_param(name)
            if component in self.offload_components:
                self._offload_buffer(idx, name, buf, component)
            elif buf.device.type != 'cuda':
                buf.data = buf.data.to(self.device)
        
        self.layers_on_gpu.discard(idx)
    
    def _offload_param(self, idx: int, name: str, param: nn.Parameter, component: str):
        """Move param to CPU (or store CPU copy if using preallocated buffers)."""
        if self.use_preallocated_buffers:
            # Store CPU copy, point param to GPU buffer
            cpu_tensor = self._to_cpu_pinned(param.data)
            self.cpu_state_dicts[idx][name] = cpu_tensor
            
            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(idx)
            buffer_set = self.gpu_buffer_manager.get_buffer_set(buffer_idx)
            
            if component in buffer_set and name in buffer_set[component].tensors:
                param.data = buffer_set[component].tensors[name]
            
            self.layer_buffer_idx[idx] = buffer_idx
        else:
            # Move param to CPU
            if param.device.type != 'cpu':
                param.data = param.data.to('cpu')
            param.data = self._pin_if_enabled(param.data)
            self.cpu_state_dicts[idx][name] = param.data
    
    def _offload_buffer(self, idx: int, name: str, buf: torch.Tensor, component: str):
        """Move buffer to CPU (or store CPU copy if using preallocated buffers)."""
        if self.use_preallocated_buffers:
            cpu_tensor = buf.data.to('cpu')
            self.cpu_state_dicts[idx][name] = cpu_tensor
            
            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(idx)
            buffer_set = self.gpu_buffer_manager.get_buffer_set(buffer_idx)
            
            if component in buffer_set and name in buffer_set[component].tensors:
                buf.data = buffer_set[component].tensors[name]
        else:
            if buf.device.type != 'cpu':
                buf.data = buf.data.to('cpu')
            self.cpu_state_dicts[idx][name] = buf
    
    def _keep_param_on_gpu(self, idx: int, param: nn.Parameter, component: str):
        """Keep param on GPU."""
        if param.device.type != 'cuda':
            param.data = param.data.to(self.device)
        self.components_on_gpu[idx].add(component)
    
    def _to_cpu_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU with optional pinning."""
        cpu_tensor = tensor.to('cpu')
        if self.pin_memory:
            pinned = cpu_tensor.pin_memory()
            del cpu_tensor
            return pinned
        return cpu_tensor
        #return self._pin_if_enabled(cpu_tensor)
    
    def _pin_if_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin tensor memory if enabled."""
        if self.pin_memory and torch.cuda.is_available():
            if tensor.is_contiguous() and not tensor.is_pinned():
                try:
                    return tensor.pin_memory()
                except RuntimeError:
                    pass
        return tensor
    
    @compiler_disable()
    def _move_components_to_gpu(
        self, 
        layer_idx: int, 
        components: List[str] = None,
        non_blocking: bool = True
    ):
        """Move components to GPU."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        components = components or self.offload_components
        block = self.blocks[layer_idx]
        cpu_state = self.cpu_state_dicts.get(layer_idx, {})
        
        if self.use_preallocated_buffers and self.gpu_buffer_manager:
            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(layer_idx)
            self.gpu_buffer_manager.copy_components(
                block, cpu_state, self.transfer_stream, components, buffer_idx
            )
            if not non_blocking:
                self.transfer_stream.synchronize()
        else:
            # Direct .to() transfer
            for name, param in block.named_parameters():
                comp = get_component_for_param(name)
                if comp in components and name in cpu_state:
                    param.data = cpu_state[name].to(self.device, non_blocking=non_blocking)
            
            for name, buf in block.named_buffers():
                comp = get_component_for_param(name)
                if comp in components and name in cpu_state:
                    buf.data = cpu_state[name].to(self.device, non_blocking=non_blocking)
        
        # Update tracking
        for comp in components:
            self.components_on_gpu[layer_idx].add(comp)
        
        if all(c in self.components_on_gpu[layer_idx] for c in self.offload_components):
            self.layers_on_gpu.add(layer_idx)
    
    @compiler_disable()
    def _move_rows_to_gpu(
        self, 
        layer_idx: int, 
        row_indices: torch.Tensor,
        component: str = "mlp",
        non_blocking: bool = True,
        stream = None,
    ):
        """Move specific rows to GPU (requires preallocated buffers)."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        if stream == None:
            stream = self.transfer_stream
        
        if not self.use_preallocated_buffers or not self.gpu_buffer_manager:
            return self._move_components_to_gpu(layer_idx, [component], non_blocking)
        
        buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(layer_idx)
        self.gpu_buffer_manager.copy_rows(
            self.blocks[layer_idx],
            self.cpu_state_dicts[layer_idx],
            row_indices,
            stream,
            component,
            buffer_idx
        )
        
        if not non_blocking:
          if stream is None:
              self.transfer_stream.synchronize()
          else:
              stream.synchronize()

        
        self.components_on_gpu[layer_idx].add(f'{component}')
    
    @compiler_disable()
    def _restore_layer_to_cpu(self, layer_idx: int):
        """Restore layer to CPU (or just update tracking for preallocated buffers)."""
        if layer_idx not in self.layers_on_gpu:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        cpu_state = self.cpu_state_dicts.get(layer_idx, {})
        if not cpu_state:
            self.layers_on_gpu.discard(layer_idx)
            return
        
        # With preallocated buffers, don't reassign (torch.compile compatibility)
        if not self.use_preallocated_buffers:
            block = self.blocks[layer_idx]
            for name, param in block.named_parameters():
                if name in cpu_state:
                    param.data = cpu_state[name]
            for name, buf in block.named_buffers():
                if name in cpu_state:
                    buf.data = cpu_state[name]
        
        # Update tracking
        for comp in self.offload_components:
            self.components_on_gpu[layer_idx].discard(comp)
            self.components_on_gpu[layer_idx].discard(f'{comp}_partial')
        self.layers_on_gpu.discard(layer_idx)

    def fetch_layer(
        self, 
        layer_idx: int, 
        components: List[str] = None,
        row_indices: torch.Tensor = None,
        non_blocking: bool = False,
    ):
        """Fetch layer synchronously."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        components = components or self.offload_components
        current_stream = torch.cuda.current_stream()
        
        with torch.cuda.stream(current_stream):
            if row_indices is not None:
                for comp in components:
                    #print_rank0(f"[CPUOffloadManager] Fetching layer {layer_idx} rows {row_indices}")
                    self._move_rows_to_gpu(layer_idx, row_indices, comp, non_blocking=False, stream=current_stream)
            else:
                self._move_components_to_gpu(layer_idx, components, non_blocking=False, stream=current_stream)
        
        
    def prefetch_layer(
        self, 
        layer_idx: int, 
        components: List[str] = None,
        row_indices: torch.Tensor = None,
    ):
        """Prefetch layer asynchronously."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self.layers_on_gpu:
            return
        
        components = components or self.offload_components
        current_stream = torch.cuda.current_stream()
        
        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_stream(current_stream)
            
            if row_indices is not None:
                for comp in components:
                    self._move_rows_to_gpu(layer_idx, row_indices, comp, non_blocking=True)
            else:
                self._move_components_to_gpu(layer_idx, components, non_blocking=True)
            
            event = torch.cuda.Event()
            event.record(self.transfer_stream)
            self.transfer_events[layer_idx] = event
    
    def wait_for_layer(self, layer_idx: int):
        """Wait for layer transfer to complete (GPU-side, non-blocking on CPU)."""
        if layer_idx in self.transfer_events:
            torch.cuda.current_stream().wait_event(self.transfer_events[layer_idx])
            self.layers_on_gpu.add(layer_idx)
            del self.transfer_events[layer_idx]
    
    def offload_layer(self, layer_idx: int):
        """Offload layer back to CPU."""
        self._restore_layer_to_cpu(layer_idx)
    
    def set_row_indices_callback(self, callback: Callable[[int], Optional[torch.Tensor]]):
        """Set callback for sparse row prefetch."""
        self.row_indices_callback = callback

    def is_inline_mode(self) -> bool:
        """Check if in inline mode."""
        return self.prefetch_mode == PrefetchMode.INLINE
    
    @contextmanager
    def layer_context(self, layer_idx: int):
        """
        Context manager for layer execution with overlapped prefetching.
        
        Usage:
            with manager.layer_context(i):
                output = layer(input)
        """
        if self.prefetch_mode == PrefetchMode.INLINE:
            yield
            return

        if layer_idx == 0:
            self.transfer_stream.wait_stream(torch.cuda.current_stream())

        # Wait for current layer
        self.wait_for_layer(layer_idx)
        
        if layer_idx not in self.layers_on_gpu:
            self._move_components_to_gpu(layer_idx, non_blocking=False)
        
        # Start prefetching next layers
        for offset in range(1, self.num_prefetch_layers + 1):
            next_idx = layer_idx + offset
            if next_idx < self.num_layers:
                row_indices = None
                if self.row_indices_callback:
                    row_indices = self.row_indices_callback(next_idx)
                self.prefetch_layer(next_idx, row_indices=row_indices)
        
        yield
        
        # Offload previous layer
        prev_idx = layer_idx - 1
        if prev_idx >= 0 and prev_idx in self.layers_on_gpu:
            self.offload_layer(prev_idx)
    
    def cleanup(self):
        """Clean up resources."""
        for idx in list(self.layers_on_gpu):
            self.offload_layer(idx)
        
        if self.gpu_buffer_manager:
            self.gpu_buffer_manager.cleanup()
            self.gpu_buffer_manager = None
        
        torch.cuda.empty_cache()
    
    def reset(self):
        """Reset the manager."""
        assert len(self.transfer_events) == 0
        self.layers_on_gpu = set()

