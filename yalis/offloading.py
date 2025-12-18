"""
CPU Offloading Module for YALIS Inference Server

This module implements CPU offloading for large language models that don't fit
entirely in GPU memory. The strategy is:
1. Keep model weights on CPU
2. Have one layer resident on GPU at a time
3. While executing the current layer, prefetch the next layer asynchronously
4. Use CUDA streams to overlap data transfer with computation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from contextlib import contextmanager
import threading
from yalis.utils import print_rank0


class CPUOffloadManager:
    """
    Manages CPU<->GPU transfers for layer-by-layer model execution.
    
    This class handles:
    - Moving layers from CPU to GPU
    - Prefetching the next layer while current layer executes
    - Using separate CUDA streams for compute and data transfer
    - Managing GPU memory by keeping only necessary layers on GPU
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
    ):
        """
        Initialize the CPU Offload Manager.
        
        Args:
            model: The model whose layers will be offloaded
            device: Target GPU device
            num_prefetch_layers: Number of layers to prefetch (default 1)
            pin_memory: Whether to pin CPU memory for faster transfers
        """
        self.model = model
        self.device = device
        self.num_prefetch_layers = num_prefetch_layers
        self.pin_memory = pin_memory
        
        # CUDA streams for overlapping compute and data transfer
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device)
        
        # Track which layers are currently on GPU
        self.layers_on_gpu: set = set()
        
        # Events for synchronization
        self.transfer_events: Dict[int, torch.cuda.Event] = {}
        
        # Get transformer blocks
        self.blocks = self._get_transformer_blocks()
        self.num_layers = len(self.blocks)
        
        # CPU storage for layer parameters
        self.cpu_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Get the transformer blocks from the model."""
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h
        raise ValueError("Model structure not recognized. Expected model.transformer.h")
    
    def prepare_for_offloading(self, dtype: torch.dtype = torch.bfloat16):
        """
        Prepare the model for CPU offloading.
        
        This moves all layers to CPU and optionally pins memory for faster transfers.
        Keeps the embedding and final norm on GPU.
        """
        # Keep embeddings and final layer norm on GPU
        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                self.model.transformer.wte.to(self.device)
            if hasattr(self.model.transformer, 'ln_f'):
                self.model.transformer.ln_f.to(self.device)
        
        if hasattr(self.model, 'lm_head'):
            self.model.lm_head.to(self.device)
        
        # Move rope cache to GPU
        if hasattr(self.model, 'cos'):
            self.model.cos = self.model.cos.to(self.device)
        if hasattr(self.model, 'sin'):
            self.model.sin = self.model.sin.to(self.device)
        
        # Ensure all transformer blocks are on CPU and optionally pin memory
        print_rank0(f"[CPUOffloadManager] Preparing {self.num_layers} blocks for offloading")
        for idx, block in enumerate(self.blocks):
            # Check if block is already on CPU
            first_param = next(block.parameters(), None)
            if first_param is not None and first_param.device.type != 'cpu':
                block.to('cpu')
            
            # Pin memory for faster CPU->GPU transfer
            if self.pin_memory and torch.cuda.is_available():
                print_rank0(f"[CPUOffloadManager] Pinning memory for block {idx}")
                self._pin_layer_memory(block)
            
            # Store reference to CPU state
            self.cpu_state_dicts[idx] = {
                name: param.data for name, param in block.named_parameters()
            }
            for name, buf in block.named_buffers():
                self.cpu_state_dicts[idx][name] = buf
            
            self.layers_on_gpu.discard(idx)
        
        # Move first layer to GPU to be ready
        self._move_layer_to_gpu(0)
        
        print_rank0(f"[CPUOffloadManager] Prepared {self.num_layers} layers for offloading")
        print_rank0(f"[CPUOffloadManager] Embeddings, final norm, and lm_head kept on GPU")
        
    def _pin_layer_memory(self, layer: nn.Module):
        """Pin memory for a layer's parameters for faster CPU->GPU transfer."""
        for param in layer.parameters():
            if param.data.is_contiguous() and not param.data.is_pinned():
                try:
                    param.data = param.data.pin_memory()
                except RuntimeError:
                    # Some tensors might not be pinnable
                    pass
        for buf in layer.buffers():
            if buf.is_contiguous() and not buf.is_pinned():
                try:
                    buf.data = buf.pin_memory()
                except RuntimeError:
                    pass
                    
    def _move_layer_to_gpu(self, layer_idx: int, non_blocking: bool = True):
        """Move a single layer to GPU."""
        if layer_idx in self.layers_on_gpu:
            return
            
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
            
        block = self.blocks[layer_idx]
        block.to(self.device, non_blocking=non_blocking)
        self.layers_on_gpu.add(layer_idx)
        
    def _restore_layer_to_cpu(self, layer_idx: int):
        """
        Restore a layer to CPU by re-assigning the original CPU tensors.
        
        This avoids D2H copy - we just point the parameters back to the
        original CPU tensors that we stored, and let the GPU tensors be freed.
        """
        if layer_idx not in self.layers_on_gpu:
            return
            
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        if layer_idx not in self.cpu_state_dicts:
            # Fallback to regular D2H if we don't have stored CPU tensors
            block = self.blocks[layer_idx]
            block.to('cpu', non_blocking=False)
            self.layers_on_gpu.discard(layer_idx)
            return
            
        block = self.blocks[layer_idx]
        cpu_state = self.cpu_state_dicts[layer_idx]
        
        # Restore parameters from stored CPU tensors
        for name, param in block.named_parameters():
            if name in cpu_state:
                param.data = cpu_state[name]
        
        # Restore buffers from stored CPU tensors
        for name, buf in block.named_buffers():
            if name in cpu_state:
                buf.data = cpu_state[name]
        
        self.layers_on_gpu.discard(layer_idx)
        
    def prefetch_layer(self, layer_idx: int):
        """
        Prefetch a layer to GPU asynchronously using the transfer stream.
        
        Args:
            layer_idx: Index of the layer to prefetch
        """
        if layer_idx in self.layers_on_gpu:
            return
            
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        
        # Record current stream state so transfer waits for current compute
        # This ensures we don't overwrite data being used
        current_stream = torch.cuda.current_stream()
        
        with torch.cuda.stream(self.transfer_stream):
            # Wait for any pending compute before starting transfer
            self.transfer_stream.wait_stream(current_stream)
            self._move_layer_to_gpu(layer_idx, non_blocking=True)
            # Record event for synchronization
            event = torch.cuda.Event()
            event.record(self.transfer_stream)
            self.transfer_events[layer_idx] = event
            
    def wait_for_layer(self, layer_idx: int):
        """
        Make the current CUDA stream wait for a layer's transfer to complete.
        Uses GPU-side synchronization (non-blocking on CPU).
        """
        if layer_idx in self.transfer_events:
            # GPU-side wait: current stream waits for transfer, but CPU continues
            torch.cuda.current_stream().wait_event(self.transfer_events[layer_idx])
            del self.transfer_events[layer_idx]
            
    def offload_layer(self, layer_idx: int):
        """
        Offload a layer back to CPU to free GPU memory.
        
        This restores the original CPU tensors instead of doing D2H copy,
        since the model weights on CPU haven't changed.
        
        Args:
            layer_idx: Index of the layer to offload
        """
        self._restore_layer_to_cpu(layer_idx)
        
    @contextmanager
    def layer_context(self, layer_idx: int):
        """
        Context manager for executing a layer with overlapped prefetching.
        
        This handles:
        1. Ensuring the current layer is on GPU (GPU-side wait, non-blocking on CPU)
        2. Prefetching the next layer(s) on transfer stream
        3. Offloading previous layer(s) after compute completes
        
        The key to overlap:
        - Prefetch of layer N+1 happens on transfer_stream
        - Compute of layer N happens on default stream
        - These can run in parallel on the GPU
        
        Usage:
            with manager.layer_context(i):
                output = layer(input)
        """
        # GPU-side wait for current layer's transfer to complete
        # This does NOT block the CPU - just makes compute stream wait
        self.wait_for_layer(layer_idx)
        
        # Ensure current layer is on GPU (blocking if not prefetched)
        if layer_idx not in self.layers_on_gpu:
            self._move_layer_to_gpu(layer_idx, non_blocking=False)
        
        # Start prefetching next layer(s) - this runs async on transfer_stream
        # The prefetch will overlap with the compute that happens after yield
        for offset in range(1, self.num_prefetch_layers + 1):
            next_idx = layer_idx + offset
            if next_idx < self.num_layers:
                self.prefetch_layer(next_idx)
        
        yield  # Layer computation happens here on default stream
        
        # Offload previous layer (synchronous to ensure correctness)
        prev_idx = layer_idx - 1
        if prev_idx >= 0 and prev_idx in self.layers_on_gpu:
            self.offload_layer(prev_idx)
            
    def cleanup(self):
        """Clean up and move remaining layers back to CPU."""
        for idx in list(self.layers_on_gpu):
            self.offload_layer(idx)
        torch.cuda.empty_cache()
        

class OffloadedGPTForward:
    """
    Wrapper that provides CPU-offloaded forward pass for GPT models.
    
    This replaces the standard forward pass with one that:
    1. Executes layers one at a time
    2. Prefetches next layers while current layer computes
    3. Offloads previous layers to free GPU memory
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
    ):
        """
        Initialize the offloaded forward pass wrapper.
        
        Args:
            model: The GPT model
            device: Target GPU device
            dtype: Model dtype
            num_prefetch_layers: Number of layers to prefetch ahead
            pin_memory: Pin CPU memory for faster transfers
        """
        self.model = model
        self.device = device
        self.dtype = dtype
        
        # Create offload manager
        print(
            f"[CPU Offloading] Creating offload manager with {num_prefetch_layers} "
            f"prefetch layers"
        )
        self.offload_manager = CPUOffloadManager(
            model=model,
            device=device,
            num_prefetch_layers=num_prefetch_layers,
            pin_memory=pin_memory,
        )
        
        # Store original forward
        self._original_forward = model.forward
        
        # Prepare model for offloading
        self.offload_manager.prepare_for_offloading(dtype)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        phase: Any,
        actual_sequence_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Offloaded forward pass that processes layers one at a time.
        
        This mirrors the original GPT forward but with CPU offloading.
        """
        model = self.model
        config = model.config
        idx = input_ids
        T = idx.size(1)
        
        if model.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, "
                f"max seq length is only {model.max_seq_length}."
            )
        
        # Handle paged KV caching block table update
        if config.use_paged_kv_caching:
            B = input_ids.shape[0]
            seq_lengths = torch.full(
                (B,),
                T,
                dtype=torch.int64,
                device=model.kvcache_block_table.device,
            )
            torch.ops.yalis.update_block_table_(
                model.kvcache_block_table[:B],
                model.tokens_assigned[:B],
                model.kvcache_next_page,
                model.kvcache_free_pages,
                seq_lengths,
                256,  # PAGE_BLOCK_SIZE
                16384 // 256,
            )
        
        # Embedding (stays on GPU)
        x = model.transformer.wte(idx)
        if config.scale_embeddings:
            x = x * torch.tensor(config.n_embd**0.5, dtype=x.dtype)
        
        # Handle tensor parallel
        if config.tensor_parallel:
            from axonn import axonn as ax
            from axonn.intra_layer.communication import Drop
            x = Drop.apply(x, ax.comm_handle.inner_intra_layer_parallel_group)
        
        # Ensure rope cache is correct dtype for flash attention
        from yalis.attention.backends import AttentionBackend
        if config.attention_backend == AttentionBackend.FLASH:
            model.cos = model.cos.to(x.dtype)
            model.sin = model.sin.to(x.dtype)
        
        # Block table for paged attention
        block_table = (
            model.kvcache_block_table
            if config.use_paged_kv_caching
            else None
        )
        
        B = x.size(0)
        
        # Flex attention block mask
        from yalis.attention.masking import create_causal_block_mask_for_flex_attention
        flex_attention_block_mask = (
            create_causal_block_mask_for_flex_attention(
                model.token_counter, model.kv_length, B
            )
            if config.attention_backend == AttentionBackend.FLEX
            else None
        )
        
        # Process each layer with offloading
        for layer_idx, block in enumerate(model.transformer.h):
            with self.offload_manager.layer_context(layer_idx):
                x = block(
                    x,
                    model.cos,
                    model.sin,
                    phase,
                    model.token_counter,
                    block_table,
                    flex_attention_block_mask,
                )
        
        # Final norm and lm_head (stay on GPU)
        if config.tensor_parallel:
            from axonn import axonn as ax
            from axonn.intra_layer.communication import Gather
            x = Gather.apply(
                x, ax.comm_handle.inner_intra_layer_parallel_group
            )
        
        x = model.transformer.ln_f(x)
        x = model.lm_head(x)
        
        # Apply final logit softcapping if configured
        if config.final_logit_softcapping is not None:
            x = (
                torch.tanh(x / config.final_logit_softcapping)
                * config.final_logit_softcapping
            )
        
        # Update token counter
        model.token_counter[:B].add_(
            T if actual_sequence_lengths is None else actual_sequence_lengths
        )
        
        # Update paged KV cache token assignments
        if config.use_paged_kv_caching:
            torch.ops.yalis.force_update_tokens_assigned_(
                model.tokens_assigned[:B], model.token_counter[:B]
            )
        
        return {"logits": x}
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def cleanup(self):
        """Clean up resources."""
        self.offload_manager.cleanup()


def enable_cpu_offloading(
    model: nn.Module,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    num_prefetch_layers: int = 1,
    pin_memory: bool = True,
) -> OffloadedGPTForward:
    """
    Enable CPU offloading for a GPT model.
    
    This function:
    1. Moves transformer layers to CPU
    2. Keeps embeddings and output layers on GPU
    3. Returns a wrapper that handles layer-by-layer execution with prefetching
    
    Args:
        model: The GPT model to enable offloading for
        device: Target GPU device
        dtype: Model data type
        num_prefetch_layers: Number of layers to prefetch (default 1)
        pin_memory: Pin CPU memory for faster transfers (default True)
        
    Returns:
        OffloadedGPTForward wrapper that can be used in place of model forward
        
    Example:
        model = get_model(...)
        offloaded_forward = enable_cpu_offloading(model)
        
        # Use offloaded_forward instead of model for inference
        output = offloaded_forward(input_ids, phase)
    """
    return OffloadedGPTForward(
        model=model,
        device=device,
        dtype=dtype,
        num_prefetch_layers=num_prefetch_layers,
        pin_memory=pin_memory,
    )


class CPUOffloadedModel(nn.Module):
    """
    A drop-in replacement for GPT that uses CPU offloading.
    
    This wraps the original model and provides the same interface,
    but uses CPU offloading internally for memory efficiency.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.model = model
        self.offloaded_forward = OffloadedGPTForward(
            model=model,
            device=device,
            dtype=dtype,
            num_prefetch_layers=num_prefetch_layers,
            pin_memory=pin_memory,
        )
        
        # Expose model attributes
        self.config = model.config
        self.transformer = model.transformer
        self.lm_head = model.lm_head
        
    def forward(self, *args, **kwargs):
        return self.offloaded_forward(*args, **kwargs)
    
    # Delegate other methods to the underlying model
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

