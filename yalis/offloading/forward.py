"""
Offloaded forward pass implementation for GPT models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List

from yalis.utils import print_rank0
from .constants import FULL_OFFLOAD, PrefetchMode, mode_to_components
from .manager import CPUOffloadManager


class OffloadedGPTForward:
    """
    Wrapper that provides CPU-offloaded forward pass for GPT models.
    
    Executes layers one at a time, prefetching next layers while 
    current layer computes.
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
        self.model = model
        self.device = device
        self.dtype = dtype
        
        # Handle legacy
        if prefetch_mode is not None:
            offload_components = mode_to_components(prefetch_mode)
        offload_components = offload_components or FULL_OFFLOAD.copy()
        
        print_rank0(f"[CPUOffloading] Creating manager with {num_prefetch_layers} prefetch layers")
        print_rank0(f"[CPUOffloading] Offloading: {offload_components}")
        
        self.offload_manager = CPUOffloadManager(
            model=model,
            device=device,
            num_prefetch_layers=num_prefetch_layers,
            pin_memory=pin_memory,
            use_preallocated_buffers=use_preallocated_buffers,
            offload_components=offload_components,
        )
        
        self._original_forward = model.forward
        self.offload_manager.prepare_for_offloading(dtype)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        phase: Any,
        actual_sequence_lengths: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """Offloaded forward pass."""
        model = self.model
        config = model.config
        idx = input_ids
        T = idx.size(1)
        
        if model.max_seq_length < T:
            raise ValueError(f"Sequence length {T} exceeds max {model.max_seq_length}")
        
        # Paged KV cache block table update
        if config.use_paged_kv_caching:
            self._update_block_table(model, input_ids, T)
        
        # Embedding
        x = model.transformer.wte(idx)
        if config.scale_embeddings:
            x = x * torch.tensor(config.n_embd**0.5, dtype=x.dtype)
        
        # Tensor parallel
        if config.tensor_parallel:
            x = self._apply_tensor_parallel_drop(x)
        
        # RoPE cache dtype
        from yalis.attention.backends import AttentionBackend
        if config.attention_backend == AttentionBackend.FLASH:
            model.cos = model.cos.to(x.dtype)
            model.sin = model.sin.to(x.dtype)
        
        block_table = model.kvcache_block_table if config.use_paged_kv_caching else None
        B = x.size(0)
        
        # Flex attention mask
        flex_mask = self._get_flex_mask(model, config, B)
        
        # Process layers with offloading
        for layer_idx, block in enumerate(model.transformer.h):
            with self.offload_manager.layer_context(layer_idx):
                x = block(
                    x, model.cos, model.sin, phase,
                    model.token_counter, block_table, flex_mask
                )
        
        # Final norm and output
        if config.tensor_parallel:
            x = self._apply_tensor_parallel_gather(x)
        
        x = model.transformer.ln_f(x)
        x = model.lm_head(x)
        
        if config.final_logit_softcapping is not None:
            x = torch.tanh(x / config.final_logit_softcapping) * config.final_logit_softcapping
        
        # Update counters
        model.token_counter[:B].add_(T if actual_sequence_lengths is None else actual_sequence_lengths)
        
        if config.use_paged_kv_caching:
            torch.ops.yalis.force_update_tokens_assigned_(model.tokens_assigned[:B], model.token_counter[:B])
        
        return {"logits": x}
    
    def _update_block_table(self, model, input_ids, T):
        """Update paged KV cache block table."""
        B = input_ids.shape[0]
        seq_lengths = torch.full((B,), T, dtype=torch.int64, device=model.kvcache_block_table.device)
        torch.ops.yalis.update_block_table_(
            model.kvcache_block_table[:B],
            model.tokens_assigned[:B],
            model.kvcache_next_page,
            model.kvcache_free_pages,
            seq_lengths,
            256, 16384 // 256,
        )
    
    def _apply_tensor_parallel_drop(self, x):
        """Apply tensor parallel drop."""
        from axonn import axonn as ax
        from axonn.intra_layer.communication import Drop
        return Drop.apply(x, ax.comm_handle.inner_intra_layer_parallel_group)
    
    def _apply_tensor_parallel_gather(self, x):
        """Apply tensor parallel gather."""
        from axonn import axonn as ax
        from axonn.intra_layer.communication import Gather
        return Gather.apply(x, ax.comm_handle.inner_intra_layer_parallel_group)
    
    def _get_flex_mask(self, model, config, B):
        """Get flex attention block mask if needed."""
        from yalis.attention.backends import AttentionBackend
        if config.attention_backend == AttentionBackend.FLEX:
            from yalis.attention.masking import create_causal_block_mask_for_flex_attention
            return create_causal_block_mask_for_flex_attention(model.token_counter, model.kv_length, B)
        return None
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def cleanup(self):
        """Clean up resources."""
        self.offload_manager.cleanup()

