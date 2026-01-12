from typing import Sequence, Optional
import torch

from flash_attn import flash_attn_with_kvcache, flash_attn_varlen_func
from flash_attn.ops.triton.rotary import apply_rotary
from yalis.attention.utils.flash_utils import update_paged_kv_cache
from yalis.attention.registry import register_attention
from yalis.constants import EnginePhase


# Custom op for rotary embeddings - needed for torch.compile compatibility
# The underlying flash_attn apply_rotary uses triton kernels that can't be traced
@torch.library.custom_op("yalis::flash_apply_rotary", mutates_args=())
def flash_apply_rotary(
    x: torch.Tensor,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    token_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor
    """
    B = x.size(0)
    return apply_rotary(x, rotary_cos, rotary_sin, token_counter[:B])


@flash_apply_rotary.register_fake
def _(x, rotary_cos=None, rotary_sin=None, token_counter=None):
    # Return tensor with same shape as input
    return torch.empty_like(x)


# Custom op for varlen flash attention (prefill)
@torch.library.custom_op(
    "yalis::torch_compile_compatible_flash_attn_varlen",
    mutates_args=(),
)
def torch_compile_compatible_flash_attn_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: torch.Tensor,
    max_seqlen_k: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    """
    Wrapper for flash_attn_varlen_func that is compatible with torch.compile.
    
    Args:
        q: Query tensor, shape (total_q, nheads, headdim)
        k: Key tensor, shape (total_k, nheads_k, headdim)
        v: Value tensor, shape (total_v, nheads_k, headdim)
        cu_seqlens_q: Cumulative sequence lengths for queries, shape (batch_size + 1,)
        cu_seqlens_k: Cumulative sequence lengths for keys, shape (batch_size + 1,)
        max_seqlen_q: Maximum query sequence length (tensor, converted to int inside op)
        max_seqlen_k: Maximum key sequence length (tensor, converted to int inside op)
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor, shape (total_q, nheads, headdim)
    """
    # Convert tensors to int inside the custom_op (outside torch.compile graph)
    max_q = int(max_seqlen_q.item()) if isinstance(max_seqlen_q, torch.Tensor) else int(max_seqlen_q)
    max_k = int(max_seqlen_k.item()) if isinstance(max_seqlen_k, torch.Tensor) else int(max_seqlen_k)
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        causal=causal,
    )


@torch_compile_compatible_flash_attn_varlen.register_fake
def _(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    causal,
):
    # Return tensor with same shape as q
    return torch.empty_like(q)


# Custom op for kvcache flash attention (decode)
@torch.library.custom_op(
    "yalis::torch_compile_compatible_flash_attention",
    mutates_args=("k_cache", "v_cache"),
)
def torch_compile_compatible_flash_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    cache_seqlens: torch.Tensor,
    rotary_cos: torch.Tensor,
    rotary_sin: torch.Tensor,
    rotary_interleaved: bool,
    window_size: Sequence[int],
    block_table: Optional[torch.Tensor],
) -> torch.Tensor:
    B = q.size(0)
    y = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        causal=causal,
        cache_seqlens=cache_seqlens[:B],
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=rotary_interleaved,
        window_size=window_size,
        block_table=block_table,
    )
    return y


@torch_compile_compatible_flash_attention.register_fake
def _(
    q,
    k_cache,
    v_cache,
    k,
    v,
    causal,
    cache_seqlens,
    rotary_cos,
    rotary_sin,
    rotary_interleaved,
    window_size,
    block_table,
):
    # This is a fake implementation that returns an empty tensor of the same shape as q
    return torch.empty_like(q)


@register_attention("flash")
def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phase: EnginePhase,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    actual_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    **kwargs
) -> torch.Tensor:
    if (
        "use_intra_head_parallelism" in kwargs
        and kwargs["use_intra_head_parallelism"]
    ):
        raise ValueError(
            "flash attention backend does not support intra head parallelism"
        )

    B, T = q.shape[0], q.shape[1]

    # Use varlen for prefill, kvcache for decode
    if phase == EnginePhase.PREFILL:
        return _flash_attention_prefill_varlen(
            q=q,
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            actual_seqlens=actual_seqlens,
            block_table=block_table,
            prestore_kv_cache=prestore_kv_cache,
        )
    else:
        return _flash_attention_decode(
            q=q,
            k=k,
            v=v,
            phase=phase,
            k_cache=k_cache,
            v_cache=v_cache,
            cache_seqlens=cache_seqlens,
            actual_seqlens=actual_seqlens,
            block_table=block_table,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            prestore_kv_cache=prestore_kv_cache,
        )


def _flash_attention_prefill_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    actual_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Prefill using flash_attn_varlen_func for variable-length sequences.
    VECTORIZED: No .item() calls to avoid CPU sync and graph breaks with torch.compile.
    
    Args:
        q, k, v: Shape (B, T, nh, hs) where T is max_seqlen (padded)
        actual_seqlens: Actual sequence lengths for each batch item, shape (B,)
    """
    B, T, n_heads, hs = q.shape
    n_kv_heads = k.shape[2]  # GQA: k/v may have fewer heads than q
    
    if actual_seqlens is None:
        # If no actual_seqlens provided, assume all sequences have length T
        actual_seqlens = torch.full((B,), T, dtype=torch.int32, device=q.device)
    
    # Build cumulative sequence lengths - all on GPU
    seqlens = actual_seqlens.to(torch.int32)
    cu_seqlens_no_zero = torch.cumsum(seqlens, dim=0)  # Contiguous for searchsorted
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
    cu_seqlens[1:] = cu_seqlens_no_zero
    
    # Keep as tensors - no .item() calls
    total_tokens_tensor = cu_seqlens_no_zero[-1]
    max_seqlen_tensor = seqlens.max()
    
    # VECTORIZED flattening: q, k, v from (B, T, nh, hs) to (total_tokens, nh, hs)
    # Create position indices for each slot in the padded tensor
    batch_indices = torch.arange(B, device=q.device).unsqueeze(1).expand(B, T)  # (B, T)
    seq_indices = torch.arange(T, device=q.device).unsqueeze(0).expand(B, T)    # (B, T)
    
    # Create mask for valid positions (within actual sequence length)
    valid_mask = seq_indices < seqlens.unsqueeze(1)  # (B, T)
    
    # Compute flat indices for valid positions
    # flat_idx[b, t] = cu_seqlens[b] + t (for valid positions)
    flat_indices = cu_seqlens[:-1].unsqueeze(1) + seq_indices  # (B, T)
    
    # Gather valid q, k, v into flat tensors
    q_flat = q[valid_mask]  # (total_tokens, n_heads, hs)
    k_flat = k[valid_mask]  # (total_tokens, n_kv_heads, hs)
    v_flat = v[valid_mask]  # (total_tokens, n_kv_heads, hs)
    
    # Store in KV cache if needed - VECTORIZED
    if prestore_kv_cache and k_cache is not None and v_cache is not None:
        if block_table is None:
            # Contiguous cache: use advanced indexing with explicit batch/seq indices
            # k_cache/v_cache shape: (max_batch, max_seq_len, n_kv_heads, hs)
            # Extract the batch and sequence indices for valid positions
            valid_batch_idx = batch_indices[valid_mask]  # (total_tokens,)
            valid_seq_idx = seq_indices[valid_mask]      # (total_tokens,)
            
            # Store using advanced indexing - maps each token to (batch, position) in cache
            k_cache[valid_batch_idx, valid_seq_idx] = k[valid_mask]
            v_cache[valid_batch_idx, valid_seq_idx] = v[valid_mask]
        else:
            # Paged cache
            update_paged_kv_cache(
                k=k,
                v=v,
                block_table=block_table[:B],
                cache_seq_len=cache_seqlens,
                actual_seqlens=actual_seqlens,
                k_cache=k_cache,
                v_cache=v_cache,
            )
    
    # Run varlen flash attention
    # Note: max_seqlen args require int - this happens inside the custom_op (outside compile graph)
    out_flat = torch_compile_compatible_flash_attn_varlen(
        q=q_flat,
        k=k_flat,
        v=v_flat,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen_tensor,
        max_seqlen_k=max_seqlen_tensor,
        causal=True,
    )
    
    # VECTORIZED reshape output back to (B, T, n_heads, hs) with padding
    out = torch.zeros(B, T, n_heads, hs, dtype=out_flat.dtype, device=out_flat.device)
    out[valid_mask] = out_flat
    
    return out


def _flash_attention_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phase: EnginePhase,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    actual_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
) -> torch.Tensor:
    """
    Decode using flash_attn_with_kvcache for incremental generation.
    """
    if block_table is not None and (
        rotary_cos is not None or rotary_sin is not None
    ):
        raise ValueError(
            "flash attention kernel does not support rotary embeddings with a block table"
        )

    B, T = q.shape[0], q.shape[1]
    
    if block_table is not None:
        block_table = block_table[:B]

    if prestore_kv_cache:
        if block_table is None:
            if phase == EnginePhase.DECODE_SINGLE:
                b_indices = torch.arange(B, device=k_cache.device)
                t_indices = cache_seqlens[:B].view(-1)
                k_cache[b_indices, t_indices, :, :] = k[:, 0, :, :]
                v_cache[b_indices, t_indices, :, :] = v[:, 0, :, :]
            elif phase == EnginePhase.DECODE_MULTI:
                nh, hs = k.shape[2], k.shape[3]
                index_kv = cache_seqlens[:B].view(-1, 1) + torch.arange(
                    T, device=cache_seqlens.device
                ).view(1, -1)
                index_kv = index_kv.view(B, T, 1, 1).expand(B, T, nh, hs)
                k_cache[:B].scatter_(dim=1, index=index_kv, src=k)
                v_cache[:B].scatter_(dim=1, index=index_kv, src=v)
        else:
            update_paged_kv_cache(
                k=k,
                v=v,
                block_table=block_table,
                cache_seq_len=cache_seqlens,
                actual_seqlens=actual_seqlens,
                k_cache=k_cache,
                v_cache=v_cache,
            )
        
        # Update cache_seqlens for the attention call
        cache_seqlens = cache_seqlens + T
        k, v = None, None

    return torch_compile_compatible_flash_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        causal=False,  # For decode, we attend to all cached tokens
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=False,
        window_size=(-1, -1),
    )
