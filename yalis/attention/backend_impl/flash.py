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
    seqlen_offsets: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: int = 0,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor.
    
    Args:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
           else (total_seqlen, nheads, headdim)
        rotary_cos: (seqlen_ro, rotary_dim / 2)
        rotary_sin: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer tensor of size (batch,) - cache positions
        cu_seqlens: (batch + 1,) cumulative sequence lengths for varlen
        max_seqlen: max sequence length (Python int, passed directly)
    """
    offsets = seqlen_offsets if seqlen_offsets is not None else 0
    # max_seqlen is already a Python int - no .item() needed
    return apply_rotary(
        x, rotary_cos, rotary_sin,
        seqlen_offsets=offsets,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen if max_seqlen > 0 else None
    )


@flash_apply_rotary.register_fake
def _(x, rotary_cos=None, rotary_sin=None, seqlen_offsets=None, cu_seqlens=None, max_seqlen=0):
    # Return tensor with same shape as input
    return torch.empty_like(x)


# Custom op for updating KV cache with flattened inputs
# Accepts pre-computed batch_idx/seq_idx to avoid recomputation per layer
@torch.library.custom_op("yalis::update_kv_cache_flattened", mutates_args=("k_cache", "v_cache"))
def _update_kv_cache_flattened(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k_flat: torch.Tensor,
    v_flat: torch.Tensor,
    cache_batch_idx: torch.Tensor,
    cache_seq_idx: torch.Tensor,
) -> None:
    """
    Update KV cache from flattened k, v tensors using pre-computed indices.
    
    Args:
        k_cache, v_cache: (max_batch, max_seq_len, n_kv_heads, hs)
        k_flat, v_flat: (total_tokens, n_kv_heads, hs)
        cache_batch_idx: (total_tokens,) batch index for each token
        cache_seq_idx: (total_tokens,) sequence position for each token
    """
    k_cache[cache_batch_idx, cache_seq_idx] = k_flat
    v_cache[cache_batch_idx, cache_seq_idx] = v_flat


@_update_kv_cache_flattened.register_fake
def _(k_cache, v_cache, k_flat, v_flat, cache_batch_idx, cache_seq_idx):
    # This op mutates k_cache and v_cache in place, returns None
    pass


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
    max_seqlen_q: int,
    max_seqlen_k: int,
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
        max_seqlen_q: Maximum query sequence length (Python int)
        max_seqlen_k: Maximum key sequence length (Python int)
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor, shape (total_q, nheads, headdim)
    """
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )


@torch_compile_compatible_flash_attn_varlen.register_fake
def _(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
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
    cu_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    max_seqlen: int = 0,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_seq_idx: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    if (
        "use_intra_head_parallelism" in kwargs
        and kwargs["use_intra_head_parallelism"]
    ):
        raise ValueError(
            "flash attention backend does not support intra head parallelism"
        )

    # Check if input is flattened (3D) or batched (4D)
    is_flattened = q.dim() == 3

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
            cu_seqlens=cu_seqlens,
            block_table=block_table,
            prestore_kv_cache=prestore_kv_cache,
            is_flattened=is_flattened,
            max_seqlen=max_seqlen,
            cache_batch_idx=cache_batch_idx,
            cache_seq_idx=cache_seq_idx,
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
            cu_seqlens=cu_seqlens,
            block_table=block_table,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            prestore_kv_cache=prestore_kv_cache,
            is_flattened=is_flattened,
        )


def _flash_attention_prefill_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    actual_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    is_flattened: bool = False,
    max_seqlen: int = 0,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_seq_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Prefill using flash_attn_varlen_func for variable-length sequences.
    
    Args:
        q, k, v: Shape (total_tokens, nh, hs) if is_flattened else (B, T, nh, hs)
        actual_seqlens: Actual sequence lengths for each batch item, shape (B,)
        cu_seqlens: Cumulative sequence lengths, shape (B+1,) - required if is_flattened
        is_flattened: Whether input is already flattened
        max_seqlen: Maximum sequence length (Python int from config, avoids .item())
        cache_batch_idx: Pre-computed batch indices for KV cache scatter (total_tokens,)
        cache_seq_idx: Pre-computed seq position indices for KV cache scatter (total_tokens,)
    """
    if is_flattened:
        # Input is already flattened (total_tokens, nh, hs)
        assert cu_seqlens is not None, "cu_seqlens required for flattened input"
        q_flat, k_flat, v_flat = q, k, v
        
        # Store in KV cache if needed
        if prestore_kv_cache and k_cache is not None and v_cache is not None:
            if block_table is None:
                # Use pre-computed indices for scatter - no per-layer computation
                _update_kv_cache_flattened(k_cache, v_cache, k_flat, v_flat, cache_batch_idx, cache_seq_idx)
            else:
                # Paged cache - need 4D for update_paged_kv_cache
                # TODO: support flattened paged cache
                raise NotImplementedError("Paged cache with flattened input not yet supported")
        
        # Run varlen flash attention - output is flattened
        # max_seqlen is already a Python int from config
        out_flat = torch_compile_compatible_flash_attn_varlen(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        
        # Return flattened output directly
        return out_flat
    else:
        # Original 4D path
        B, T, n_heads, hs = q.shape
        n_kv_heads = k.shape[2]
        
        if actual_seqlens is None:
            actual_seqlens = torch.full((B,), T, dtype=torch.int32, device=q.device)
        
        seqlens = actual_seqlens.to(torch.int32)
        cu_seqlens_no_zero = torch.cumsum(seqlens, dim=0)
        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=q.device)
        cu_seqlens[1:] = cu_seqlens_no_zero
        
        batch_indices = torch.arange(B, device=q.device).unsqueeze(1).expand(B, T)
        seq_indices = torch.arange(T, device=q.device).unsqueeze(0).expand(B, T)
        valid_mask = seq_indices < seqlens.unsqueeze(1)
        
        q_flat = q[valid_mask]
        k_flat = k[valid_mask]
        v_flat = v[valid_mask]
        
        if prestore_kv_cache and k_cache is not None and v_cache is not None:
            if block_table is None:
                valid_batch_idx = batch_indices[valid_mask]
                valid_seq_idx = seq_indices[valid_mask]
                k_cache[valid_batch_idx, valid_seq_idx] = k[valid_mask]
                v_cache[valid_batch_idx, valid_seq_idx] = v[valid_mask]
            else:
                update_paged_kv_cache(
                    k=k,
                    v=v,
                    block_table=block_table[:B],
                    cache_seq_len=cache_seqlens,
                    actual_seqlens=actual_seqlens,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )
        
        # Use T (padded dimension) as max_seqlen - it's already a Python int
        out_flat = torch_compile_compatible_flash_attn_varlen(
            q=q_flat,
            k=k_flat,
            v=v_flat,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=T,
            max_seqlen_k=T,
            causal=True,
        )
        
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
    cu_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    is_flattened: bool = False,
) -> torch.Tensor:
    """
    Decode using flash_attn_with_kvcache for incremental generation.
    
    Args:
        q, k, v: Shape (total_tokens, nh, hs) if is_flattened else (B, T, nh, hs)
        is_flattened: Whether input is already flattened
    """
    if block_table is not None and (
        rotary_cos is not None or rotary_sin is not None
    ):
        raise ValueError(
            "flash attention kernel does not support rotary embeddings with a block table"
        )

    if is_flattened:
        # Flattened input (total_tokens, nh, hs)
        # For decode, each sequence has T=1 (single token), so total_tokens = B
        # Reshape to (B, 1, nh, hs) for flash_attn_with_kvcache
        B = q.shape[0]  # total_tokens = B for decode (each seq has 1 token)
        T = 1
        n_heads, hs = q.shape[1], q.shape[2]
        n_kv_heads = k.shape[1]
        
        # Reshape for flash_attn_with_kvcache which expects (B, T, nh, hs)
        q = q.unsqueeze(1)  # (B, 1, nh, hs)
        k = k.unsqueeze(1)  # (B, 1, nh_kv, hs)
        v = v.unsqueeze(1)  # (B, 1, nh_kv, hs)
    else:
        B, T = q.shape[0], q.shape[1]
        n_heads, hs = q.shape[2], q.shape[3]
        n_kv_heads = k.shape[2]
    
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
                index_kv = cache_seqlens[:B].view(-1, 1) + torch.arange(
                    T, device=cache_seqlens.device
                ).view(1, -1)
                index_kv = index_kv.view(B, T, 1, 1).expand(B, T, n_kv_heads, hs)
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

    out = torch_compile_compatible_flash_attention(
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
    
    # If input was flattened, return flattened output
    if is_flattened:
        # out is (B, 1, nh, hs), flatten to (B, nh, hs)
        out = out.squeeze(1)
    
    return out
