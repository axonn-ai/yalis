import math
from typing import Optional
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention

from axonn import axonn as ax
from axonn.intra_layer.communication import Drop, Gather

from .registry import register_attention
from yalis.constants import EnginePhase


def build_mask_from_index(index: torch.Tensor, t_max: int) -> torch.Tensor:
    # Create a range [0, ..., t_max-1] and reshape to [1, t_max] to broadcast
    arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
    # Compare to index[:, None]: [B, 1] which will broadcast to [B, t_max]
    return arange_t <= index.unsqueeze(1)


def index_into_rope_cache_gen(
    cache: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    # index - [B, T]
    assert index.dim() == 1, "this method is only for the generation phase"
    # Debug: print index range before performing index_select to help
    # diagnose out-of-bounds errors that result in CUDA asserts.
    try:
        idx_min = int(index.min().item())
        idx_max = int(index.max().item())
        cache_len = int(cache.size(0))
        print(f"[sdpa-debug-rope-index] index_min={idx_min}, index_max={idx_max}, cache_len={cache_len}", flush=True, file=sys.stderr)
        if idx_min < 0 or idx_max >= cache_len:
            print(f"[sdpa-debug-rope-index] WARNING: RoPE index out of bounds (will proceed to index_select and may raise).", flush=True, file=sys.stderr)
    except Exception:
        print(f"[sdpa-debug-rope-index] unable to read index min/max (index shape: {tuple(index.shape)})", flush=True, file=sys.stderr)

    return torch.index_select(
        cache,
        0,
        index.view(-1),
    ).reshape(index.size(0), 1, -1)


def create_upper_mask(dim: int, device: torch.device) -> torch.Tensor:
    mask = torch.triu(
        torch.ones(dim, dim, dtype=torch.bool, device=device),
        diagonal=1,
    )
    mask = mask.to(torch.float32)
    mask.masked_fill_(mask.bool(), -float("inf"))
    return mask


def intra_head_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor,
    process_group: dist.ProcessGroup,
    enable_gqa: bool,
    parallel: bool = True,
) -> torch.Tensor:
    mask = (
        create_upper_mask(q.size(2), q.device)
        if attn_mask is None
        else attn_mask
    )
    if enable_gqa:
        B, h, n_q, d = q.shape
        g = k.size(1)
        hpg = h // g
        q = q.view(B, g, hpg, n_q, d)
        B2, g2, n_k, d2 = k.shape
        k = k.view(B2, g2, 1, n_k, d2)
        B3, g3, n_v, d3 = v.shape
        v = v.view(B3, g3, 1, n_v, d3)
    if parallel:
        q = Drop.apply(q, process_group).contiguous()
        # k = Drop.apply(k, process_group).contiguous()
    # keep scale in the same dtype/device as `q` to avoid dtype promotion
    scale = torch.tensor(1.0 / math.sqrt(d), dtype=q.dtype, device=q.device)
    if enable_gqa:
        q = q * scale
        S = (
            torch.einsum("b g h n d, b g o d t -> b g h n t", q, k.mT)
            .clone()
            .contiguous()
        )
    else:
        q = q * scale
        S = (q @ k.mT).clone().contiguous()
    if parallel:
        dist.all_reduce(S, op=dist.ReduceOp.SUM, group=process_group)
    S = S + mask
    A = torch.nn.functional.softmax(
        S,
        dim=-1,
        dtype=torch.float,
    ).to(dtype=q.dtype)
    Out = A @ v
    if enable_gqa:
        Out = Out.view(B, g * hpg, n_q, -1)
    Out = Gather.apply(Out, process_group)
    return Out


def rotary_kv_update_sdpa_gen(
    q: torch.Tensor,  # B,nh,1,hs
    k: torch.Tensor,  # B,nh,1,hs
    v: torch.Tensor,  # B,nh,1,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    use_intra_head_parallelism: bool = False,
    use_flex: bool = False,
    flex_attention_block_mask=None,
    sliding_window: int = 0,
) -> torch.Tensor:
    # Get current batch size
    B = q.size(0)
    if cos is not None and sin is not None:
        cos = index_into_rope_cache_gen(cos, token_counter[:B])
        sin = index_into_rope_cache_gen(sin, token_counter[:B])

        if cos.dim() > 1:
            # batch dimensions must align
            # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
            # we count from back because all of apply_rope does
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        roped_tensors = []
        for x in [q, k]:
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            roped = (x * cos) + (rotated * sin)
            roped = roped.to(dtype=x.dtype)
            roped_tensors.append(roped)

        q, k = roped_tensors

    b_indices = torch.arange(B, device=k_cache.device)
    t_indices = token_counter[:B].view(-1)

    if hasattr(ax.config, 'G_intra_c') and ax.config.G_intra_c > 1 and use_intra_head_parallelism:
        k_cache[b_indices, :, t_indices, :] = Drop.apply(
            k[:, :, 0, :], ax.comm_handle.inner_intra_layer_parallel_group
        ).to(k_cache.dtype)
        v_cache[b_indices, :, t_indices, :] = Drop.apply(
            v[:, :, 0, :], ax.comm_handle.inner_intra_layer_parallel_group
        ).to(v_cache.dtype)
    else:
        k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :].to(k_cache.dtype)
        v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :].to(v_cache.dtype)
    # Build causal mask (allows keys up to the current index)
    # Optionally constrain to a fixed sliding window if `sliding_window > 0`.
    t_max = k_cache.size(-2)
    arange_t = torch.arange(t_max, device=k_cache.device).view(1, -1)
    upper_mask = arange_t <= token_counter[:B].view(-1, 1)
    if sliding_window and sliding_window > 0:
        lower_bound = (token_counter[:B].view(-1, 1) - sliding_window).clamp(min=0)
        lower_mask = arange_t >= lower_bound
        mask = upper_mask & lower_mask
    else:
        mask = upper_mask

    enable_gqa = q.size(1) != k.size(1)

    # For GQA, expand K and V to match Q's head count
    # PyTorch SDPA doesn't natively support GQA, so we repeat K/V heads
    if enable_gqa:
        q_heads = q.size(1)
        kv_heads = k.size(1)
        q_per_kv = q_heads // kv_heads
        # Expand K and V caches by repeating each head q_per_kv times
        k_cache_expanded = k_cache.repeat_interleave(q_per_kv, dim=1)
        v_cache_expanded = v_cache.repeat_interleave(q_per_kv, dim=1)
    else:
        k_cache_expanded = k_cache
        v_cache_expanded = v_cache

    # Pass sliced K V caches for current batch size
    if use_intra_head_parallelism:
        assert (
            not use_flex
        ), "Intra head parallelism is not supported with flex attention"
        mask_float = torch.zeros_like(mask, dtype=torch.float32)
        mask_float = mask_float.masked_fill(~mask, float("-inf"))
        mask_float = mask_float[:, None, None, :]
        mask_float = mask_float.unsqueeze(1)
        out = intra_head_sdpa(
            q,
            k_cache_expanded[:B],
            v_cache_expanded[:B],
            mask_float,
            ax.comm_handle.inner_intra_layer_parallel_group,
            enable_gqa,
            parallel=True,
        )
        return out
    else:
        if use_flex:
            assert (
                flex_attention_block_mask is not None
            ), "flex attention requires a block mask"
            out = flex_attention(
                q,
                k_cache_expanded[:B],
                v_cache_expanded[:B],
                enable_gqa=enable_gqa,
                block_mask=flex_attention_block_mask,
            )
            # Ensure output matches original Q dtype
            return out.to(dtype=q.dtype)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k_cache_expanded[:B],
                v_cache_expanded[:B],
                attn_mask=mask[:, None, None, :],
            )
        # Ensure output matches original Q dtype
        return out.to(dtype=q.dtype)


def rotary_kv_update_sdpa_gen_gptoss(
    q: torch.Tensor,  # B, nh, T, hs (T can be 1 for decode or >1 for prefill)
    k: torch.Tensor,  # B, nh, T, hs
    v: torch.Tensor,  # B, nh, T, hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: int = 0,
    use_intra_head_parallelism: bool = False,
    use_flex: bool = False,
    flex_attention_block_mask=None,
) -> torch.Tensor:
    """
    Implements GPT-OSS-specific semantics: a sliding-window lower-bound
    on the causal mask and an optional per-head "sink" logit column
    appended to the attention logits. This function is intentionally
    kept separate from :func:`rotary_kv_update_sdpa_gen` to avoid
    adding branching inside the optimized, backend-aware path
    (Flash/Flex/intra-head-parallelism) to reduce regression risk.

    Behavior: compute Q@K^T, apply the combined mask, optionally
    append sink logits, softmax, drop the sink column, and apply the
    resulting weights to V.

    Inputs are the per-batch Q and the key/value caches.
    
    Now supports both PREFILL (T > 1) and DECODE (T = 1) phases.
    """
    B = q.size(0)
    _, nh, T, hs = q.shape

    process_group = ax.comm_handle.inner_intra_layer_parallel_group
    if use_intra_head_parallelism:
        assert not use_flex, "GPT-OSS helper does not support flex attention"
        q = Drop.apply(q, process_group).contiguous()
        # Also apply Drop to sinks if provided, since sinks is per-head.
        # Drop semantics vary by tensor rank; explicitly expand sinks to
        # (1, n_head, 1, 1) and drop along dim=1 (head dim) so the
        # resulting sinks matches the reduced per-inner-rank head count.
        if sinks is not None:
            S = sinks.view(1, -1, 1, 1)
            print(f"[sinks-debug] before Drop: S.shape={tuple(S.shape)}, process_group={process_group}", flush=True)
            S_dropped = Drop.apply(S, process_group, 1)
            try:
                print(f"[sinks-debug] after Drop: S.shape={tuple(S_dropped.shape)}", flush=True)
            except Exception:
                print(f"[sinks-debug] after Drop: S has no .shape attribute", flush=True)
            S = S_dropped.contiguous()
            # keep sinks in the expanded 4D form so downstream code can
            # use it directly when concatenating with QK
            sinks = S

    # Apply RoPE to Q and K
    # For PREFILL (T > 1): use cos/sin[:T] directly (positions 0 to T-1)
    # For DECODE (T = 1): use token_counter to index into RoPE cache
    if cos is not None and sin is not None:
        if T == 1:
            # DECODE: single token at position token_counter
            cos = index_into_rope_cache_gen(cos, token_counter[:B])
            sin = index_into_rope_cache_gen(sin, token_counter[:B])
            if cos.dim() > 1:
                cos = cos.unsqueeze(-3)
                sin = sin.unsqueeze(-3)
        else:
            # PREFILL: T tokens starting at position 0
            cos, sin = cos[:T], sin[:T]
            # cos and sin are of shape (T, hs)
            # Add singleton dimensions - (1, 1, T, hs)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        
        # Apply RoPE to both Q and K
        roped_tensors = []
        for x in [q, k]:
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]
            x2 = x[..., head_size // 2 :]
            rotated = torch.cat((-x2, x1), dim=-1)
            roped = ((x * cos) + (rotated * sin)).to(dtype=x.dtype)
            roped_tensors.append(roped)
        
        q, k = roped_tensors

    # Write the newly generated K and V to the cache at the current position
    # For PREFILL (T > 1): write T tokens starting at position 0
    # For DECODE (T = 1): write 1 token at position token_counter
    if T == 1:
        # DECODE: single token
        b_indices = torch.arange(B, device=k_cache.device)
        t_indices = token_counter[:B].view(-1)
        # Extra diagnostics: print token indices and cache shape before writing
        try:
            print(f"[sdpa-debug-write-cache] B={B}, t_indices_min={int(t_indices.min().item())}, t_indices_max={int(t_indices.max().item())}, k_cache_tmax={k_cache.size(-2)}, k_cache_shape={tuple(k_cache.shape)}", flush=True, file=sys.stderr)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[sdpa-debug-write-cache] sync/error: {e}", flush=True, file=sys.stderr)

        if use_intra_head_parallelism:
            k_cache[b_indices, :, t_indices, :] = Drop.apply(
                k[:, :, 0, :], process_group
            ).to(k_cache.dtype)
            v_cache[b_indices, :, t_indices, :] = Drop.apply(
                v[:, :, 0, :], process_group
            ).to(v_cache.dtype)
        else:
            k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :].to(k_cache.dtype)
            v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :].to(v_cache.dtype)
    else:
        # PREFILL: T tokens starting at position 0
        if use_intra_head_parallelism:
            k_cache[:B, :, :T, :] = Drop.apply(
                k, process_group
            ).to(k_cache.dtype)
            v_cache[:B, :, :T, :] = Drop.apply(
                v, process_group
            ).to(v_cache.dtype)
        else:
            k_cache[:B, :, :T, :] = k.to(k_cache.dtype)
            v_cache[:B, :, :T, :] = v.to(v_cache.dtype)

    # Build mask for attention
    # For PREFILL: attend only to the T tokens being prefilled (causal within T)
    # For DECODE: attend to all cached tokens up to token_counter
    
    if T == 1:
        # DECODE: Q is (B, h, 1, d), attend to cache up to token_counter
        # K and V are from cache (B, h, t_max, d)
        Q = q
        K = k_cache[:B]
        V = v_cache[:B]
        
        # Mask: attend to all positions 0..token_counter
        t_max = k_cache.size(-2)
        arange_t = torch.arange(t_max, device=k_cache.device).view(1, -1)
        upper_mask = arange_t <= token_counter[:B].view(-1, 1)
        
        if sliding_window and sliding_window > 0:
            lower_bound = (token_counter[:B].view(-1, 1) - sliding_window).clamp(min=0)
            lower_mask = arange_t >= lower_bound
            mask = upper_mask & lower_mask
        else:
            mask = upper_mask
        # mask shape: (B, t_max)
        n_k = t_max
    else:
        # PREFILL: Q is (B, h, T, d), attend to the T tokens (causal)
        # K and V are the T tokens we just wrote to cache
        Q = q
        K = k_cache[:B, :, :T, :]
        V = v_cache[:B, :, :T, :]
        
        # Causal mask: query i attends to keys 0..i
        # Create lower-triangular mask (T, T)
        causal_mask = torch.tril(
            torch.ones(T, T, dtype=torch.bool, device=q.device)
        )
        
        if sliding_window and sliding_window > 0:
            # Sliding window: query i attends to keys max(0, i-window)..i
            # Create band matrix
            row_indices = torch.arange(T, device=q.device).view(T, 1)
            col_indices = torch.arange(T, device=q.device).view(1, T)
            window_mask = (row_indices - col_indices) <= sliding_window
            mask = causal_mask & window_mask
        else:
            mask = causal_mask
        # mask shape: (T, T) - will broadcast to (B, 1, T, T) later
        n_k = T

    # Handle GQA: Q may have more heads than K/V
    # Q, K, V were set above based on PREFILL vs DECODE
    enable_gqa = Q.size(1) != K.size(1)
    
    # keep scale in the same dtype/device as `Q` to avoid dtype promotion
    scale = torch.tensor(1.0 / math.sqrt(hs), dtype=Q.dtype, device=Q.device)
    
    # Extract dimensions for later use
    B_q, h, n_q, d = Q.shape
    g = None
    hpg = None
    
    if enable_gqa:
        # Reshape for GQA: Q has h heads, K/V have g groups
        # Following OpenAI's convention: expand K/V to match Q's head count
        g = K.size(1)  # number of key/value groups
        hpg = h // g   # heads per group (q_mult in OpenAI's code)
        # Expand K and V by repeating each head hpg times to match Q's head count
        K_expanded = K.repeat_interleave(hpg, dim=1)  # (B, h, t_max, hs)
        V_expanded = V.repeat_interleave(hpg, dim=1)  # (B, h, t_max, hs)
        # Now both Q and K/V have the same number of heads
        Q_scaled = (Q * scale).to(dtype=K_expanded.dtype)
        QK = torch.einsum("b h q d, b h k d -> b h q k", Q_scaled, K_expanded)
    else:
        # Standard MHA path
        # Ensure both operands are in the same dtype before einsum
        Q_scaled = (Q * scale).to(dtype=K.dtype)
        QK = torch.einsum("b h q d, b h k d -> b h q k", Q_scaled, K)

    # Apply mask (broadcast over batch & heads)
    # For DECODE: mask is (B, t_max), QK is (B, h, 1, t_max)
    # For PREFILL: mask is (T, T), QK is (B, h, T, T)
    mask_float = torch.zeros_like(mask, dtype=torch.float32)
    mask_float = mask_float.masked_fill(~mask, float("-inf"))
    
    if T == 1:
        # DECODE: mask shape (B, n_k) -> (B, 1, 1, n_k)
        QK = QK + mask_float.view(B, 1, 1, n_k)
    else:
        # PREFILL: mask shape (T, T) -> (1, 1, T, T)
        QK = QK + mask_float.view(1, 1, T, T)

    # append sinks column if provided
    # sinks is (n_head, 1, 1) and matches the query head count
    # QK is (B, h, n_q, n_k) where h is the number of query heads
    if sinks is not None:
        # sinks shape: (n_head, 1, 1) -> reshape to (1, n_head, 1, 1) for broadcasting
        S = sinks.view(1, -1, 1, 1)
        print(f"[sinks-debug] concat: S.shape={tuple(S.shape)}, h={h}, n_q={n_q}, use_intra={use_intra_head_parallelism}", flush=True)
        # Diagnostic sanity check: if shapes are incompatible, log sizes
        if S.size(1) != h:
            print(f"[sinks-debug] MISMATCH S.shape={tuple(S.shape)}, h={h}, n_q={n_q}", flush=True)
        QK = torch.cat([QK, S.expand(B, h, n_q, 1)], dim=-1)

    if use_intra_head_parallelism:
        dist.all_reduce(QK, op=dist.ReduceOp.SUM, group=process_group)
    W = torch.nn.functional.softmax(QK, dim=-1, dtype=torch.float).to(dtype=QK.dtype)
    # drop sinks column
    if sinks is not None:
        W = W[..., :-1]

    # weighted sum over KV: W (B, nh, 1, t_max) @ V (B, nh, t_max, hs) -> (B, nh, 1, hs)
    if enable_gqa:
        # Both W and V_expanded now have matching head dimensions
        Out = torch.einsum("b h q k, b h k d -> b h q d", W, V_expanded)
    else:
        Out = torch.einsum("b h q k, b h k d -> b h q d", W, V)
    # Ensure output matches original Q dtype (may have been promoted to float during softmax)
    Out = Out.to(dtype=q.dtype)
    if use_intra_head_parallelism:
        Out = Gather.apply(Out, process_group)
    return Out


def rotary_kv_update_sdpa_prefill(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    use_intra_head_parallelism: bool = False,
) -> torch.Tensor:
    T = q.shape[-2]
    if cos is not None and sin is not None:
        cos, sin = cos[:T], sin[:T]
        # cos and sin are of shape (T, hs)
        # we want to add singleton dimensions - (1, 1, T, hs)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        roped_tensors = []
        for x in [q, k]:
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            roped = (x * cos) + (rotated * sin)
            roped = roped.to(dtype=x.dtype)
            roped_tensors.append(roped)

        q, k = roped_tensors

    # Get current batch size
    B = q.size(0)

    # Index K V caches with respect to current batch size
    if hasattr(ax.config, 'G_intra_c') and ax.config.G_intra_c > 1 and use_intra_head_parallelism:
        k_cache[:B, :, :T, :] = Drop.apply(
            k[:B, :, :T, :], ax.comm_handle.inner_intra_layer_parallel_group
        ).to(k_cache.dtype)
        v_cache[:B, :, :T, :] = Drop.apply(
            v[:B, :, :T, :], ax.comm_handle.inner_intra_layer_parallel_group
        ).to(v_cache.dtype)
    else:
        k_cache[:B, :, :T, :] = k[:B, :, :T, :].to(k_cache.dtype)
        v_cache[:B, :, :T, :] = v[:B, :, :T, :].to(v_cache.dtype)

    enable_gqa = q.size(1) != k.size(1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=enable_gqa
    )
    return out.to(dtype=q.dtype)


def rotary_kv_update_sdpa_multi(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
) -> torch.Tensor:
    _, nh, t_max, hs = k_cache.shape

    # Get current batch size
    B = q.size(0)

    T = q.shape[-2]
    index_pos = token_counter[:B].view(B, 1) + torch.arange(
        T, device=token_counter.device
    ).view(1, -1)

    if cos is not None and sin is not None:
        index_rotary = index_pos.view(B, 1, T, 1).expand(B, 1, T, hs)
        cos = cos[None, None, :, :].expand(B, 1, t_max, hs)
        sin = sin[None, None, :, :].expand(B, 1, t_max, hs)
        # Debug: check indices before gather to avoid device-side assert
        idx_min = int(index_pos.min().item())
        idx_max = int(index_pos.max().item())
        print(f"[sdpa-debug-multi] B={B}, nh={nh}, T={T}, t_max={t_max}, index_pos_min={idx_min}, index_pos_max={idx_max}", flush=True, file=sys.stderr)
        print(f"[sdpa-debug-multi] index_pos sample={index_pos.view(-1)[:8].cpu().numpy()}", flush=True, file=sys.stderr)
        try:
            torch.cuda.synchronize()
        except Exception as e:
            print(f"[sdpa-debug-multi] pre-gather sync/error: {e}", flush=True, file=sys.stderr)
        cos = torch.gather(cos, dim=2, index=index_rotary)
        sin = torch.gather(sin, dim=2, index=index_rotary)

        roped_tensors = []
        for x in [q, k]:
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
            x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
            rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
            roped = (x * cos) + (rotated * sin)
            roped = roped.to(dtype=x.dtype)
            roped_tensors.append(roped)

        q, k = roped_tensors

    index_kv = index_pos.view(B, 1, T, 1).expand(B, nh, T, hs)
    # Debug: check scatter indices
    idx_min = int(index_pos.min().item())
    idx_max = int(index_pos.max().item())
    print(f"[sdpa-debug-scatter] index_pos_min={idx_min}, index_pos_max={idx_max}, k_cache_tmax={k_cache.size(-2)}", flush=True, file=sys.stderr)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[sdpa-debug-scatter] pre-scatter sync/error: {e}", flush=True, file=sys.stderr)
    k_cache[:B].scatter_(dim=2, index=index_kv, src=k.to(k_cache.dtype))
    v_cache[:B].scatter_(dim=2, index=index_kv, src=v.to(v_cache.dtype))

    # Create the mask
    arange_t = torch.arange(k_cache.size(-2), device=k_cache.device).view(
        1, 1, 1, -1
    )
    arange_l = token_counter[:B].view(B, 1, 1, 1) + torch.arange(
        T, device=k_cache.device
    ).view(1, 1, -1, 1)
    mask = arange_t <= arange_l

    enable_gqa = q.size(1) != k.size(1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k_cache[:B], v_cache[:B], attn_mask=mask, enable_gqa=enable_gqa
    )

    return out


def sdpa_and_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phase: EnginePhase,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    use_flex: bool = False,
    flex_attention_block_mask=None,
    **kwargs,
) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError(
            "'block_table' or paged kv-caching is not compatible with SDPA attention."  # noqa: E501
        )

    if use_flex:
        assert (
            flex_attention_block_mask is not None
        ), "flex attention requires a block mask"

    if phase == EnginePhase.DECODE_SINGLE:
        sliding_window = kwargs.get("sliding_window", 0)
        sliding_window_mode = kwargs.get("sliding_window_mode", None)
        # opt-in GPT-OSS style SDPA
        if sliding_window_mode == "gpt_oss":
            sinks = kwargs.get("sinks", None)
            y = rotary_kv_update_sdpa_gen_gptoss(
                q,
                k,
                v,
                rotary_cos,
                rotary_sin,
                cache_seqlens,
                k_cache,
                v_cache,
                sinks=sinks,
                sliding_window=sliding_window,
                use_intra_head_parallelism=use_intra_head_parallelism,
                use_flex=use_flex,
                flex_attention_block_mask=flex_attention_block_mask,
            )
        else:
            y = rotary_kv_update_sdpa_gen(
                q,
                k,
                v,
                rotary_cos,
                rotary_sin,
                cache_seqlens,  # B,1
                k_cache,  # B,nh,t_max,hs
                v_cache,  # B,nh,t_max,hs
                use_intra_head_parallelism,
                use_flex,
                flex_attention_block_mask,
                sliding_window=sliding_window,
            )
    elif phase == EnginePhase.DECODE_MULTI:
        y = rotary_kv_update_sdpa_multi(
            q,
            k,
            v,
            rotary_cos,
            rotary_sin,
            cache_seqlens,
            k_cache,
            v_cache,
        )
    else:  # Prefill
        sliding_window = kwargs.get("sliding_window", 0)
        sliding_window_mode = kwargs.get("sliding_window_mode", None)
        # opt-in GPT-OSS style SDPA for prefill
        if sliding_window_mode == "gpt_oss":
            sinks = kwargs.get("sinks", None)
            y = rotary_kv_update_sdpa_gen_gptoss(
                q,
                k,
                v,
                rotary_cos,
                rotary_sin,
                cache_seqlens,
                k_cache,
                v_cache,
                sinks=sinks,
                sliding_window=sliding_window,
                use_intra_head_parallelism=use_intra_head_parallelism,
                use_flex=use_flex,
                flex_attention_block_mask=flex_attention_block_mask,
            )
        else:
            y = rotary_kv_update_sdpa_prefill(
                q,
                k,
                v,
                rotary_cos,
                rotary_sin,
                k_cache,  # B,nh,t_max,hs
                v_cache,  # B,nh,t_max,hs
                use_intra_head_parallelism,
            )
    return y


@register_attention("sdpa")
def sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    **kwargs,
):
    assert "phase" in kwargs, "phase is required for SDPA attention"
    return sdpa_and_flex_attention(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        use_intra_head_parallelism=use_intra_head_parallelism,
        use_flex=False,
        **kwargs,
    )


@register_attention("flex")
def flex_attention_(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    **kwargs,
):
    assert (
        "flex_attention_block_mask" in kwargs
    ), "flex attention requires a block mask"
    return sdpa_and_flex_attention(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        use_intra_head_parallelism=use_intra_head_parallelism,
        use_flex=True,
        **kwargs,
    )
