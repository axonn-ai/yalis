from .registry import register_attention
import torch
from typing import Optional
from yalis.attention.nowmp_thresh.threshold_attention_nowmp import (
    nowmp_attention_forward,
)


def build_mask_from_index(index, t_max):
    B = index.size(0)
    arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
    return arange_t <= index.unsqueeze(1)


def index_into_rope_cache_gen(cache, index):
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(
        index.size(0), 1, -1
    )


def lit_rotary_kv_update_gen(
    q: torch.Tensor,  # B,nh,1,hs
    k: torch.Tensor,  # B,nh,1,hs
    v: torch.Tensor,  # B,nh,1,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs
    threshold_percentile: float,
    nowmp_state: dict,
    retain_perc: torch.Tensor,
) -> torch.Tensor:
    cos = index_into_rope_cache_gen(cos, token_counter)
    sin = index_into_rope_cache_gen(sin, token_counter)

    if cos.dim() > 1:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)
    roped_tensors = []
    for x in [q, k]:
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]
        x2 = x[..., head_size // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        roped = (x * cos) + (rotated * sin)
        roped = roped.to(dtype=x.dtype)
        roped_tensors.append(roped)

    q, k = roped_tensors

    B = k_cache.size(0)
    b_indices = torch.arange(B, device=k_cache.device)
    t_indices = token_counter.view(-1)

    k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
    v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]
    mask = build_mask_from_index(token_counter, t_max=k_cache.size(-2))

    enable_gqa = q.size(1) != k.size(1)
    out, keep_counts = nowmp_attention_forward(
        q,
        k_cache,
        v_cache,
        token_counter,
        nowmp_state,
        threshold_percentile,
        attn_mask=mask[:, None, None, :],
        enable_gqa=enable_gqa,
    )

    denom = token_counter.to(torch.float32) + 1.0
    retain = keep_counts / denom.view(-1, 1)
    retain = retain.mean(dim=-1, keepdim=True) * 100.0
    retain_perc.add_(retain)
    return out


def lit_rotary_kv_update_prefill(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs
) -> torch.Tensor:
    B, T = q.shape[0], q.shape[-2]
    cos, sin = cos[:T], sin[:T]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    roped_tensors = []
    for x in [q, k]:
        head_size = x.size(-1)
        x1 = x[..., : head_size // 2]
        x2 = x[..., head_size // 2 :]
        rotated = torch.cat((-x2, x1), dim=-1)
        roped = (x * cos) + (rotated * sin)
        roped = roped.to(dtype=x.dtype)
        roped_tensors.append(roped)

    q, k = roped_tensors

    k_cache[:, :, :T, :] = k[:, :, :T, :]
    v_cache[:, :, :T, :] = v[:, :, :T, :]

    enable_gqa = q.size(1) != k.size(1)
    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=enable_gqa
    )
    return out


def threshold_attention_nowmp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    threshold_percentile: float = 0.0,
    nowmp_state: Optional[dict] = None,
    retain_perc: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with Threshold attention.")
    T = q.shape[-2]

    if nowmp_state is None:
        raise ValueError("nowmp attention requires state tensors (alpha, b, counters, lrs)")

    if T == 1:
        y = lit_rotary_kv_update_gen(
            q,
            k,
            v,
            rotary_cos,
            rotary_sin,
            cache_seqlens,
            k_cache,
            v_cache,
            threshold_percentile,
            nowmp_state,
            retain_perc,
        )
    else:
        y = lit_rotary_kv_update_prefill(
            q,
            k,
            v,
            rotary_cos,
            rotary_sin,
            k_cache,
            v_cache,
        )
    return y


@register_attention("thresh_attn_nowmp")
def thresh_attn_nowmp(
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
    assert "threshold_percentile" in kwargs, "nowmp attention requires a threshold percentile"
    assert "retain_perc" in kwargs, "nowmp attention requires a retain percentage"
    assert "nowmp_state" in kwargs, "nowmp attention requires state tensors"
    return threshold_attention_nowmp(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        threshold_percentile=kwargs["threshold_percentile"],
        nowmp_state=kwargs["nowmp_state"],
        retain_perc=kwargs["retain_perc"],
        **kwargs,
    )
