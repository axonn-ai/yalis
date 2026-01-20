"""SpargeAttn backend wrapper (kernel lives in attention/sparge_attn)."""

from typing import Optional

import torch

from .registry import register_attention
from .sparge_attn.sparge_attention import sparge_attention_forward


def _apply_rope(q, k, cos, sin):
    if cos is not None and sin is not None:
        if cos.dim() > 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        roped = []
        for x in (q, k):
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]
            x2 = x[..., head_size // 2 :]
            rotated = torch.cat((-x2, x1), dim=-1)
            roped.append((x * cos + rotated * sin).to(dtype=x.dtype))
        q, k = roped
    return q, k


def _index_into_rope_cache(cache, index):
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(index.size(0), 1, -1)


def sparge_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    **kwargs,
) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with SpargeAttn.")
    if not prestore_kv_cache:
        raise ValueError("sparge attention requires prestore_kv_cache=True")
    if k_cache is None or v_cache is None:
        raise ValueError("sparge attention requires k_cache and v_cache")

    T = q.shape[-2]
    simthreshd1 = kwargs.get("sparge_simthreshd1", 0.3)
    cdfthreshd = kwargs.get("sparge_cdfthreshd", 0.96)
    pvthreshd = kwargs.get("sparge_pvthreshd", 20)
    smooth_k = kwargs.get("sparge_smooth_k", True)
    attention_sink = kwargs.get("sparge_attention_sink", False)

    if T == 1:
        if cache_seqlens is None:
            raise ValueError("sparge attention requires cache_seqlens for decode")
        cos = _index_into_rope_cache(rotary_cos, cache_seqlens) if rotary_cos is not None else None
        sin = _index_into_rope_cache(rotary_sin, cache_seqlens) if rotary_sin is not None else None
        q, k = _apply_rope(q, k, cos, sin)

        B = k_cache.size(0)
        b_indices = torch.arange(B, device=k_cache.device)
        t_indices = cache_seqlens.view(-1)
        k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
        v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]

        kv_len = int(cache_seqlens.max().item()) + 1
        k_full = k_cache[:, :, :kv_len, :]
        v_full = v_cache[:, :, :kv_len, :]
        return sparge_attention_forward(
            q=q,
            k=k_full,
            v=v_full,
            is_causal=False,
            smooth_k=smooth_k,
            simthreshd1=simthreshd1,
            cdfthreshd=cdfthreshd,
            pvthreshd=pvthreshd,
            attention_sink=attention_sink,
        )

    if rotary_cos is not None and rotary_sin is not None:
        cos = rotary_cos[:T][None, None, :, :]
        sin = rotary_sin[:T][None, None, :, :]
        q, k = _apply_rope(q, k, cos, sin)

    k_cache[:, :, :T, :] = k[:, :, :T, :]
    v_cache[:, :, :T, :] = v[:, :, :T, :]

    return sparge_attention_forward(
        q=q,
        k=k,
        v=v,
        is_causal=True,
        smooth_k=smooth_k,
        simthreshd1=simthreshd1,
        cdfthreshd=cdfthreshd,
        pvthreshd=pvthreshd,
        attention_sink=attention_sink,
    )


@register_attention("sparge")
def sparge_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    prestore_kv_cache: bool = True,
    **kwargs,
):
    if use_intra_head_parallelism:
        raise ValueError("sparge attention does not support intra head parallelism")
    return sparge_attention(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        prestore_kv_cache=prestore_kv_cache,
        **kwargs,
    )
