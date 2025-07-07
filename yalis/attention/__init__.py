import torch
from typing import Optional

from .registry import get_attention
from .backends import AttentionBackend
from .masking import create_block_mask  # noqa: F401


def attention_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    prestore_kv_cache: bool = True,
    backend: AttentionBackend = AttentionBackend.FLASH,
    flex_attention_block_mask=None,
) -> torch.Tensor:
    fn = get_attention(backend.value)
    return fn(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        use_intra_head_parallelism=use_intra_head_parallelism,
        prestore_kv_cache=prestore_kv_cache,
        flex_attention_block_mask=flex_attention_block_mask,
    )
