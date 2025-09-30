from flash_attn import flash_attn_with_kvcache
import torch
from typing import Sequence, Optional
from .registry import register_attention
from .update_kv_cache import update_paged_kv_cache
from flash_attn.ops.triton.rotary import apply_rotary
from yalis.constants import EnginePhase


# A recent change (Commit a9a3170) added a wrap_triton call to the rotary
# kernel invocation Torch compile requires this call to be inside a triton_op,
# otherwise compilation breaks. Ideally, this should be fixed in the
# flash attention repo but for now, this workaround works
# TODO: Once flash attention fixes this, remove this
# @torch.library.triton_op("yalis::flash_apply_rotary", mutates_args=())
def flash_apply_rotary(
    x: torch.Tensor,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    token_counter: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies rotary embeddings to the input tensor
    """
    return apply_rotary(x, rotary_cos, rotary_sin, token_counter)


# We are registering the flash_attn_with_kv_cache kernel as a custom pytorch
# op so that it doesn't lead to compile graph breaks more info can be found
# here - https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial # noqa: E501
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
    # NOTE: printing q, k, v shapes
    # print(f"[AKARSH LOGS] q.shape={q.shape[0]}")
    # print(f"[AKARSH LOGS] cache_seqlens.shape={cache_seqlens.shape}")
    # print(f"[AKARSH LOGS] k_cache.shape={k_cache.shape}, v_cache.shape={v_cache.shape}")
    
    y = flash_attn_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        causal=causal,
        cache_seqlens=cache_seqlens,
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
    # This is a fake implementation that returns an empty tensor of the same shape as q # noqa: E501
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
    block_table: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    **kwargs
) -> torch.Tensor:
    if (
        "use_intra_head_parallelism" in kwargs
        and kwargs["use_intra_head_parallelism"]
    ):  # noqa: E501
        raise ValueError(
            "flash attention backend does not support intra head parallelism"
        )
    if block_table is not None and (
        rotary_cos is not None or rotary_sin is not None
    ):  # noqa: E501
        raise ValueError(
            "flash attention kernel does not support rotary embeddings with a block table"  # noqa: E501
        )

    # pre-store in k_cache, v_cache - this should be under a conditional
    B, T = q.shape[0], q.shape[1]
    causal = T > 1
    if prestore_kv_cache:
        if block_table is None:
            if phase == EnginePhase.DECODE_SINGLE:
                b_indices = torch.arange(B, device=k_cache.device)
                t_indices = cache_seqlens.view(-1)
                k_cache[b_indices, t_indices, :, :] = k[:, 0, :, :]
                v_cache[b_indices, t_indices, :, :] = v[:, 0, :, :]
            elif phase == EnginePhase.DECODE_MULTI:
                nh, hs = k.shape[2], k.shape[3]
                index_kv = cache_seqlens.view(-1, 1) + torch.arange(
                    T, device=cache_seqlens.device
                ).view(1, -1)
                index_kv = index_kv.view(B, T, 1, 1).expand(B, T, nh, hs)
                k_cache.scatter_(dim=1, index=index_kv, src=k)
                v_cache.scatter_(dim=1, index=index_kv, src=v)
            else:  # Prefill
                k_cache[:B, :T, :, :] = k[:, :T, :, :]
                v_cache[:B, :T, :, :] = v[:, :T, :, :]
        else:
            update_paged_kv_cache(
                k=k,
                v=v,
                block_table=block_table,
                cache_seq_len=cache_seqlens,
                k_cache=k_cache,
                v_cache=v_cache,
            )
        # since the kv-cache has been updated, we need to update cache_seqlens
        # note: do not update this in-place as the original tensor is needed by
        # subsequent layers to update their kv-caches.
        cache_seqlens = cache_seqlens + T

        k, v = None, None

    return torch_compile_compatible_flash_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k,
        v=v,
        causal=causal,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        rotary_interleaved=False,
        window_size=(-1, -1),
    )
