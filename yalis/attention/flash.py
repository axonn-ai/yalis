from flash_attn import flash_attn_with_kvcache
import torch
from typing import Sequence, Optional, Union
from .registry import register_attention
from .update_kv_cache import update_paged_kv_cache


# here we are registering the flash_attn_with_kv_cache kernel as a custom pytorch op 
# so that it doesn't lead to torch compile graph breaks
# more info can be found here - https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial
@torch.library.custom_op("yalis::torch_compile_compatible_flash_attention", mutates_args=("k_cache", "v_cache"))
def torch_compile_compatible_flash_attention(q: torch.Tensor, 
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
                            block_table: Optional[torch.Tensor]) -> torch.Tensor:
    y = flash_attn_with_kvcache(q=q,
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
                block_table=block_table
            )
    return y

@torch_compile_compatible_flash_attention.register_fake
def _(q, k_cache, v_cache, k, v, causal, cache_seqlens, rotary_cos, rotary_sin, rotary_interleaved, window_size, block_table):
    # This is a fake implementation that returns an empty tensor of the same shape as q
    return torch.empty_like(q)

@register_attention("flash")
def flash_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,
              block_table: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              prestore_kv_cache: bool = True,
              **kwargs) -> torch.Tensor:
    if "use_intra_head_parallelism" in kwargs and kwargs["use_intra_head_parallelism"]:
        raise ValueError("flash attention backend does not support intra head parallelism")
    if block_table is not None and (rotary_cos is not None or rotary_sin is not None):
        raise ValueError("flash attention kernel does not support rotary embeddings with a block table")
    
    # pre-store in k_cache, v_cache - this should be under a conditional
    B, T = q.shape[0], q.shape[1]
    causal = T > 1
    if prestore_kv_cache:
        if block_table is None:
            if T == 1: 
                b_indices = torch.arange(B, device=k_cache.device)
                t_indices = cache_seqlens.view(-1)
                k_cache[b_indices, t_indices, :, :] = k[:, 0, :, :]
                v_cache[b_indices, t_indices, :, :] = v[:, 0, :, :]
                cache_seqlens = cache_seqlens + 1
            else:
                k_cache[:, :T, :, :] = k[:, :T, :, :]
                v_cache[:, :T, :, :] = v[:, :T, :, :]
                cache_seqlens = torch.full_like(cache_seqlens, T) 
        else:
            update_paged_kv_cache(k=k, 
                                   v=v, 
                                   block_table=block_table, 
                                   cache_seq_len=cache_seqlens, 
                                   k_cache=k_cache, 
                                   v_cache=v_cache) 
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
