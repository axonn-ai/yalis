from .registry import register_attention
import torch
from axonn import axonn as ax 
import math 
from axonn.intra_layer.communication import Drop, Gather
import torch.distributed as dist
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention

def index_into_rope_cache_gen(cache, index):
    # index - [B, T]
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(
        index.size(0), 1, -1
    )

def rotary_kv_update_flex_gen(
    q: torch.Tensor,  # B,nh,1,hs
    k: torch.Tensor,  # B,nh,1,hs
    v: torch.Tensor,  # B,nh,1,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    flex_attention_block_mask,
) -> torch.Tensor:
    if cos is not None and sin is not None:
        cos = index_into_rope_cache_gen(cos, token_counter)
        sin = index_into_rope_cache_gen(sin, token_counter)

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

    B = k_cache.size(0)
    b_indices = torch.arange(B, device=k_cache.device)
    t_indices = token_counter.view(-1)

    k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
    v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]
    
    enable_gqa = q.size(1) != k.size(1)
    out = flex_attention(q, k_cache, v_cache, enable_gqa=enable_gqa, block_mask=flex_attention_block_mask)
    return out

def rotary_kv_update_flex_prefill(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    flex_attention_block_mask,
) -> torch.Tensor:
    B, T = q.shape[0], q.shape[-2]
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
    
    k_cache[:, :, :T, :] = k[:, :, :T, :]
    v_cache[:, :, :T, :] = v[:, :, :T, :]

    enable_gqa = q.size(1) != k.size(1)
    out = flex_attention(q, k, v, enable_gqa=enable_gqa, block_mask=flex_attention_block_mask)
    return out


def flex_attention_inner(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              flex_attention_block_mask = None,
              **kwargs) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with Flex attention.")
    T = q.shape[-2]


    if T==1:
        y = rotary_kv_update_flex_gen(
                 q,
                 k,
                 v,
                 rotary_cos,
                 rotary_sin,
                 cache_seqlens,  # B,1
                 k_cache,  # B,nh,t_max,hs
                 v_cache,  # B,nh,t_max,hs
                 flex_attention_block_mask
             )
    else:
        y = rotary_kv_update_flex_prefill(
                     q,
                     k,
                     v,
                     rotary_cos,
                     rotary_sin,
                     k_cache,  # B,nh,t_max,hs
                     v_cache,  # B,nh,t_max,hs
                     flex_attention_block_mask,
                 )
    return y

@register_attention("flex")
def flex_attention_(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              **kwargs):  
    assert "flex_attention_block_mask" in kwargs, "flex attention requires a block mask"
    return flex_attention_inner(q=q,
                            k=k,
                            v=v,
                            k_cache=k_cache, 
                            v_cache=v_cache,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            #flex_attention_block_mask=flex_attention_block_mask,
                            **kwargs)
