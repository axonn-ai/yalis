from .registry import register_attention
import torch
from axonn import axonn as ax 
import math 
from axonn.intra_layer.communication import Drop, Gather
import torch.distributed as dist
from typing import Optional
from torch.nn.attention.flex_attention import flex_attention

def build_mask_from_index(index, t_max):
        B = index.size(0)
        # Create a range [0, 1, 2, ..., t_max-1] and reshape to [1, t_max] so it can broadcast.
        arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
        # Compare to index[:, None]: [B, 1] which will broadcast to [B, t_max]
        return arange_t <= index.unsqueeze(1)

def index_into_rope_cache_gen(cache, index):
    # index - [B, T]
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(
        index.size(0), 1, -1
    )

def create_upper_mask(dim, device):
    mask = torch.triu(torch.ones(dim, dim, dtype=torch.bool, device=device), diagonal=1)
    mask = mask.to(torch.float32)
    mask.masked_fill_(mask.bool(), -float("inf"))
    return mask

def intra_head_sdpa(q, k, v, attn_mask, process_group, enable_gqa, parallel=True):
    mask = create_upper_mask(q.size(2), q.device) if attn_mask is None else attn_mask
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
        #k = Drop.apply(k, process_group).contiguous()
    scale = 1.0 / math.sqrt(d)
    if enable_gqa:
        q = q * scale
        S = torch.einsum("b g h n d, b g o d t -> b g h n t", q, k.mT).clone().contiguous()
    else:
        q = q * scale
        S = (q @ k.mT).clone().contiguous()
    if parallel:
        dist.all_reduce(S, op=dist.ReduceOp.SUM, group=process_group)
    S = S + mask
    A = torch.nn.functional.softmax(S, dim=-1, dtype=torch.float).to(dtype=q.dtype)
    O = A @ v
    if enable_gqa:
        O = O.view(B, g * hpg, n_q, -1)
    O = Gather.apply(O, process_group)
    return O



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
    flex_attention_block_mask = None,
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

    if ax.config.G_intra_c > 1 and use_intra_head_parallelism:
        k_cache[b_indices, :, t_indices, :] = Drop.apply(k[:, :, 0, :], ax.comm_handle.inner_intra_layer_parallel_group)
        v_cache[b_indices, :, t_indices, :] = Drop.apply(v[:, :, 0, :], ax.comm_handle.inner_intra_layer_parallel_group)
    else:
        k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
        v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]
    mask = build_mask_from_index(token_counter, t_max=k_cache.size(-2))
    
    
    enable_gqa = q.size(1) != k.size(1)
    if use_intra_head_parallelism:
        assert not use_flex, "Intra head parallelism is not supported with flex attention"
        mask_float = torch.zeros_like(mask, dtype=torch.float32)
        mask_float = mask_float.masked_fill(~mask, float("-inf"))
        mask_float = mask_float[:, None, None, :]
        mask_float = mask_float.unsqueeze(1)
        out = intra_head_sdpa(
            q, k_cache, v_cache, mask_float,
            ax.comm_handle.inner_intra_layer_parallel_group,
            enable_gqa, parallel=True
        )
        return out
    else:
        if use_flex:
            assert flex_attention_block_mask is not None, "flex attention requires a block mask" 
            out = flex_attention(q, k_cache, v_cache, enable_gqa=enable_gqa, block_mask=flex_attention_block_mask)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k_cache, v_cache, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa
            )
        return out

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
    
    if ax.config.G_intra_c > 1 and use_intra_head_parallelism:
        k_cache[:, :, :T, :] = Drop.apply(k[:, :, :T, :], ax.comm_handle.inner_intra_layer_parallel_group)
        v_cache[:, :, :T, :] = Drop.apply(v[:, :, :T, :], ax.comm_handle.inner_intra_layer_parallel_group)
    else:
        k_cache[:, :, :T, :] = k[:, :, :T, :]
        v_cache[:, :, :T, :] = v[:, :, :T, :]

    enable_gqa = q.size(1) != k.size(1)
    if False: # do not use intra head in 
        out = intra_head_sdpa(
            q, k, v, None,
            ax.comm_handle.inner_intra_layer_parallel_group,
            enable_gqa, parallel=True
        )
        return out
    else:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        return out


def sdpa_and_flex_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              use_flex: bool = False,
              flex_attention_block_mask = None,
              **kwargs) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with SDPA attention.")
    T = q.shape[-2]


    if use_flex: 
        assert flex_attention_block_mask is not None, "flex attention requires a block mask" 

    if T==1:
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
                 flex_attention_block_mask
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
                     use_intra_head_parallelism
                 )
    return y

@register_attention("sdpa")
def sdpa_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              **kwargs):
    return sdpa_and_flex_attention(q=q,
                                   k=k,
                                   v=v,
                                   k_cache=k_cache, 
                                   v_cache=v_cache,
                                   cache_seqlens=cache_seqlens,
                                   rotary_cos=rotary_cos,
                                   rotary_sin=rotary_sin,
                                   use_intra_head_parallelism=use_intra_head_parallelism,
                                   use_flex=False,
                                   **kwargs)

@register_attention("sdpa_and_flex")
def sdpa_and_flex_attention_(q: torch.Tensor, 
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
    return sdpa_and_flex_attention(q=q,
                            k=k,
                            v=v,
                            k_cache=k_cache, 
                            v_cache=v_cache,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            use_intra_head_parallelism=use_intra_head_parallelism,
                            use_flex=True,
                            **kwargs)
