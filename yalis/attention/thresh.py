from .registry import register_attention
import torch
from axonn import axonn as ax 
import math 
import torch.distributed as dist
from typing import Optional
from yalis.attention.threshold_attention import thresh_attention_forward, thresh_attention_warmup_forward

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

def lit_rotary_kv_update_gen(
    q: torch.Tensor,  # B,nh,1,hs
    k: torch.Tensor,  # B,nh,1,hs
    v: torch.Tensor,  # B,nh,1,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    token_counter: torch.Tensor,  # B,1
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
    threshold_percentile: float,
    generation_counter: torch.Tensor, # B,1
    warmup_quantiles: torch.Tensor, # B,nh, num_warmups
    retain_perc: torch.Tensor, # B,nh, num_warmups
    powerlaw_a: torch.Tensor, # B,nh, num_warmups
    powerlaw_b: torch.Tensor, # B,nh, num_warmups
    warmup: bool = False,
) -> torch.Tensor:
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
    mask = build_mask_from_index(token_counter, t_max=k_cache.size(-2))
    #[:, None, None, :]
    
    enable_gqa = q.size(1) != k.size(1)
    if warmup:
        out, quantiles = thresh_attention_warmup_forward(q, k_cache, v_cache, threshold_percentile, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa)
        g_indices = generation_counter.view(-1)
        warmup_quantiles[b_indices, :, g_indices - 1] = quantiles[b_indices, :, 0]
        return out
    else:
        #out = torch.nn.functional.scaled_dot_product_attention(
        #    q, k_cache, v_cache, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa
        #)
        out, retain_ = thresh_attention_forward(q, k_cache, v_cache, generation_counter, token_counter, powerlaw_a, powerlaw_b, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa)
        retain_perc.add_(retain_)
        return out


def lit_rotary_kv_update_prefill(
    q: torch.Tensor,  # B,nh,T,hs
    k: torch.Tensor,  # B,nh,T,hs
    v: torch.Tensor,  # B,nh,T,hs
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,  # B,nh,t_max,hs
    v_cache: torch.Tensor,  # B,nh,t_max,hs,
) -> torch.Tensor:
    B, T = q.shape[0], q.shape[-2]
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
    #print("NOT USING IHP *****")
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
    return out

def threshold_attention(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              generation_counter: Optional[torch.Tensor] = None,
              warmup_quantiles: Optional[torch.Tensor] = None,
              warmup: bool = False,
              threshold_percentile: float = 0.0,
              retain_perc: Optional[torch.Tensor] = None,
              powerlaw_a: Optional[torch.Tensor] = None,
              powerlaw_b: Optional[torch.Tensor] = None,
              **kwargs) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with Threshold attention.")
    T = q.shape[-2]

    if T == 1:
        # generative phase
        y = lit_rotary_kv_update_gen(
            q,
            k,
            v,
            rotary_cos,
            rotary_sin,
            cache_seqlens,  # B,1
            k_cache,  # B,nh,t_max,hs
            v_cache,  # B,nh,t_max,hs
            threshold_percentile,
            generation_counter,
            warmup_quantiles,
            retain_perc,
            powerlaw_a,
            powerlaw_b,
            warmup=warmup,

        )
    else:
        # prefill
        y = lit_rotary_kv_update_prefill(
            q,
            k,
            v,
            rotary_cos,
            rotary_sin,
            k_cache,  # B,nh,t_max,hs
            v_cache,  # B,nh,t_max,hs
        )
    
    return y


@register_attention("thresh")
def thresh_attn(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              **kwargs):  
    assert "generation_counter" in kwargs, "thresh attention requires a generation counter"
    assert "warmup_quantiles" in kwargs, "thresh attention requires warmup quantiles"
    assert "warmup" in kwargs, "thresh attention requires warmup flag"
    assert "threshold_percentile" in kwargs, "thresh attention requires a threshold percentile"
    assert "retain_perc" in kwargs, "thresh attention requires a retain percentage"
    return threshold_attention(q=q,
                            k=k,
                            v=v,
                            k_cache=k_cache, 
                            v_cache=v_cache,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            **kwargs)