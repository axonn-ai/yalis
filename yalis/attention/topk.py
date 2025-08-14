from .registry import register_attention
import torch
from axonn import axonn as ax 
import math 
import torch.distributed as dist
from typing import Optional, Tuple

def topk_sdpa_quantile(query, key, value, topk, token_counter,attn_mask=None, enable_gqa=False) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    attn_bias_nan = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_bias_nan.masked_fill_(attn_mask.logical_not(), torch.nan)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    quantile = 1 - topk


    attn_weight_nan = attn_weight.masked_fill(attn_mask.logical_not(), torch.nan)
    quantiles = torch.nanquantile(attn_weight_nan, quantile, dim=-1, keepdim=True)

    mask = attn_weight >= quantiles
    
    attn_weight = attn_weight.masked_fill(mask.logical_not(), 0.0)

    count_nonzero = mask.count_nonzero(dim=-1)

    return attn_weight @ value, count_nonzero


def topk_sdpa(query, key, value, topk, token_counter,attn_mask=None, enable_gqa=False) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    attn_bias_nan = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            attn_bias_nan.masked_fill_(attn_mask.logical_not(), torch.nan)
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    if topk < 1:
        assert B == 1, "topk must be a scalar for batch size > 1"
        token_counter_scalar = token_counter.item()
        topk = int(topk * token_counter_scalar)
        #print(f"token_counter_scalar: {token_counter_scalar}, topk: {topk}")

    if L == 1: # Generation Step
        topk_vals, topk_indices = torch.topk(attn_weight, k=topk, dim=-1)
        #print(f"topk_vals: {topk_vals}, topk_indices: {topk_indices}")
        attn_weight_masked = torch.zeros_like(attn_weight)
        attn_weight_masked.scatter_(-1, topk_indices, topk_vals)
        attn_weight = attn_weight_masked

    return attn_weight @ value


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
    topk: float,
    retain_perc: torch.Tensor, # B,nh, num_warmups
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
    out, count_nonzero = topk_sdpa_quantile(q, k_cache, v_cache, topk, token_counter, attn_mask=mask[:, None, None, :], enable_gqa=enable_gqa)
    #retain_perc.add_(topk)
    retain_p = count_nonzero / (token_counter.unsqueeze(-1).unsqueeze(-1) + 1) * 100
    retain_p = retain_p.squeeze().mean(dim=-1, keepdim=True) # B, 1
    retain_perc.add_(retain_p)
    #print (f"Retain Percentage: {retain_p}")
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

def topk_attention_forward(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              threshold_percentile: float = 0.0,
              retain_perc: Optional[torch.Tensor] = None,
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
            topk=threshold_percentile,
            retain_perc=retain_perc,
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


@register_attention("topk")
def topk_attn(q: torch.Tensor, 
              k: torch.Tensor, 
              v: torch.Tensor,  
              k_cache: Optional[torch.Tensor] = None, 
              v_cache: Optional[torch.Tensor] = None,
              cache_seqlens: Optional[torch.Tensor] = None,  
              rotary_cos: Optional[torch.Tensor] = None,
              rotary_sin: Optional[torch.Tensor] = None,
              use_intra_head_parallelism: bool = False,
              **kwargs):  
    assert "threshold_percentile" in kwargs, "topk attention requires a threshold percentile"
    assert "retain_perc" in kwargs, "topk attention requires a retain percentage"
    return topk_attention_forward(q=q,
                            k=k,
                            v=v,
                            k_cache=k_cache, 
                            v_cache=v_cache,
                            cache_seqlens=cache_seqlens,
                            rotary_cos=rotary_cos,
                            rotary_sin=rotary_sin,
                            **kwargs)
