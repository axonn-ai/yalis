from typing import Optional, Tuple
import torch
import math
from .utils import fit_powerlaw_linreg_torch
#from .threshold_attention_triton import thresh_attn_fused, thresh_attn_reference
from .threshold_attention_triton import thresh_attn_reference

# This function has been modified from torch's SDPA attention example: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def thresh_attn(query, key, value, threshold, attn_mask=None, enable_gqa=False) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    thresh_mask = attn_weight < threshold
    #thresh_mask[..., -1] = False
    
    attn_weight = attn_weight.masked_fill(thresh_mask, 0.0)

    return attn_weight @ value

# This function has been modified from torch's SDPA attention example: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
def thresh_warmup(query, key, value, percentile, attn_mask=None, enable_gqa=False) -> torch.Tensor:
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

    #attn_bias_nanattn_weight_nan = attn_weight
    attn_weight_nan = attn_weight.masked_fill(attn_mask.logical_not(), torch.nan).to(dtype=torch.float32)
    quantiles = torch.nanquantile(attn_weight_nan, percentile, dim=-1)

    return attn_weight @ value, quantiles


def thresh_attention_warmup_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    percentile: float,
    attn_mask: Optional[torch.Tensor],
    enable_gqa: Optional[bool] = False,
    **kwargs,
):
    attn_output, quantiles = thresh_warmup(
        query,
        key,
        value,
        percentile=percentile,
        attn_mask=attn_mask,
        enable_gqa=enable_gqa,
    )
    return attn_output, quantiles

# This function has been modified from HuggingFace's SPDA implementation: https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py
#@torch.compiler.disable()
def thresh_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    generation_counter: torch.Tensor,
    token_counter: torch.Tensor,
    powerlaw_a: torch.Tensor,
    powerlaw_b: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    enable_gqa: Optional[bool] = False,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    #print ("[DEBUG] Executing generation stage")
    #print ("[DEBUG] Generation step: ", generation_counter.shape)
    #print ("[DEBUG] Power law a:", module.powerlaw_a.shape)
    #print ("[DEBUG] Power law b:", module.powerlaw_b.shape)
    threshold = powerlaw_a * (generation_counter.unsqueeze(-1) ** powerlaw_b) + 1e-9
    #threshold = threshold.to(dtype=torch.float32)
    #print ("[DEBUG] Threshold: ", threshold)

    attn_output, count_nonzero = thresh_attn_reference(
        query,
        key,
        value,
        threshold,
        attn_mask=attn_mask,
        enable_gqa=enable_gqa,
    )
    retain_perc = count_nonzero / token_counter.unsqueeze(-1).unsqueeze(-1) * 100
    mean_retain_perc = torch.mean(retain_perc)
    #print ("[DEBUG] Retain percentage: ", mean_retain_perc)


    #attn_output = thresh_attn_reference(
    #    query,
    #    key,
    #    value,
    #    threshold,
    #    attn_mask=attn_mask,
    #    enable_gqa=enable_gqa,
    #)

    #rtol = 1e-2
    #assert torch.allclose(attn_output, attn_output_fused, atol=1e-2, rtol=rtol), f"Outputs do not match! {attn_output} vs {attn_output_fused}"
    return attn_output, mean_retain_perc 