from typing import Optional, Tuple
import torch
import math
from .utils import fit_powerlaw_linreg_torch
from .threshold_attention_triton import thresh_attn_fused_wrapped, thresh_attn_reference
#from .threshold_attention_triton import thresh_attn_reference


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

# This function has been modified from torch's SDPA attenion example: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
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
    attn_weight_nan = attn_weight.masked_fill(attn_mask.logical_not(), torch.nan)
    quantiles = torch.nanquantile(attn_weight_nan, percentile, dim=-1)

    #thresh_mask = attn_weight >= quantiles.unsqueeze(-1)


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
    #threshold= torch.zeros_like(generation_counter).unsqueeze(-1)
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

    retain_perc = count_nonzero / (token_counter.unsqueeze(-1).unsqueeze(-1) + 1) * 100
    retain_perc = retain_perc.squeeze().mean(dim=-1, keepdim=True) # B, 1
    #print ("[DEBUG] Retain percentage: ", retain_perc)
    #mean_retain_perc = torch.mean(retain_perc)
    #print ("[DEBUG] Mean retain percentage: ", mean_retain_perc)
    


    #attn_output_reference, _ = thresh_attn_reference(
    #    query,
    #    key,
    #    value,
    #    threshold,
    #    attn_mask=attn_mask,
    #    enable_gqa=enable_gqa,
    #)

    #rtol = 0.0
    ##print(f"attn_output: {attn_output - attn_output_reference}")

    ## Print the norm of the difference
    ##print(f"norm of the difference: {torch.norm(attn_output - attn_output_reference)}")

    ## Check where they differ
    #diff_mask = ~torch.isclose(attn_output, attn_output_reference, rtol=0.0, atol=1e-2)

    ## Print indices and values where they differ
    #if diff_mask.any():
    #    mismatched_indices = diff_mask.nonzero(as_tuple=False)
    #    print(f"mismatched_indices: {mismatched_indices}")
    #    for i in range(mismatched_indices.shape[0]):
    #        idx = tuple(mismatched_indices[i].tolist())
    #        val_a = attn_output[idx].item()
    #        val_b = attn_output_reference[idx].item()
    #        print(f"Index {idx}: a={val_a}, b={val_b}, diff={abs(val_a - val_b)}")
    #    #for idx in mismatched_indices:
    #    #    print(f"Index {idx}: a={attn_output[idx]}, b={attn_output_reference[idx]}, diff={abs(attn_output[idx] - attn_output_reference[idx])}")
    #else:
    #    print("All elements are close.")



    #assert torch.allclose(attn_output, attn_output_reference, atol=1e-2, rtol=rtol), f"Outputs do not match! {attn_output} vs {attn_output_reference}
    return attn_output, retain_perc
