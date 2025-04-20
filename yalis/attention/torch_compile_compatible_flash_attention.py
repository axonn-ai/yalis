# here we are registering the flash_attn_with_kv_cache kernel as a custom pytorch op 
# so that it doesn't lead to torch compile graph breaks
# more info can be found here - https://pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial


from flash_attn import flash_attn_with_kvcache
import torch
from typing import Sequence
    

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
                            window_size: Sequence[int]) -> torch.Tensor:
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
            )
    return y

@torch_compile_compatible_flash_attention.register_fake
def _(q, k_cache, v_cache, k, v, causal, cache_seqlens, rotary_cos, rotary_sin, rotary_interleaved, window_size):
    return torch.empty_like(q)