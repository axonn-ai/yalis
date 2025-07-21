import torch

import triton
import triton.language as tl
import math
import random
from torch.library import triton_op, wrap_triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from typing import Optional

DEVICE = "cuda"

decode_attn_cuda = load(
    name="decode_attn_cuda",
    sources=["thresh_attn_c.cpp", "thresh_attn_cuda.cu"],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_90', '-O3']
)

@torch.library.custom_op("yalis::decode_attn", mutates_args=())
def decode_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    threshold: torch.Tensor,
    scale_factor: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return decode_attn_cuda.decode_attn_fwd(
        query,
        key,
        value,
        attn_mask,
        threshold,
        scale_factor,
    )

@decode_attn.register_fake
def _(query, key, value, threshold, scale_factor, attn_mask=None):
    return torch.empty_like(query)

def thresh_attn_fused(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    threshold: torch.Tensor,
    attn_mask: torch.Tensor = None,
    enable_gqa: bool = False) -> torch.Tensor:
    L, T = query.size(-2), key.size(-2)
    assert L == 1, "Only decodes for one query at a time"
    B, H = query.shape[0], query.shape[1]
    scale_factor = 1 / math.sqrt(query.size(-1))
    HEAD_DIM = query.size(-1)

    #assert HEAD_DIM in {16, 32, 64, 128, 256}
    #assert threshold.shape == (B, H), f"Threshold shape {threshold.shape} does not match query shape {query.shape}"

    attn_bias = torch.zeros(B, H, L, T, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
    
    #print (f"attn_bias: {attn_bias}")
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    grid = lambda args: (query.shape[0], query.shape[1], 1)
    o = torch.empty_like(query)

    # Does the scores need to float32?
    scores = torch.empty((B, H, 1, T), device=query.device, dtype=torch.float32)
    valid_indices = torch.zeros((B, H, 1, T), device=query.device, dtype=torch.int32) - 1

    threshold = threshold.to(torch.float32).contiguous()
    attn_bias = attn_bias.to(torch.float32).contiguous()

    return decode_attn(
        query,
        key,
        value,
        threshold,
        scale_factor,
        attn_bias,
    )

@torch.compile(mode="max-autotune-no-cudagraphs")
def thresh_attn_fused_wrapped(
   query: torch.Tensor,
   key: torch.Tensor,
   value: torch.Tensor,
   threshold: torch.Tensor,
   attn_mask: torch.Tensor = None,
   enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Wrapper function for the Triton implementation of thresholded attention.
    """

    return thresh_attn_fused(
        query=query,
        key=key,
        value=value,
        threshold=threshold,
        attn_mask=attn_mask,
        enable_gqa=enable_gqa,
    ), None


def thresh_attn_reference(
    query, key, value, threshold, scale=None, attn_mask=None, enable_gqa=False
) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    assert L == 1, "Only decodes for one query at a time"
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    B, H = query.shape[0], query.shape[1]
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    threshold = threshold.unsqueeze(-1).unsqueeze(-1)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)

    thresh_mask = attn_weight >= threshold

    attn_weight = attn_weight.masked_fill_(thresh_mask.logical_not(), 0.0)

    count_nonzero = thresh_mask.count_nonzero(dim=-1)

    return attn_weight @ value, count_nonzero 


def get_threshold(query, key, value, percentile=0.5, attn_mask=None, enable_gqa=False):
    L, S = query.size(-2), key.size(-2)
    assert L == 1, "Only decodes for one query at a time"
    scale_factor = 1 / math.sqrt(query.size(-1))
    B, H = query.shape[0], query.shape[1]

    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias
  
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1).to(torch.float32)
    # print (f"Attn weight: {attn_weight}")
    attn_weight += attn_bias

    # Change -inf to NaN
    attn_weight = torch.where(
        attn_weight == float("-inf"), torch.nan, attn_weight
    )
    # print (f"Attn weight: {attn_weight}")
    # print (f"Attn weight shape: {attn_weight.dtype}")

    threshold = torch.nanquantile(attn_weight, percentile, dim=-1)
    return threshold


def test():
    B, H, T, D = 4, 8, 2048, 128
    query = torch.randn((B, H, 1, D), device=DEVICE).half()
    key = torch.randn((B, H, T, D), device=DEVICE).half()
    value = torch.randn((B, H, T, D), device=DEVICE).half()
    print (f"Query: {query} \n\n")
    print (f"Key: {key} \n\n")
    scores = query @ key.transpose(-2, -1) / math.sqrt(D)
    scores = torch.softmax(scores, dim=-1)
    print (f"Scores: {scores} \n\n")


    attn_mask = (
        torch.arange(0, T, device=DEVICE).unsqueeze(0).unsqueeze(0) < 1024
    )
    # print (f"Value: {value} \n\n")
    threshold = (
        get_threshold(query, key, value, attn_mask=attn_mask)
        .to(device=DEVICE, dtype=torch.float16)
        .reshape((B, H))
    )
    # print (f"Threshold: {threshold} \n\n")

    # Run the reference implementation
    ref_out, _ = thresh_attn_reference(
        query, key, value, threshold, attn_mask=attn_mask
    )
    print(ref_out.shape, ref_out.dtype)

    # Run the Triton implementation
    triton_out = thresh_attn_fused(
        query, key, value, threshold, attn_mask=attn_mask
    )
    print(triton_out.shape, triton_out.dtype)
    torch.cuda.synchronize()  

    # Compare the outputs
    rtol = 0.0
    if (
        torch.version.hip is not None
        and triton.runtime.driver.active.get_current_target().arch == "gfx90a"
    ):
        rtol = 1e-2
    assert torch.allclose(
        ref_out, triton_out, atol=1e-2, rtol=rtol
    ), f"Outputs do not match! {ref_out} vs {triton_out}"
#
#
@torch.compile(mode="max-autotune-no-cudagraphs")
def sdpa_attn(
    query, key, value, attn_mask=None, enable_gqa=False
) -> torch.Tensor:
    with sdpa_kernel(SDPBackend.MATH):
        return torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            enable_gqa=enable_gqa,
        )


# Triton benchmarking to compare performance
BATCH, N_HEADS, HEAD_DIM = 32, 8, 128

# vary seq length for fixed head and batch=4
config = triton.testing.Benchmark(
    x_names=["N_CTX"],
    x_vals=[512, 1024, 2048, 4096],
    line_arg="mode",
    line_vals=[0, 0.5, 0.75, 0.875, -1],
    line_names=[
        "CUDA (Percentile: 0)",
        "CUDA (Percentile: 0.5)",
        "CUDA (Percentile: 0.75)",
        "CUDA (Percentile: 0.875)",
        "Torch",
    ],
    styles=[
        ("red", "-"),
        ("blue", "-"),
        ("green", "-"),
        ("orange", "-"),
        ("black", "--"),
    ],
    ylabel="Time (ms)",
    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
    args={
        "H": N_HEADS,
        "BATCH": BATCH,
        "HEAD_DIM": HEAD_DIM,
    },
)
#
#

@triton.testing.perf_report(config)
def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, device=DEVICE):
    dtype = torch.float16
    q = torch.randn((BATCH, H * 4, 1, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    attn_mask = torch.arange(0, N_CTX, device=device).unsqueeze(0).unsqueeze(0) < N_CTX // 2
    if mode == -1:
        fn = lambda: sdpa_attn(q, k, v, attn_mask=attn_mask, enable_gqa=True)
    else:
        threshold = get_threshold(q, k, v, attn_mask=attn_mask, percentile=mode, enable_gqa=True).reshape((BATCH, H * 4)).to(device=device, dtype=dtype)
        fn = lambda: thresh_attn_fused_wrapped(q, k, v, threshold, attn_mask=attn_mask, enable_gqa=True)

    ms = triton.testing.do_bench(fn)
    return ms

if __name__ == "__main__":
    #test()
    bench_flash_attention.run(save_path=".", print_data=True)
