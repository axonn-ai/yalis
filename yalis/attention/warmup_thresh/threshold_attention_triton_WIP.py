import torch

import triton
import triton.language as tl
import math
import random
from torch.library import triton_op, wrap_triton
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load
from typing import Optional
import os
import sys
import time
DEVICE = "cuda"

_BASE_DIR = os.path.dirname(__file__)
_DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192]

def _parse_seq_lens(argv):
    seq_arg = None
    for i, arg in enumerate(argv[1:]):
        if arg.startswith("--seqs="):
            seq_arg = arg.split("=", 1)[1]
            break
        if arg == "--seqs" and i + 2 <= len(argv):
            seq_arg = argv[i + 2]
            break
    if not seq_arg:
        return _DEFAULT_SEQ_LENS
    parts = [p.strip() for p in seq_arg.split(",") if p.strip()]
    seqs = []
    for p in parts:
        if p.lower().endswith("k"):
            seqs.append(int(float(p[:-1]) * 1024))
        else:
            seqs.append(int(p))
    return seqs or _DEFAULT_SEQ_LENS
_kernel_arg = next((arg for arg in sys.argv[1:] if arg.endswith(".cu")), None)
SEQ_LENS = _parse_seq_lens(sys.argv)
if _kernel_arg is None:
    _kernel_source = os.path.join(_BASE_DIR, "thresh_attn_cuda.cu")
else:
    _kernel_source = _kernel_arg
    if not os.path.isabs(_kernel_source):
        _kernel_source = os.path.join(_BASE_DIR, _kernel_source)
    _kernel_source = os.path.normpath(_kernel_source)

# Currently only have 1 kernel that uses gmem, can make a more advanced way to specify callers later
USE_GMEM_KERNEL = "tiled_fused_v1" in os.path.basename(_kernel_source)

decode_attn_cuda = load(
    name="decode_attn_cuda",
    sources=[os.path.join(_BASE_DIR, "thresh_attn_c.cpp"), _kernel_source],
    verbose=True,
    extra_cuda_cflags=['-arch=sm_90', '-O3']
)

print(f"PID: {os.getpid()} | kernel: {_kernel_source} | gmem: {USE_GMEM_KERNEL} | seqs: {SEQ_LENS}")

@torch.library.custom_op("yalis::decode_attn", mutates_args=())
def decode_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    threshold: torch.Tensor,
    scale_factor: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if USE_GMEM_KERNEL:
        return decode_attn_cuda.decode_attn_fwd_gmem(
            query,
            key,
            value,
            attn_mask,
            threshold,
            scale_factor,
        )
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
    
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

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

#@torch.compile(mode="max-autotune-no-cudagraphs")
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
    B, H, T, D = 32, 16, 8192, 128
    query = torch.randn((B, H, 1, D), device=DEVICE).half()
    key = torch.randn((B, H, T, D), device=DEVICE).half()
    value = torch.randn((B, H, T, D), device=DEVICE).half()
    # print (f"Query: {query} \n\n")
    # print (f"Key: {key} \n\n")
    scores = query @ key.transpose(-2, -1) / math.sqrt(D)
    # print (f"Scores: {scores} \n\n")
    scores = torch.softmax(scores, dim=-1)
    # print (f"Scores: {scores} \n\n")
    #print (f"Scores: {scores} \n\n")


    attn_mask = (
        torch.arange(0, T, device=DEVICE).unsqueeze(0).unsqueeze(0) < 1024
    )
    # print (f"Value: {value} \n\n")
    threshold = (
        get_threshold(query, key, value, attn_mask=None)
        .to(device=DEVICE, dtype=torch.float16)
        .reshape((B, H))
    )
    # print (f"Threshold: {threshold} \n\n")

    # Run the reference implementation
    ref_out, _ = thresh_attn_reference(
        query, key, value, threshold, attn_mask=None
    )
    print(ref_out.shape, ref_out.dtype)

    # Run the Triton implementation
    triton_out = thresh_attn_fused(
        query, key, value, threshold, attn_mask=None
    )
    print(triton_out.shape, triton_out.dtype)
    torch.cuda.synchronize()  

    # Compare the outputs
    rtol = 1e-2
    if (
        torch.version.hip is not None
        and triton.runtime.driver.active.get_current_target().arch == "gfx90a"
    ):
        rtol = 1e-2
    
    if not torch.allclose(ref_out, triton_out, atol=1e-2, rtol=rtol):
        diff = ref_out - triton_out
        abs_diff = diff.abs()
        print("Outputs do NOT match.")
        print(" Max diff:", abs_diff.max().item())
        print(" Mean diff:", abs_diff.mean().item())
        # Print a small slice so it doesn't spam the console
        print(" diff:", abs_diff)
        raise AssertionError("Outputs do not match within atol=1e-2, rtol=1e-2")
    else:
        print("Outputs match within atol=1e-2, rtol=1e-2")
    # assert torch.allclose(
    #     ref_out, triton_out, atol=1e-2, rtol=rtol
    # ), f"Outputs do not match! {ref_out} vs {triton_out}"
#
#
#@torch.compile(mode="max-autotune-no-cudagraphs")
def sdpa_attn(
    query, key, value, attn_mask=None, enable_gqa=False
) -> torch.Tensor:
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            enable_gqa=enable_gqa,
        )


# Triton benchmarking to compare performance
BATCH, N_HEADS, HEAD_DIM = 32, 32, 128

# vary seq length for fixed head and batch=4
config = triton.testing.Benchmark(
    x_names=["N_CTX"],
    x_vals=SEQ_LENS,
    line_arg="mode",
    line_vals=[0, 0.5, 0.75, 0.875, 0.95, -1],
    line_names=[
        "CUDA (Percentile: 0)",
        "CUDA (Percentile: 0.5)",
        "CUDA (Percentile: 0.75)",
        "CUDA (Percentile: 0.875)",
        "CUDA (Percentile: 0.95)",
        "Torch",
    ],
    styles=[
        ("red", "-"),
        ("blue", "-"),
        ("green", "-"),
        ("orange", "-"),
        ("purple", "-"),
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
    torch.compiler.reset()
    dtype = torch.float16
    q = torch.randn((BATCH, H, 1, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    kt = k.transpose(-2, -1).contiguous()
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    #attn_mask = torch.arange(0, N_CTX, device=device).unsqueeze(0).unsqueeze(0) < N_CTX // 2
    if mode == -1:
        compiled_fn = torch.compile(sdpa_attn, mode="max-autotune-no-cudagraphs")
        fn = lambda: compiled_fn(q, k, v, attn_mask=None, enable_gqa=False)
    else:
        compiled_fn = torch.compile(thresh_attn_fused_wrapped, mode="max-autotune-no-cudagraphs")
        threshold = get_threshold(q, k, v, attn_mask=None, percentile=mode, enable_gqa=False).reshape((BATCH, H)).to(device=device, dtype=dtype)
        fn = lambda: compiled_fn(q, k, v, threshold, attn_mask=None, enable_gqa=False)
    
    ms = triton.testing.do_bench(fn)
    return ms


# We want to run the run the threshold attention kernel with a given batch size, for nvtx profiling
def run_thresh_attn_kernel(BATCH, H, N_CTX, HEAD_DIM, percentile, device=DEVICE):
    dtype = torch.float16
    q = torch.randn((BATCH, H * 4, 1, HEAD_DIM), dtype=dtype, device=device)
    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
    attn_mask = torch.arange(0, N_CTX, device=device).unsqueeze(0).unsqueeze(0) < N_CTX // 2

    threshold = get_threshold(q, k, v, attn_mask=attn_mask, percentile=percentile, enable_gqa=True).reshape((BATCH, H * 4)).to(device=device, dtype=dtype)
    fn = lambda: thresh_attn_fused_wrapped(q, k, v, threshold, attn_mask=attn_mask, enable_gqa=True)

    # Warmup
    for i in range(5):
        fn()

    torch.cuda.nvtx.range_push("thresh_attn_kernel")
    for i in range(5):
        fn()
    torch.cuda.nvtx.range_pop()


def benchmark_torch_compiled_kernels(
    fn1, fn2, args1=(), args2=(), kwargs1={}, kwargs2={}, 
    warmup=10, iters=100, verbose=True
):
    def time_fn(fn, args, kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Warmup
        for _ in range(warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Timed runs
        start.record()
        for _ in range(iters):
            fn(*args, **kwargs)
        end.record()

        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end) / iters
        return elapsed_ms

    compiled_fn1 = torch.compile(fn1, mode="max-autotune-no-cudagraphs")
    compiled_fn2 = torch.compile(fn2, mode="max-autotune-no-cudagraphs")

    ms1 = time_fn(compiled_fn1, args1, kwargs1)
    ms2 = time_fn(compiled_fn2, args2, kwargs2)

    if verbose:
        print(f"{fn1.__name__} : {ms1:.3f} ms")
        print(f"{fn2.__name__} : {ms2:.3f} ms")

    return ms1, ms2

def benchmark_thresh_attn_fused_wrapped(BATCH, H, N_CTX, HEAD_DIM, percentile):
    query = torch.randn((BATCH, H * 4, 1, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    key = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    value = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    attn_mask = torch.arange(0, N_CTX, device=DEVICE).unsqueeze(0).unsqueeze(0) < N_CTX // 2
    threshold = get_threshold(query, key, value, attn_mask=attn_mask, percentile=percentile, enable_gqa=True).reshape((BATCH, H * 4)).to(device=DEVICE, dtype=torch.float16)

    ms_thresh, ms_sdpa = benchmark_torch_compiled_kernels(
        thresh_attn_fused_wrapped,
        sdpa_attn,
        args1=(query, key, value, threshold, attn_mask, True),
        args2=(query, key, value, attn_mask, True),
        kwargs1={},
    )

    print (f"N_CTX: {N_CTX}, Percentile: {percentile}, Thresh: {ms_thresh:.3f} ms, SDPA: {ms_sdpa:.3f} ms")


def run_all_benchmarks():
    for N_CTX in SEQ_LENS:
        for percentile in [0.5, 0.75, 0.875, 0.95]:
            torch.compiler.reset()
            benchmark_thresh_attn_fused_wrapped(BATCH, N_HEADS, N_CTX, HEAD_DIM, percentile)


if __name__ == "__main__":
    test()
    bench_flash_attention.run(save_path=".", print_data=True)
    #run_all_benchmarks()
    #run_thresh_attn_kernel(16, 8, 4096, 128, 0.75)
