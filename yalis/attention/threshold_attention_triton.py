import torch

import triton
import triton.language as tl
import math
import random
from torch.library import triton_op, wrap_triton

#DEVICE = triton.runtime.driver.active.get_active_torch_device()
#
#def is_hip():
#   return triton.runtime.driver.active.get_current_target().backend == "hip"
#
#configs = [
#   triton.Config({'BLOCK_N': BN, 'BLOCK_M': BM}, num_stages=s, num_warps=w) \
#   #for BN in [16, 32, 64, 128, 256, 512]\
#   #for BN in [16, 32, 64, 128, 256, 512]\
#   #for BN in [16, 32, 64, 128, 256, 512]\
#   for BN in [64]
#   for BM in [16]
#   for s in ([1] if is_hip() else [3, 4, 7])\
#   for w in [4]\
#]
#
#def keep(conf):
#    BLOCK_N = conf.kwargs["BLOCK_N"]
#    BLOCK_M = conf.kwargs["BLOCK_M"]
#    return True
#
#
#def is_hip():
#   return triton.runtime.driver.active.get_current_target().backend == "hip"
#
#def get_next_power_of_2(x):
# # If x is already a power of 2, return it
# if (x & (x - 1)) == 0:
#   return x
# else:
#   return 1 << (x - 1).bit_length()
#
#
#
#@triton.autotune(list(filter(keep, configs)), key=["T", "HEAD_DIM"])
#@triton.jit
#def decode_attn_fwd(
#    # data pointers
#    Q,            # [B, H, 1, D]
#    K_cache,      # [B, H, T, D]
#    V_cache,      # [B, H, T, D]
#    sm_scale,     # scalar float
#    attn_bias,    # [B, H, 1, T]
#    Out,          # [B, H, 1, D]  ← output
#    scores,       # [B, H, 1, T]  ← scores output
#    valid_indices,  # [B, H, 1, T]  ← valid indices output
#    thresholds,   # [B, H]  ← thresholds
#    # strides
#    stride_qb, stride_qh, stride_qm, stride_qk,
#    stride_kb, stride_kh, stride_kk, stride_kn,
#    stride_vb, stride_vh, stride_vn, stride_vk,
#    stride_ob, stride_oh, stride_om, stride_ok,
#    stride_sb, stride_sh, stride_sm, stride_sn,
#    stride_vib, stride_vih, stride_vim, stride_vin,
#    stride_tb, stride_th,
#    # sizes
#    B, H,
#    T: tl.constexpr,
#    HEAD_DIM: tl.constexpr,
#    BLOCK_M: tl.constexpr,
#    BLOCK_N: tl.constexpr
#):
#    batch_id = tl.program_id(0)
#    head_id = tl.program_id(1)
#
#    q_base = Q + batch_id*stride_qb + head_id*stride_qh
#    k_base = K_cache + batch_id*stride_kb + head_id*stride_kh
#    v_base = V_cache + batch_id*stride_vb + head_id*stride_vh
#    o_base = Out + batch_id*stride_ob + head_id*stride_oh
#    scores_base = scores + batch_id*stride_sb + head_id*stride_sh
#    valid_indices_base = valid_indices + batch_id*stride_vib + head_id*stride_vih
#    thresh_base = thresholds + batch_id*stride_tb + head_id*stride_th
#
#    q_ptr = ( 
#        q_base
#        + tl.zeros((BLOCK_M,), tl.int32)[:, None] * stride_qm 
#        + tl.arange(0, HEAD_DIM)[None, :] * stride_qk 
#    )
#
#    o_ptr = (
#        o_base
#        + tl.zeros((BLOCK_M,), tl.int32)[:, None] * stride_om
#        + tl.arange(0, HEAD_DIM)[None, :] * stride_ok
#    )
#
#    # pointers for Q and outputs: one vector of length HEAD_DIM
#    k_ptr = tl.make_block_ptr(base=k_base, shape=(T, HEAD_DIM), strides=(stride_kn, stride_kk), offsets=(0, 0), block_shape=(HEAD_DIM, BLOCK_N), order=(0,1))
#    attn_bias_ptr = tl.make_block_ptr(base=attn_bias, shape=(1, T), strides=(stride_sm, stride_sn), offsets=(0, 0), block_shape=(1, BLOCK_N), order=(1,0))
#    scores_ptr = tl.make_block_ptr(base=scores_base, shape=(1, T), strides=(stride_sm, stride_sn), offsets=(0,0), block_shape=(1, BLOCK_N), order=(1,0))
#
#
#    # initialize softmax accumulators
#    m = tl.full((1,), -1e9, dtype=tl.float32)   # max
#    l = tl.zeros((1,), dtype=tl.float32) + 1.0         # sum of exps
#    acc = tl.zeros((1, HEAD_DIM), dtype=tl.float32)
#
#    threshold = tl.load(thresh_base)
#
#    # load the single query vector
#    q = tl.load(q_ptr)  # shape [BLOCK_M, HEAD_DIM]
#    scale = sm_scale
#
#    lo, hi = 0, T
#
#    k_ptr = tl.advance(k_ptr , (0, lo))
#    scores_ptr = tl.advance(scores_ptr, (0, lo))
#
#    # Iterate over K and compute the attention scores
#    for start_n in range(0, T, BLOCK_N):
#        # clamp the last block size
#        k = tl.load(k_ptr)  # shape [HEAD_DIM, BLOCK_N]
#        qk = tl.dot(q, k) * scale # shape [1, BLOCK_N]
#        #qk = qk.to(tl.bfloat16)  # convert to float32 for softmax
#
#        bias = tl.load(attn_bias_ptr)  # shape [1, BLOCK_N]
#        qk = qk + bias  # add the attention bias
#
#        # Store the scores in shared memory
#        tl.store(scores_ptr, qk)
#
#        qk_block_max = tl.max(qk, axis=1)  # shape [1]
#        new_m = tl.maximum(m, qk_block_max)  # new max
#        exp_m = tl.exp(m - new_m)  # rescale old sum
#        exp_block = tl.exp(qk - new_m)  # exponential of the current block
#        l = l * exp_m + tl.sum(exp_block, axis=1)
#        m = new_m
#
#        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
#        scores_ptr = tl.advance(scores_ptr, (0, BLOCK_N))
#        attn_bias_ptr = tl.advance(attn_bias_ptr, (0, BLOCK_N))
#
#    scores_offsets = scores_base + tl.arange(0, T)
#
#    scores = tl.load(scores_offsets, mask=tl.arange(0, T) < T)  # [T]
#    probs = tl.exp(scores - m) / l
#    #probs = probs.to(tl.bfloat16)  # [1, T]
#
#    prob_mask = probs >= threshold  # [1, T]
#
#    # Exclusive prefix sum
#    mask_sum = tl.cumsum(prob_mask.to(tl.int32), axis=0)  # [1]
#    mask_indices = tl.arange(0, T)  # [T]
#    mask_sum_exclusive = mask_sum - prob_mask.to(tl.int32)  # [1]
#
#
#    tl.store(valid_indices_base + mask_sum_exclusive, mask_indices, mask=prob_mask)
#
#    for start_n in range(0, T, BLOCK_N):
#        start_n = tl.multiple_of(start_n, BLOCK_N)
#
#        indices = tl.load(valid_indices_base + start_n + tl.arange(0, BLOCK_N))  # [BLOCK_N]
#        idx_mask = indices >= 0
#
#        idx_mask_sum = tl.sum(idx_mask.to(tl.int32), axis=0)
#
#        if idx_mask_sum > 0:
#            qk = tl.load(scores_base + indices, mask=idx_mask, other=0.0)[None, :]  # [1, BLOCK_N]
#
#            exp_scores = tl.exp(qk - m)
#            probs_l = exp_scores / l
#
#
#            v_offsets_n = indices
#            v_offsets_d = tl.arange(0, HEAD_DIM)
#
#            # Create a position mask to load tokens less than the max sequence length
#            mask_pos = v_offsets_n >= 0
#            # Combine the masks
#            mask = (idx_mask & mask_pos)[:, None]  # [BLOCK_N, 1]
#
#
#            v_offsets = v_offsets_n[:, None] * stride_vn + v_offsets_d[None, :]  # [BLOCK_N, HEAD_DIM]
#
#            v = tl.load(v_base + v_offsets, mask=mask, other=0.0)  # [BLOCK_N, HEAD_DIM]
#            probs_l = probs_l.to(tl.float16)  # [1, BLOCK_N]
#            acc = tl.dot(probs_l, v, acc)
#
#    tl.store(o_ptr, acc.to(tl.float16))  # [1, HEAD_DIM]
#   #tl.store(o_ptr, acc)  # [1, HEAD_DIM]
#
#
#
#@triton_op("thresh_attn::thresh_attn_fused", mutates_args={})
#def thresh_attn_fused(
#    query: torch.Tensor,
#    key: torch.Tensor,
#    value: torch.Tensor,
#    threshold: torch.Tensor,
#    attn_mask: torch.Tensor = None,
#    enable_gqa: bool = False) -> torch.Tensor:
#    L, T = query.size(-2), key.size(-2)
#    assert L == 1, "Only decodes for one query at a time"
#    B, H = query.shape[0], query.shape[1]
#    scale_factor = 1 / math.sqrt(query.size(-1))
#    HEAD_DIM = query.size(-1)
#
#    #assert HEAD_DIM in {16, 32, 64, 128, 256}
#    #assert threshold.shape == (B, H), f"Threshold shape {threshold.shape} does not match query shape {query.shape}"
#
#    attn_bias = torch.zeros(B, H, L, T, dtype=query.dtype, device=query.device)
#    if attn_mask is not None:
#        if attn_mask.dtype == torch.bool:
#            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#        else:
#            attn_bias = attn_mask + attn_bias
#
#    if enable_gqa:
#        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
#        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)
#
#    grid = lambda args: (query.shape[0], query.shape[1], 1)
#    o = torch.empty_like(query)
#
#    # Does the scores need to float32?
#    scores = torch.empty((B, H, 1, T), device=query.device, dtype=torch.float32)
#    valid_indices = torch.zeros((B, H, 1, T), device=query.device, dtype=torch.int32) - 1
#
#    wrap_triton(decode_attn_fwd)[grid](
#        query,
#        key,
#        value,
#        scale_factor,
#        attn_bias,
#        o,
#        scores,
#        valid_indices,
#        threshold,
#        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
#        key.stride(0), key.stride(1), key.stride(2), key.stride(3),
#        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
#        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
#        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
#        valid_indices.stride(0), valid_indices.stride(1), valid_indices.stride(2), valid_indices.stride(3),
#        threshold.stride(0), threshold.stride(1),
#        # sizes
#        B, H, T,
#        HEAD_DIM=HEAD_DIM,
#    )
#    #exit (0)
#
#    return o


# def thresh_attn_fused_wrapped(
#    query: torch.Tensor,
#    key: torch.Tensor,
#    value: torch.Tensor,
#    threshold: torch.Tensor,
#    attn_mask: torch.Tensor = None,
#    enable_gqa: bool = False,
# ) -> torch.Tensor:
#    """
#    Wrapper function for the Triton implementation of thresholded attention.
#    """
#    return thresh_attn_fused(
#        query=query,
#        key=key,
#        value=value,
#        threshold=threshold,
#        attn_mask=attn_mask,
#        enable_gqa=enable_gqa,
#    )


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


def get_threshold(query, key, value, percentile=0.5, attn_mask=None):
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


#def test():
#    B, H, T, D = 16, 32, 8192, 128
#    query = torch.randn((B, H, 1, D), device=DEVICE).half()
#    key = torch.randn((B, H, T, D), device=DEVICE).half()
#    value = torch.randn((B, H, T, D), device=DEVICE).half()
#
#    attn_mask = (
#        torch.arange(0, T, device=DEVICE).unsqueeze(0).unsqueeze(0) < 1051
#    )
#    # print (f"Value: {value} \n\n")
#    threshold = (
#        get_threshold(query, key, value, attn_mask=attn_mask)
#        .to(device=DEVICE, dtype=torch.float16)
#        .reshape((B, H))
#    )
#    # print (f"Threshold: {threshold} \n\n")
#
#    # Run the reference implementation
#    ref_out = thresh_attn_reference(
#        query, key, value, threshold, attn_mask=attn_mask
#    )
#    print(ref_out.shape, ref_out.dtype)
#
#    # Run the Triton implementation
#    triton_out = thresh_attn_fused(
#        query, key, value, threshold, attn_mask=attn_mask
#    )
#    print(triton_out.shape, triton_out.dtype)
#
#    # Compare the outputs
#    rtol = 0.0
#    if (
#        torch.version.hip is not None
#        and triton.runtime.driver.active.get_current_target().arch == "gfx90a"
#    ):
#        rtol = 1e-2
#    assert torch.allclose(
#        ref_out, triton_out, atol=1e-2, rtol=rtol
#    ), f"Outputs do not match! {ref_out} vs {triton_out}"
#
#
#@torch.compile()
#def sdpa_attn(
#    query, key, value, attn_mask=None, enable_gqa=False
#) -> torch.Tensor:
#    return torch.nn.functional.scaled_dot_product_attention(
#        query=query,
#        key=key,
#        value=value,
#        attn_mask=attn_mask,
#        enable_gqa=enable_gqa,
#    )
#
#
## Triton benchmarking to compare performance
#BATCH, N_HEADS, HEAD_DIM = 32, 32, 128
#
## vary seq length for fixed head and batch=4
#config = triton.testing.Benchmark(
#    x_names=["N_CTX"],
#    x_vals=[512, 1024, 2048, 4096, 8192],
#    line_arg="mode",
#    line_vals=[0, 0.5, 0.75, 0.875, -1],
#    line_names=[
#        "Triton (Percentile: 0)",
#        "Triton (Percentile: 0.5)",
#        "Triton (Percentile: 0.75)",
#        "Triton (Percentile: 0.875)",
#        "Torch",
#    ],
#    styles=[
#        ("red", "-"),
#        ("blue", "-"),
#        ("green", "-"),
#        ("orange", "-"),
#        ("black", "--"),
#    ],
#    ylabel="Time (ms)",
#    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}",
#    args={
#        "H": N_HEADS,
#        "BATCH": BATCH,
#        "HEAD_DIM": HEAD_DIM,
#    },
#)


# @triton.testing.perf_report(config)
# def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, mode, device=DEVICE):
#    dtype = torch.float32
#    q = torch.randn((BATCH, H, 1, HEAD_DIM), dtype=dtype, device=device)
#    k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
#    v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
#    attn_mask = torch.arange(0, N_CTX, device=device).unsqueeze(0).unsqueeze(0) < N_CTX // 2
#    if mode == -1:
#        fn = lambda: sdpa_attn(q, k, v, attn_mask=attn_mask)
#    else:
#        threshold = get_threshold(q, k, v, attn_mask=attn_mask, percentile=mode).reshape((BATCH, H)).to(device=device, dtype=dtype)
#        fn = lambda: thresh_attn_fused(q, k, v, threshold, attn_mask=attn_mask)
#
#    ms = triton.testing.do_bench(fn)
#    return ms
#
# if __name__ == "__main__":
#    #test()
#    bench_flash_attention.run(save_path=".", print_data=True)
