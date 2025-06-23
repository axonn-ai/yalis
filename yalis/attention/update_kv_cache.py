import triton
import triton.language as tl
import torch


@triton.jit
def update_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    block_table_ptr,
    cache_seq_len_ptr,
    cache_k_ptr,
    cache_v_ptr,
    B,
    S,
    H,
    D,
    max_pages_per_seq,
    page_block_size,
    layout_mode: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= B * S:
        return

    # Token index
    b = pid // S
    s = pid % S

    # Offset in kv cache
    cache_offset = tl.load(cache_seq_len_ptr + b)
    token_offset = cache_offset + s
    page_id = token_offset // page_block_size
    offset_in_block = token_offset % page_block_size
    block_id = tl.load(block_table_ptr + b * max_pages_per_seq + page_id)

    # Shared across layouts
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    for h_offset in range(0, H, BLOCK_H):
        offs_h = tl.arange(0, BLOCK_H) + h_offset
        mask_h = offs_h < H

        offs_h_broadcast = offs_h[:, None]
        offs_d_broadcast = offs_d[None, :]

        token_idx = b * S + s
        k_src_ptrs = (
            k_ptr + token_idx * H * D + offs_h_broadcast * D + offs_d_broadcast
        )
        v_src_ptrs = (
            v_ptr + token_idx * H * D + offs_h_broadcast * D + offs_d_broadcast
        )

        if layout_mode == 0:  # this is what flash attention uses
            # Unified layout: [num_blocks, page_block_size, H, D]
            dst_base = ((block_id * page_block_size + offset_in_block) * H) * D
            dst_ptrs = dst_base + offs_h_broadcast * D + offs_d_broadcast

            k_dst_ptrs = cache_k_ptr + dst_ptrs
            v_dst_ptrs = cache_v_ptr + dst_ptrs

            k_vals = tl.load(
                k_src_ptrs,
                mask=mask_h[:, None] & mask_d[None, :],
            )
            v_vals = tl.load(
                v_src_ptrs,
                mask=mask_h[:, None] & mask_d[None, :],
            )
            tl.store(
                k_dst_ptrs,
                k_vals,
                mask=mask_h[:, None] & mask_d[None, :],
            )
            tl.store(
                v_dst_ptrs,
                v_vals,
                mask=mask_h[:, None] & mask_d[None, :],
            )

        else:  # this is what sdpa and flex attention use
            # Per-head layout: [H, num_blocks, page_block_size, D]
            for i in range(BLOCK_H):
                if offs_h[i] >= H:
                    continue
                h = offs_h[i]
                k_val = tl.load(k_src_ptrs[i, :], mask=mask_d)
                v_val = tl.load(v_src_ptrs[i, :], mask=mask_d)

                dst_offset = (
                    (block_id * page_block_size + offset_in_block) * D
                ) + offs_d
                k_dst_ptr = (
                    cache_k_ptr
                    + h * page_block_size * max_pages_per_seq * D
                    + dst_offset
                )
                v_dst_ptr = (
                    cache_v_ptr
                    + h * page_block_size * max_pages_per_seq * D
                    + dst_offset
                )

                tl.store(k_dst_ptr, k_val, mask=mask_d)
                tl.store(v_dst_ptr, v_val, mask=mask_d)


def update_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: torch.Tensor,
    cache_seq_len: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
):

    B, S, H, D = k.shape
    BLOCK_D = D
    BLOCK_H = min(1024 // BLOCK_D, H)

    grid = (B * S,)  # one program per token

    max_pages_per_seq = block_table.shape[1]
    page_block_size = k_cache.shape[1]

    update_paged_kv_cache_kernel[grid](
        k,
        v,
        block_table,
        cache_seq_len,
        k_cache,
        v_cache,
        B,
        S,
        H,
        D,
        max_pages_per_seq,
        page_block_size,
        layout_mode=0,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )
