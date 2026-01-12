import triton
import triton.language as tl
import torch
from typing import Optional

@triton.jit
def update_paged_kv_cache_kernel(
    k_ptr,
    v_ptr,
    slot_mapping_ptr,  # [num_tokens] - PRE-COMPUTED slot indices
    cache_k_ptr,
    cache_v_ptr,
    num_tokens,
    H,
    D,
    block_size,
    num_blocks,  # Add this parameter
    layout_mode: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Simplified version that uses pre-computed slot_mapping.
    This matches vLLM's approach.
    """
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    
    # Get pre-computed slot index
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        # Padding token - skip
        return
    
    # Compute block and offset from slot
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    # Shared across layouts
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    for h_offset in range(0, H, BLOCK_H):
        offs_h = tl.arange(0, BLOCK_H) + h_offset
        mask_h = offs_h < H

        offs_h_broadcast = offs_h[:, None]
        offs_d_broadcast = offs_d[None, :]

        # Source pointers (flattened format: total_tokens, H, D)
        k_src_ptrs = k_ptr + token_idx * H * D + offs_h_broadcast * D + offs_d_broadcast
        v_src_ptrs = v_ptr + token_idx * H * D + offs_h_broadcast * D + offs_d_broadcast

        if layout_mode == 0:  # Flash Attention layout: [num_blocks, block_size, H, D]
            # Unified layout
            dst_base = ((block_idx * block_size + block_offset) * H) * D
            dst_ptrs = dst_base + offs_h_broadcast * D + offs_d_broadcast

            k_dst_ptrs = cache_k_ptr + dst_ptrs
            v_dst_ptrs = cache_v_ptr + dst_ptrs

            k_vals = tl.load(k_src_ptrs, mask=mask_h[:, None] & mask_d[None, :])
            v_vals = tl.load(v_src_ptrs, mask=mask_h[:, None] & mask_d[None, :])

            tl.store(k_dst_ptrs, k_vals, mask=mask_h[:, None] & mask_d[None, :])
            tl.store(v_dst_ptrs, v_vals, mask=mask_h[:, None] & mask_d[None, :])

        else:  # SDPA/Flex Attention layout: [H, num_blocks, block_size, D]
            # Per-head layout
            for i in range(BLOCK_H):
                if offs_h[i] >= H:
                    continue
                h = offs_h[i]
                k_val = tl.load(k_src_ptrs[i, :], mask=mask_d)
                v_val = tl.load(v_src_ptrs[i, :], mask=mask_d)

                dst_offset = (block_idx * block_size + block_offset) * D + offs_d
                k_dst_ptr = cache_k_ptr + h * (num_blocks * block_size * D) + dst_offset
                v_dst_ptr = cache_v_ptr + h * (num_blocks * block_size * D) + dst_offset

                tl.store(k_dst_ptr, k_val, mask=mask_d)
                tl.store(v_dst_ptr, v_val, mask=mask_d)


def compute_slot_mapping(
    positions: torch.Tensor,  # [num_tokens] - flattened positions
    req_indices: torch.Tensor,  # [num_tokens] - which request each token belongs to
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_req]
    block_size: int,
) -> torch.Tensor:
    """
    Pre-compute slot_mapping from positions and block_table.
    This matches vLLM's BlockTable.compute_slot_mapping() logic.

    Returns:
        slot_mapping: [num_tokens] - slot indices for each token
    """
    num_tokens = positions.shape[0]
    slot_mapping = torch.zeros(num_tokens, dtype=torch.int64, device=positions.device)

    for token_idx in range(num_tokens):
        req_idx = req_indices[token_idx].item()
        pos = positions[token_idx].item()

        # Compute which block this position belongs to
        block_idx_in_seq = pos // block_size
        offset_in_block = pos % block_size

        # Get physical block number from block table
        block_number = block_table[req_idx, block_idx_in_seq].item()

        # Compute slot: block_number * block_size + offset_in_block
        slot_mapping[token_idx] = block_number * block_size + offset_in_block

    return slot_mapping


def update_paged_kv_cache(
    k: torch.Tensor,  # [total_tokens, H, D]
    v: torch.Tensor,  # [total_tokens, H, D]
    positions: torch.Tensor,  # [total_tokens] - token positions
    req_indices: torch.Tensor,  # [total_tokens] - request index for each token
    block_table: torch.Tensor,  # [num_reqs, max_blocks_per_req]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_size: int,
    layout_mode: int = 0,  # 0 = Flash, 1 = SDPA/Flex
):
    """
    Update paged KV cache with new K/V tensors.

    Args:
        k, v: Flattened format (total_tokens, H, D)
        positions: Token positions (total_tokens,)
        req_indices: Which request each token belongs to (total_tokens,)
        block_table: Block table mapping logical to physical blocks
        k_cache, v_cache: KV cache tensors
        block_size: Size of each cache block
        layout_mode: 0 for Flash Attention layout, 1 for SDPA/Flex layout
    """
    total_tokens, H, D = k.shape

    # Pre-compute slot_mapping (like vLLM does)
    slot_mapping = compute_slot_mapping(positions, req_indices, block_table, block_size)

    BLOCK_D = D
    BLOCK_H = min(1024 // BLOCK_D, H)

    grid = (total_tokens,)  # Launch with total_tokens

    # Compute num_blocks from cache shape
    # For Flash layout: [num_blocks, block_size, H, D]
    # For SDPA/Flex layout: [H, num_blocks, block_size, D]
    if layout_mode == 0:
        num_blocks = k_cache.shape[0]  # First dimension
    else:
        num_blocks = k_cache.shape[1]  # Second dimension

    update_paged_kv_cache_kernel[grid](
        k,
        v,
        slot_mapping,  # Pre-computed!
        k_cache,
        v_cache,
        total_tokens,
        H,
        D,
        block_size,
        num_blocks,  # Add this argument
        layout_mode=layout_mode,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
    )