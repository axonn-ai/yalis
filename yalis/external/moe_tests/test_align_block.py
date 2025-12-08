import torch
import moe_ops

OP_NAMESPACE = "moe_ops"           # e.g. "yalis_moe_ops" or "vllm"

# Op name you registered in TORCH_LIBRARY: m.def("topk_softmax(...")...
OP_NAME = "moe_align_block_size"
# ------------------------------------------------------------------


def get_op():
    # Import the extension so its registration runs

    # Access the op via torch.ops.<namespace>.<name>
    ns = getattr(torch.ops, OP_NAMESPACE)
    return getattr(ns, OP_NAME)


def compute_expected_padded_len(topk_ids: torch.Tensor,
                                num_experts: int,
                                block_size: int) -> torch.Tensor:
    """
    Pure-Py helper to compute how many padded tokens we *expect*
    moe_align_block_size to produce.

    topk_ids: [num_tokens, topk] (or any shape; we just flatten)
    """
    flat = topk_ids.view(-1)
    # count how many times each expert appears
    counts = torch.bincount(flat, minlength=num_experts)  # [num_experts]

    # pad each expert's count up to a multiple of block_size
    padded_counts = ((counts + block_size - 1) // block_size) * block_size

    total_padded = int(padded_counts.sum().item())
    return counts, padded_counts, total_padded


def test_moe_align_block_size():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_ids = torch.int32

    num_tokens = 5
    topk = 2
    num_experts = 4
    block_size = 4

    topk_ids = torch.randint(
        low=0,
        high=num_experts,
        size=(num_tokens, topk),
        dtype=dtype_ids,
        device=device,
    )

    print("=== Input topk_ids ===")
    print(topk_ids)

    counts, padded_counts, total_padded = compute_expected_padded_len(
        topk_ids, num_experts, block_size
    )

    print("\nExpert counts per expert:", counts.tolist())
    print("Padded counts per expert:", padded_counts.tolist())
    print("Expected total padded tokens:", total_padded)

    flat_len = topk_ids.numel()
    sentinel = flat_len  # must match vLLM / RedHat usage

    # sorted_token_ids: tokens + padding, prefilled with sentinel
    sorted_token_ids = torch.full(
        (total_padded,), fill_value=sentinel,
        dtype=torch.int32, device=device
    )

    # expert_ids: one per block
    num_blocks = total_padded // block_size
    experts_ids = torch.empty(
        (num_blocks,), dtype=torch.int32, device=device
    )

    num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    torch.ops.moe_ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
    )

    print("\n=== Outputs from moe_align_block_size ===")
    print("num_tokens_post_pad tensor:", num_tokens_post_pad)
    ntokens = int(num_tokens_post_pad.item())
    print("num_tokens_post_pad.item():", ntokens)
    print("sorted_token_ids:", sorted_token_ids)
    print("experts_ids:", experts_ids)

    # 1. Total padded tokens matches our expectation
    assert ntokens == total_padded, (
        f"num_tokens_post_pad ({ntokens}) != expected total_padded ({total_padded})"
    )

    # 2. Total padded tokens divisible by block_size
    assert ntokens % block_size == 0

    num_blocks = ntokens // block_size
    experts_ids_cpu = experts_ids[:num_blocks].cpu().tolist()
    sorted_cpu = sorted_token_ids[:ntokens].cpu().tolist()

    # 3. Per-block expert consistency:
    flat = topk_ids.view(-1)
    for b in range(num_blocks):
        e = experts_ids_cpu[b]
        block_idx_start = b * block_size
        block_idx_end = (b + 1) * block_size
        block_positions = sorted_cpu[block_idx_start:block_idx_end]
        real_positions = [i for i in block_positions if i < flat_len]

        # all real tokens in this block must belong to expert e
        for pos in real_positions:
            assert int(flat[pos].item()) == e, (
                f"Block {b} expects expert {e}, "
                f"but position {pos} has expert {int(flat[pos].item())}"
            )

    # 4. Every original token index appears at least once (ignoring padding)
    used_real_indices = {i for i in sorted_cpu if i < flat_len}
    assert used_real_indices == set(range(flat_len)), (
        f"Expected each of the {flat_len} original positions to appear once; "
        f"got {len(used_real_indices)} unique positions instead."
    )

    print("\nAll sanity checks passed ")

if __name__ == "__main__":
    test_moe_align_block_size()
