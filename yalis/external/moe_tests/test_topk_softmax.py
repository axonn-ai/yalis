import torch

# ------------------------------------------------------------------
# EDIT THESE TO MATCH YOUR SETUP
# ------------------------------------------------------------------
# This should be the name you used in CUDAExtension(name="...").
EXTENSION_MODULE_NAME = "yalis_moe_ops"   # e.g. "yalis_moe_ops" or "vllm_moe_ops"

# This should be the namespace used in TORCH_LIBRARY(...)
# If you used TORCH_LIBRARY(EXTENSION_MODULE_NAME, m), they’re the same.
OP_NAMESPACE = "moe_ops"           # e.g. "yalis_moe_ops" or "vllm"

# Op name you registered in TORCH_LIBRARY: m.def("topk_softmax(...")...
OP_NAME = "topk_softmax"
# ------------------------------------------------------------------


def get_topk_softmax_op():
    # Import the extension so its registration runs
    __import__(EXTENSION_MODULE_NAME)

    # Access the op via torch.ops.<namespace>.<name>
    ns = getattr(torch.ops, OP_NAMESPACE)
    return getattr(ns, OP_NAME)


def reference_topk_softmax(gating_output: torch.Tensor, topk: int, renormalize: bool):
    """
    Pure PyTorch reference:
    - softmax over experts
    - select topk over expert dimension
    - optionally renormalize the topk weights per token
    """
    # gating_output: [num_tokens, num_experts]
    probs = torch.softmax(gating_output, dim=-1)  # [T, E]

    # Top-k along expert dimension
    ref_vals, ref_idx = torch.topk(probs, k=topk, dim=-1)  # [T, K] each

    if renormalize:
        denom = ref_vals.sum(dim=-1, keepdim=True)
        denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        ref_vals = ref_vals / denom

    # token_expert_indices in vLLM is a "source mapping" but for correctness
    # of weights/indices we don't really care about its exact values here.
    # We'll just return a dummy tensor of the right shape.
    T = gating_output.shape[0]
    dummy_token_expert_indices = torch.arange(T, device=gating_output.device).unsqueeze(-1).repeat(1, topk)

    return ref_vals, ref_idx, dummy_token_expert_indices


def run_single_test(
    num_tokens=8,
    num_experts=16,
    topk=2,
    dtype=torch.float32,
    device="cuda",
    renormalize=True,
):
    print(f"\n=== Test: T={num_tokens}, E={num_experts}, K={topk}, dtype={dtype}, device={device}, renorm={renormalize} ===")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but this op is CUDA-only.")

    op = get_topk_softmax_op()

    # Create random gating output
    gating_output = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)

    # Allocate output tensors (same shapes as vLLM topk_softmax expects)
    topk_weights = torch.empty(num_tokens, topk, device=device, dtype=torch.float32)
    topk_indices = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)
    token_expert_indices = torch.empty(num_tokens, topk, device=device, dtype=torch.int32)

    # Call custom CUDA op
    op(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize,
    )

    # Compute reference result in PyTorch
    ref_weights, ref_indices, _ = reference_topk_softmax(gating_output.to(torch.float32), topk, renormalize)

    # Compare
    # vLLM stores weights as float32 even if input is fp16/bf16
    atol = 1e-4 if dtype == torch.float32 else 5e-3
    rtol = 1e-3 if dtype == torch.float32 else 5e-2

    ok_weights = torch.allclose(topk_weights, ref_weights, atol=atol, rtol=rtol)
    ok_indices = torch.equal(topk_indices.to(ref_indices.dtype), ref_indices)

    print(f"topk_weights match reference: {ok_weights}")
    print(f"topk_indices match reference: {ok_indices}")
    if not ok_weights:
        print("  max abs diff:", (topk_weights - ref_weights).abs().max().item())
    if not ok_indices:
        print("  example indices (custom):   ", topk_indices[0].tolist())
        print("  example indices (reference):", ref_indices[0].tolist())

    return ok_weights and ok_indices


def main():
    device = "cuda"

    all_ok = True

    # Test a few configs, including power-of-2 experts (fused path) and a weird one (fallback path)
    tests = [
        dict(num_tokens=8, num_experts=16, topk=2, dtype=torch.float32),
        dict(num_tokens=8, num_experts=64, topk=4, dtype=torch.float32),
        dict(num_tokens=8, num_experts=192, topk=4, dtype=torch.float16),
        dict(num_tokens=8, num_experts=20, topk=2, dtype=torch.float32),  # triggers moeSoftmax + moeTopK path
    ]

    for cfg in tests:
        ok = run_single_test(device=device, renormalize=True, **cfg)
        all_ok = all_ok and ok

    print("\n===================================")
    print("ALL TESTS PASS" if all_ok else "SOME TESTS FAILED")
    print("===================================")


if __name__ == "__main__":
    main()
