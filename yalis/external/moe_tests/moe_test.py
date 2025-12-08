import math
import torch
import torch.nn.functional as F

# TODO: change this import to wherever your fused_moe lives
# from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
from fused_moe import fused_moe  # adjust this


def reference_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    """
    Pure PyTorch reference implementation that mirrors the fused_moe math:
    - softmax over experts
    - top-k routing
    - SwiGLU MLP per expert (using w1/w2 layout)
    - weighted sum over experts
    """
    # hidden_states: [N, D]
    # gating_output: [N, E]
    # w1: [E, 2H, D]
    # w2: [E, D, H]
    N, D = hidden_states.shape
    E, twoH, D_w1 = w1.shape
    assert D == D_w1, "hidden_states and w1 last dim mismatch"
    assert gating_output.shape == (N, E)

    H = twoH // 2
    assert twoH == 2 * H, "w1 middle dim should be 2 * hidden_dim"

    # 1) softmax over experts
    gate_probs = F.softmax(gating_output, dim=-1)  # [N, E]

    # 2) top-k per token
    topk_vals, topk_ids = torch.topk(gate_probs, k=topk, dim=-1)  # [N, K], [N, K]

    # 3) optional renormalization
    if renormalize:
        topk_weights = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    else:
        topk_weights = topk_vals

    # 4) compute expert outputs
    # We'll do a simple (not super optimized) loop for clarity
    out = torch.zeros_like(hidden_states)  # [N, D]

    # Pre-transpose weights for easier matmul
    # w1[e]: [2H, D] -> [D, 2H]
    w1_t = w1.transpose(1, 2)        # [E, D, 2H]
    # w2[e]: [D, H] -> [H, D]
    w2_t = w2.transpose(1, 2)        # [E, H, D]

    for i in range(N):
        x_i = hidden_states[i]       # [D]
        for k in range(topk):
            e = topk_ids[i, k].item()
            alpha = topk_weights[i, k]

            # first linear: x_i @ W1_e  -> [2H]
            pre = x_i @ w1_t[e]      # [2H]
            gate, up = pre.chunk(2, dim=-1)  # [H], [H]

            # SwiGLU
            h = F.silu(gate) * up    # [H]

            # second linear: h @ W2_e -> [D]
            y_e = h @ w2_t[e]        # [D]

            out[i] += alpha * y_e

    return out


def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Small test config
    num_tokens = 16   # N
    d_model = 32      # D
    hidden_dim = 64   # H
    num_experts = 4   # E
    topk = 2
    renormalize = True

    # Create random inputs
    hidden_states = torch.randn(num_tokens, d_model, device=device, dtype=torch.float16)
    gating_output = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float16)

    # Create random expert weights with the expected layout
    # w1: [E, 2H, D]
    w1 = torch.randn(num_experts, 2 * hidden_dim, d_model, device=device, dtype=torch.float16)
    # w2: [E, D, H]
    w2 = torch.randn(num_experts, d_model, hidden_dim, device=device, dtype=torch.float16)

    # Run reference implementation (in higher precision to reduce numeric noise)
    ref_out = reference_moe(
        hidden_states.to(torch.float32),
        w1.to(torch.float32),
        w2.to(torch.float32),
        gating_output.to(torch.float32),
        topk=topk,
        renormalize=renormalize,
    ).to(hidden_states.dtype)

    # Run fused_moe, which internally uses fused_experts + Triton kernels
    fused_out = fused_moe(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        gating_output=gating_output,
        topk=topk,
        renormalize=renormalize,
        inplace=False,
        override_config=None,
        use_grouped_topk=False,
        num_expert_group=None,
        topk_group=None,
        custom_routing_function=None,
        use_fp8_w8a8=False,
        use_int8_w8a16=False,
        w1_scale=None,
        w2_scale=None,
        a1_scale=None,
        a2_scale=None,
    )

    # Compare
    max_diff = (fused_out - ref_out).abs().max().item()
    print(f"max absolute difference: {max_diff:.6f}")

    # tolerances can be tweaked depending on dtype and your env
    assert torch.allclose(fused_out, ref_out, atol=1e-2, rtol=1e-2), \
        "fused_moe output does not match reference_moe!"

    print("fused_moe matches reference_moe within tolerance.")


if __name__ == "__main__":
    main()
