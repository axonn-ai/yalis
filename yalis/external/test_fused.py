import math
import torch

# Adjust this import to wherever you put fused_moe
# e.g. from yalis.external.ops.fused_moe import fused_moe
from fused_moe import fused_moe   # <- change if needed


def naive_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> torch.Tensor:
    """
    Pure-PyTorch reference implementation for the simplest fused_moe case:
    - no grouped_topk
    - no custom routing
    - no quantization
    - standard SiLU-and-mul MLP per expert

    Shapes follow the vLLM/RedHat fused_moe conventions:
      hidden_states: [M, H]
      gating_output: [M, E]
      w1: [E, 2D, H]    (gate+up: output dim 2D, input dim H)
      w2: [E, H, D]     (down: output dim H, input dim D)
    """
    assert hidden_states.dim() == 2
    assert gating_output.dim() == 2

    M, H = hidden_states.shape
    E = gating_output.shape[1]
    Ew1, N2, Hw1 = w1.shape
    Ew2, Hw2, D = w2.shape

    assert Ew1 == Ew2 == E, "num_experts mismatch between w1/w2/gating"
    assert Hw1 == Hw2 == H, "hidden size mismatch"
    assert N2 == 2 * D, "w1 second dim must be 2 * intermediate size"

    # 1) Router: softmax over experts, then topk
    probs = torch.softmax(gating_output, dim=-1)           # [M, E]
    topk_weights, topk_ids = torch.topk(probs, k=topk, dim=-1)  # [M, topk]

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    # 2) For each token, mix experts
    out = torch.zeros_like(hidden_states)  # [M, H]

    for i in range(M):
        x = hidden_states[i]  # [H]
        for j in range(topk):
            weight = topk_weights[i, j]
            expert_id = int(topk_ids[i, j].item())

            # Expert weights
            w1_e = w1[expert_id]  # [2D, H]
            w2_e = w2[expert_id]  # [H, D]

            # First projection: [H] @ [H, 2D] -> [2D]
            h1 = torch.matmul(x, w1_e.t())  # [2D]

            # SiLU-and-mul split: [2D] -> ([D], [D]) -> [D]
            gate, up = h1.chunk(2, dim=-1)  # each [D]
            h2 = torch.nn.functional.silu(gate) * up  # [D]

            # Second projection: [D] -> [H]
            h3 = torch.matmul(w2_e, h2)  # [H]

            out[i] += weight * h3

    return out


def test_fused_moe_simple():
    """
    Tests the simplest fused_moe path:
      - use_grouped_topk = False
      - custom_routing_function = None
      - no quantization flags
    by comparing against naive_moe.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping fused_moe test.")
        return

    device = torch.device("cuda")
    dtype = torch.float16  # matches typical usage

    # Small but non-trivial sizes
    M = 7   # num tokens
    H = 16  # hidden size
    E = 4   # num experts
    D = 8   # intermediate size per expert (so w1 second dim = 2*D)
    topk = 2
    renormalize = True

    # Inputs
    hidden_states = torch.randn(M, H, device=device, dtype=dtype).contiguous()
    gating_output = torch.randn(M, E, device=device, dtype=dtype).contiguous()

    # Expert weights:
    # w1: [E, 2D, H], w2: [E, H, D]
    w1 = torch.randn(E, 2 * D, H, device=device, dtype=dtype).contiguous()
    w2 = torch.randn(E, H, D, device=device, dtype=dtype).contiguous()

    # Reference
    ref_out = naive_moe(
        hidden_states=hidden_states.to(torch.float32),  # do math in fp32 for ref
        w1=w1.to(torch.float32),
        w2=w2.to(torch.float32),
        gating_output=gating_output.to(torch.float32),
        topk=topk,
        renormalize=renormalize,
    ).to(dtype)

    # Fused
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

    assert fused_out.shape == ref_out.shape

    # Because fused path may be fp16 + Triton, allow a relaxed tolerance
    assert torch.allclose(fused_out, ref_out, rtol=1e-2, atol=1e-2), (
        f"fused_moe output does not match naive reference.\n"
        f"max abs diff: {(fused_out - ref_out).abs().max().item()}"
    )

    print("fused_moe simple test passed")


if __name__ == "__main__":
    test_fused_moe_simple()
