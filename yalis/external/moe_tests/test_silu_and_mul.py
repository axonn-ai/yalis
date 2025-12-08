import math
import torch
import moe_ops
# Adjust this if you used a different namespace
# e.g. "_C", "activation_ops", etc.
OP_NAMESPACE = "moe_ops"
OP_NAME = "silu_and_mul"


def get_op():
    """Fetch the custom op from torch.ops.<namespace>.<name>."""
    ns = getattr(torch.ops, OP_NAMESPACE)
    return getattr(ns, OP_NAME)


def silu_and_mul_ref(x: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation:

        out = silu(x[..., :d]) * x[..., d:]

    where d = x.shape[-1] // 2
    """
    assert x.shape[-1] % 2 == 0, "Last dimension must be even for silu_and_mul"
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.nn.functional.silu(x1) * x2


def run_single_case(shape, device, dtype, rtol=1e-4, atol=1e-4):
    """
    Run a single numeric correctness test for silu_and_mul
    on a given shape/device/dtype.
    """
    print(f"\n=== Test case: shape={shape}, device={device}, dtype={dtype} ===")

    x = torch.randn(shape, device=device, dtype=dtype)
    d = shape[-1] // 2

    # Output tensor: (..., d)
    out = torch.empty(*shape[:-1], d, device=device, dtype=dtype)

    op = get_op()
    op(out, x)

    ref = silu_and_mul_ref(x)

    print("Output (first few elements):", out.flatten()[:8])
    print("Ref    (first few elements):", ref.flatten()[:8])

    # Basic shape check
    assert out.shape == ref.shape, f"Shape mismatch: out={out.shape}, ref={ref.shape}"

    # Numerical check
    if dtype in (torch.float16, torch.bfloat16):
        # Looser tolerances for low-precision dtypes
        rtol = max(rtol, 1e-2)
        atol = max(atol, 1e-3)

    if not torch.allclose(out, ref, rtol=rtol, atol=atol):
        max_abs_diff = (out - ref).abs().max().item()
        max_rel_diff = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-8)
        raise AssertionError(
            f"silu_and_mul mismatch: max_abs_diff={max_abs_diff}, "
            f"max_rel_diff={max_rel_diff}, rtol={rtol}, atol={atol}"
        )

    print("silu_and_mul matches reference for this case.")


def test_silu_and_mul():
    # This kernel is typically CUDA-only; skip if no GPU.
    if not torch.cuda.is_available():
        print("CUDA not available; silu_and_mul test skipped (CUDA-only op).")
        return

    device = "cuda"

    # A few shapes to cover:
    # - simple 2D [tokens, 2*d]
    # - 3D [batch, tokens, 2*d]
    # - weird non-power-of-two dimension
    d = 16

    shapes = [
        (8, 2 * d),         # [tokens, 2*d]
        (4, 8, 2 * d),      # [batch, tokens, 2*d]
        (3, 5, 2 * 13),     # [batch, tokens, 2*13] just to avoid only powers of 2
    ]

    dtypes = [torch.float16, torch.float32]

    for shape in shapes:
        assert shape[-1] % 2 == 0
        for dtype in dtypes:
            run_single_case(shape, device=device, dtype=dtype)


if __name__ == "__main__":
    test_silu_and_mul()
