import sys
from pathlib import Path
import argparse
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from yalis.attention.nowmp_reference import nowmp_reference_attention
from yalis.attention.threshold_attention_nowmp import init_nowmp_state, nowmp_attention_forward

try:
    import triton
    import triton.testing as triton_testing
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False
    triton = None
    triton_testing = None


DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_PCTS = [0.0, 0.5, 0.75, 0.875, 0.95]


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
        return DEFAULT_SEQ_LENS
    parts = [p.strip() for p in seq_arg.split(",") if p.strip()]
    seqs = []
    for p in parts:
        if p.lower().endswith("k"):
            seqs.append(int(float(p[:-1]) * 1024))
        else:
            seqs.append(int(p))
    return seqs or DEFAULT_SEQ_LENS


def _parse_percentiles(pct_arg, fallback_pct):
    if pct_arg:
        parts = [p.strip() for p in pct_arg.split(",") if p.strip()]
        return [float(p) for p in parts]
    if fallback_pct is not None:
        return [float(fallback_pct)]
    return DEFAULT_PCTS


def _fmt_pct(value: float) -> str:
    return f"{value:g}"


def _time_fn(fn, warmup=10, iters=50):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench_fn(fn, warmup=10, iters=50):
    if HAVE_TRITON:
        return triton_testing.do_bench(fn, warmup=warmup, rep=iters)
    return _time_fn(fn, warmup=warmup, iters=iters)


def _maybe_compile(fn, enabled):
    if not enabled:
        return fn
    try:
        return torch.compile(fn, mode="max-autotune-no-cudagraphs")
    except Exception:
        return fn


def _flash_attn(q, k, v):
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=False
        )


def _get_state_alpha_b(state):
    if "packed" in state:
        alpha = state["packed"][..., 0]
        b = state["packed"][..., 1]
        return alpha, b
    return state["alpha"], state["b"]


def run_correctness(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    torch.manual_seed(0)
    device = torch.device("cuda")

    B, H, Q, K, D = args.batch, args.heads, args.query_len, args.key_len, args.head_dim
    percentiles = _parse_percentiles(args.percentiles, args.percentile)

    query = torch.randn((B, H, Q, D), device=device, dtype=torch.float16)
    key = torch.randn((B, H, K, D), device=device, dtype=torch.float16)
    value = torch.randn((B, H, K, D), device=device, dtype=torch.float16)

    scale_factor = 1.0 / (D ** 0.5)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale_factor
    q_idx = torch.arange(Q, device=device)
    k_idx = torch.arange(K, device=device)
    causal = k_idx[None, :] > q_idx[:, None]
    scores_masked = scores.masked_fill(causal[None, None, :, :], float("-inf"))
    attn_weight = torch.softmax(scores_masked, dim=-1)

    trace_steps = {0, 3, 7, 15, 31, 63, 127, 255}
    trace_steps = {q for q in trace_steps if q < Q}

    for percentile in percentiles:
        print(f"percentile={_fmt_pct(percentile)}")
        ref = nowmp_reference_attention(
            attn_weight=attn_weight,
            value=value,
            percentile=percentile,
            return_traces=True,
            logits=scores_masked,
            emulate_kernel=True,
        )
        out_ref = ref["output"]

        state = init_nowmp_state(B, H, device=device)
        outputs = []
        retain_sum = torch.zeros((), device=device, dtype=torch.float32)
        alpha_snap = {}
        b_snap = {}

        for q in range(Q):
            t = q + 1
            if q in trace_steps:
                alpha, b = _get_state_alpha_b(state)
                alpha_snap[q] = alpha.clone()
                b_snap[q] = b.clone()
            q_step = query[:, :, q:q+1, :].contiguous()
            token_counter = torch.full((B,), q, device=device, dtype=torch.int32)
            mask = torch.arange(K, device=device).unsqueeze(0) <= q
            out, keep_counts = nowmp_attention_forward(
                query=q_step,
                key=key,
                value=value,
                token_counter=token_counter,
                state=state,
                percentile=percentile,
                attn_mask=mask[:, None, None, :],
            )
            outputs.append(out)
            retain_sum += (keep_counts / t).mean()

        out_kernel = torch.cat(outputs, dim=2)
        diff = (out_ref.float() - out_kernel.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        retain_mean_kernel = (retain_sum / Q).item()
        retain_mean_ref = ref["retain_mean"].item()

        print(f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
        print(f"retain_mean_ref={retain_mean_ref:.6f} retain_mean_kernel={retain_mean_kernel:.6f}")

        alpha_ref = ref["traces"]["alpha"]
        b_ref = ref["traces"]["b"]
        for q in sorted(trace_steps):
            alpha_err = (alpha_ref[:, :, q] - alpha_snap[q]).abs().max().item()
            b_err = (b_ref[:, :, q] - b_snap[q]).abs().max().item()
            print(f"step={q}: alpha_err={alpha_err:.6f} b_err={b_err:.6f}")

        assert max_diff < args.diff_tol, "Output mismatch exceeds tolerance"
        assert abs(retain_mean_ref - retain_mean_kernel) < args.retain_tol, "Retain mean mismatch"


def run_bench(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = torch.device("cuda")
    seq_lens = _parse_seq_lens(sys.argv)
    B, H, D = args.batch, args.heads, args.head_dim
    percentiles = _parse_percentiles(args.percentiles, args.percentile)

    if HAVE_TRITON and args.perf_report:
        pct_vals = list(percentiles) + [-1.0]
        line_names = [f"CUDA (Percentile: {_fmt_pct(p)})" for p in percentiles] + ["Torch"]
        styles = [
            ("red", "-"),
            ("blue", "-"),
            ("green", "-"),
            ("orange", "-"),
            ("purple", "-"),
            ("black", "--"),
        ]
        config = triton_testing.Benchmark(
            x_names=["N_CTX"],
            x_vals=seq_lens,
            line_arg="mode",
            line_vals=pct_vals,
            line_names=line_names,
            styles=styles[: len(line_names)],
            ylabel="Time (ms)",
            plot_name=f"nowmp-attn-batch{B}-head{H}-d{D}",
            args={
                "H": H,
                "BATCH": B,
                "HEAD_DIM": D,
            },
        )

        @triton_testing.perf_report(config)
        def bench_nowmp(BATCH, H, N_CTX, HEAD_DIM, mode, device=device):
            if hasattr(torch, "compiler"):
                torch.compiler.reset()
            dtype = torch.float16
            q = torch.randn((BATCH, H, 1, HEAD_DIM), device=device, dtype=dtype)
            k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=device, dtype=dtype)
            v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), device=device, dtype=dtype)
            token_counter = torch.full((BATCH,), N_CTX - 1, device=device, dtype=torch.int32)

            if mode < 0:
                fn = _maybe_compile(lambda: _flash_attn(q, k, v), args.compile)
                return _bench_fn(fn, warmup=args.warmup, iters=args.iters)

            state = init_nowmp_state(BATCH, H, device=device)
            fn = _maybe_compile(
                lambda: nowmp_attention_forward(
                    query=q,
                    key=k,
                    value=v,
                    token_counter=token_counter,
                    state=state,
                    percentile=float(mode),
                    attn_mask=None,
                ),
                args.compile,
            )
            return _bench_fn(fn, warmup=args.warmup, iters=args.iters)

        bench_nowmp.run(print_data=True)
        return

    pct_labels = [f"CUDA (Percentile: {_fmt_pct(p)})" for p in percentiles]
    header = "N_CTX, " + ", ".join(pct_labels) + ", Torch"
    print(header)
    for T in seq_lens:
        q = torch.randn((B, H, 1, D), device=device, dtype=torch.float16)
        k = torch.randn((B, H, T, D), device=device, dtype=torch.float16)
        v = torch.randn((B, H, T, D), device=device, dtype=torch.float16)
        token_counter = torch.full((B,), T - 1, device=device, dtype=torch.int32)
        fn_flash = _maybe_compile(lambda: _flash_attn(q, k, v), args.compile)

        nowmp_ms_vals = []
        for percentile in percentiles:
            state = init_nowmp_state(B, H, device=device)
            fn_nowmp = _maybe_compile(
                lambda: nowmp_attention_forward(
                    query=q,
                    key=k,
                    value=v,
                    token_counter=token_counter,
                    state=state,
                    percentile=percentile,
                    attn_mask=None,
                ),
                args.compile,
            )
            nowmp_ms_vals.append(_bench_fn(fn_nowmp, warmup=args.warmup, iters=args.iters))

        flash_ms = _bench_fn(fn_flash, warmup=args.warmup, iters=args.iters)
        nowmp_cols = ", ".join(f"{ms:.4f}" for ms in nowmp_ms_vals)
        print(f"{T}, {nowmp_cols}, {flash_ms:.4f}")


def main():
    parser = argparse.ArgumentParser(description="nowmp correctness + bench")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--query-len", type=int, default=256)
    parser.add_argument("--key-len", type=int, default=4096)
    parser.add_argument("--percentile", type=float, default=None)
    parser.add_argument("--percentiles", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--diff-tol", type=float, default=1e-2)
    parser.add_argument("--retain-tol", type=float, default=1e-3)
    parser.add_argument("--perf-report", action="store_true", default=None)
    parser.add_argument("--no-perf-report", dest="perf_report", action="store_false")
    parser.add_argument("--compile", action="store_true", default=None)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    args = parser.parse_args()
    if args.perf_report is None:
        args.perf_report = HAVE_TRITON
    if args.compile is None:
        args.compile = HAVE_TRITON

    if not args.no_check:
        run_correctness(args)
    if not args.no_bench:
        run_bench(args)


if __name__ == "__main__":
    main()
