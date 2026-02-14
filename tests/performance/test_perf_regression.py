"""
Performance regression tests for YALIS.

Workflow
--------
1. On the develop (baseline) branch, generate baselines::

       pytest tests/performance/ --perf-update-baselines

2. On your feature branch, run the tests to check for regressions::

       pytest tests/performance/

   Any metric that regresses beyond the tolerance (default 10 %)
   will cause the test to fail with a detailed report.

Options
-------
--perf-tolerance FLOAT     Allowed regression fraction (default 0.10).
--perf-warmup-iters INT    Warmup iterations (default 3).
--perf-measure-iters INT   Measurement iterations (default 5).
--perf-baseline-path PATH  Path to the baselines JSON file.
"""

import pytest
import torch.distributed as dist

from tests.basic_correctness.utils import alpaca_prompt

BATCH_SIZES = [1, 8]
PROMPT_LENGTHS = [128, 512]
DECODE_LENGTHS = [32, 128]

# Metrics where *lower* is better (latencies).
_LOWER_IS_BETTER = {"ttft_ms", "tbt_ms", "e2e_ms"}
# Metrics where *higher* is better (throughput).
_HIGHER_IS_BETTER = {"throughput_tps"}

_ALL_METRICS = [
    ("ttft_ms", "TTFT"),
    ("tbt_ms", "TBT"),
    ("throughput_tps", "Throughput"),
    ("e2e_ms", "E2E"),
]


def _bench_key(batch_size, prompt_length, decode_length):
    return (
        f"batch_{batch_size}"
        f"_prompt_{prompt_length}"
        f"_decode_{decode_length}"
    )


def _run_iterations(engine, prompts, decode_length, n_iters):
    """Run *n_iters* generate calls and return the list of metric dicts."""
    collected = []
    for _ in range(n_iters):
        _, metrics = engine.generate(
            prompts,
            report_throughput=False,
            tokens_to_generate=decode_length,
            ignore_eos=True,
        )
        collected.append(metrics)
    return collected


def _average_metrics(metrics_list):
    n = len(metrics_list)
    return {
        "ttft_ms": sum(m["TTFT"] for m in metrics_list) / n,
        "tbt_ms": sum(m["TBT"] for m in metrics_list) / n,
        "throughput_tps": sum(m["Throughput"] for m in metrics_list) / n,
        "e2e_ms": sum(m["E2E"] for m in metrics_list) / n,
    }


def _check_regressions(baseline, current, tolerance):
    """Return a list of (metric, baseline_val, current_val, pct) tuples
    for every metric that regressed beyond *tolerance*."""
    regressions = []
    for key, label in _ALL_METRICS:
        base_val = baseline[key]
        curr_val = current[key]

        if base_val == 0:
            continue

        if key in _LOWER_IS_BETTER:
            pct = (curr_val - base_val) / base_val
            regressed = pct > tolerance
        else:
            pct = (base_val - curr_val) / base_val
            regressed = pct > tolerance
            pct = -pct  # show as negative when throughput drops

        if regressed:
            regressions.append((label, base_val, curr_val, pct))

    return regressions


def _format_report(key, current, baseline, regressions, tolerance):
    """Build a human-readable report string."""
    lines = [f"Performance regression detected for [{key}]:"]
    lines.append("")
    lines.append(
        f"  {'Metric':<14} {'Baseline':>12} {'Current':>12} {'Change':>10}"
    )
    lines.append(f"  {'-'*50}")

    for mkey, label in _ALL_METRICS:
        base_val = baseline[mkey]
        curr_val = current[mkey]
        if base_val != 0:
            pct = (curr_val - base_val) / base_val
            marker = (
                " << REGRESSION"
                if any(r[0] == label for r in regressions)
                else ""
            )
            lines.append(
                f"  {label:<14} {base_val:>12.4f} {curr_val:>12.4f}"
                f" {pct:>+9.1%}{marker}"
            )
        else:
            lines.append(
                f"  {label:<14} {base_val:>12.4f} {curr_val:>12.4f}"
                f"       N/A"
            )

    lines.append("")
    lines.append(f"  Tolerance: {tolerance:.0%}")
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Tests                                                              #
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.parametrize("decode_length", DECODE_LENGTHS)
def test_perf_regression(
    perf_engine,
    tokenizer,
    alpaca_dataset,
    baseline_store,
    perf_results,
    batch_size,
    prompt_length,
    decode_length,
    request,
):
    config = request.config
    update_mode = config.getoption("--perf-update-baselines")
    tolerance = config.getoption("--perf-tolerance")
    warmup_iters = config.getoption("--perf-warmup-iters")
    measure_iters = config.getoption("--perf-measure-iters")

    key = _bench_key(batch_size, prompt_length, decode_length)

    # --- prepare prompts ------------------------------------------ #
    prompts = alpaca_prompt(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )

    # --- warmup --------------------------------------------------- #
    _run_iterations(perf_engine, prompts, decode_length, warmup_iters)

    # --- measure -------------------------------------------------- #
    raw = _run_iterations(perf_engine, prompts, decode_length, measure_iters)
    current = _average_metrics(raw)

    # Only rank 0 performs the baseline comparison / update.
    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # --- update mode: store and return ---------------------------- #
    if update_mode:
        baseline_store.put(
            key,
            {
                "batch_size": batch_size,
                "prompt_length": prompt_length,
                "decode_length": decode_length,
                **current,
            },
        )
        perf_results.append(
            {
                "key": key,
                "metrics": [
                    (label, current[mkey]) for mkey, label in _ALL_METRICS
                ],
            }
        )
        return

    # --- compare mode --------------------------------------------- #
    baseline = baseline_store.get(key)
    if baseline is None:
        pytest.skip(
            f"No baseline for {key}. "
            "Run with --perf-update-baselines first."
        )

    regressions = _check_regressions(baseline, current, tolerance)

    comparisons = []
    for mkey, label in _ALL_METRICS:
        base_val = baseline[mkey]
        curr_val = current[mkey]
        pct = (curr_val - base_val) / base_val if base_val != 0 else 0
        comparisons.append((label, base_val, curr_val, pct))
    perf_results.append({"key": key, "comparisons": comparisons})

    if regressions:
        report = _format_report(key, current, baseline, regressions, tolerance)
        pytest.fail(report)
