# YALIS Tests

## Performance Regression Tests

The performance tests live in `tests/performance/` and measure TTFT, TBT,
end-to-end latency, and throughput across a matrix of batch sizes, prompt
lengths, and decode lengths.

### Quick start

Performance testing is a two-step workflow: first record baselines on a
known-good branch, then compare against those baselines on a feature branch.

#### 1. Generate baselines (on the develop / baseline branch)

```bash
PERF_PYTEST_ARGS="--perf-update-baselines" ./tests/scripts/run_perf_regression_tests.sh
```

This runs every benchmark combination and writes the results to
`tests/performance/baselines/perf_baselines.json`.

#### 2. Check for regressions (on your feature branch)

```bash
./tests/scripts/run_perf_regression_tests.sh
```

Any metric that regresses beyond the tolerance (default 10 %) will cause the
test to fail with a detailed report.

### Configuration

The script uses `srun` to launch on a Slurm cluster. You can control the
number of GPUs with the `GPUS` environment variable:

```bash
GPUS=4 ./tests/scripts/run_perf_regression_tests.sh
```

Additional pytest options can be passed through the `PERF_PYTEST_ARGS`
environment variable:

| Option                        | Default                                         | Description                                       |
| ----------------------------- | ----------------------------------------------- | ------------------------------------------------- |
| `--perf-update-baselines`     | off                                             | Record new baselines instead of comparing.        |
| `--perf-tolerance FLOAT`      | `0.10`                                          | Max allowed regression fraction (10 %).           |
| `--perf-warmup-iters INT`     | `3`                                             | Warmup iterations before measurement.             |
| `--perf-measure-iters INT`    | `5`                                             | Measurement iterations for averaging.             |
| `--perf-baseline-path PATH`   | `tests/performance/baselines/perf_baselines.json` | Path to the baselines JSON file.                |

Example — tighter tolerance with more measurement iterations:

```bash
PERF_PYTEST_ARGS="--perf-tolerance 0.05 --perf-measure-iters 10" \
  ./tests/scripts/run_perf_regression_tests.sh
```

The model, precision, and attention backend are configured via pytest ini
settings. The defaults (set in `tests/performance/conftest.py`) are:

| Setting                  | Default                              |
| ------------------------ | ------------------------------------ |
| `model`                  | `meta-llama/Llama-3.1-8B-Instruct`  |
| `dtype`                  | `bf16`                               |
| `attn_backend`           | `sdpa`                               |
| `use_paged_kv_caching`   | `False`                              |

To override these, create a `pytest.ini` (or add an `[pytest]` section to
`pyproject.toml`) with the desired values, or pass a custom `-c <ini-file>`
through `PERF_PYTEST_ARGS`.

## Correctness Tests

<!-- TODO: Add correctness test documentation -->
