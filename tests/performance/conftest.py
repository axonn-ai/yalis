import json
import os
import subprocess
from datetime import datetime, timezone

import pytest
import torch.distributed as dist
from transformers import AutoTokenizer

from yalis import ModelConfig, InferenceConfig, LLMEngine
from tests.sample_dataset import AlpacaDataset

BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baselines")
DEFAULT_BASELINE_PATH = os.path.join(BASELINE_DIR, "perf_baselines.json")


# ------------------------------------------------------------------ #
#  CLI options                                                        #
# ------------------------------------------------------------------ #


def pytest_addoption(parser):
    parser.addini(
        "model",
        "Model to use for the test",
        type="string",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.addini(
        "dtype",
        "Data type to use for the test",
        type="string",
        default="bf16",
    )
    parser.addini(
        "attn_backend",
        "Attention backend to use for the test",
        type="string",
        default="sdpa",
    )
    parser.addoption(
        "--perf-update-baselines",
        action="store_true",
        default=False,
        help="Update performance baselines instead of comparing.",
    )
    parser.addoption(
        "--perf-tolerance",
        type=float,
        default=0.10,
        help="Max allowed regression fraction (default: 0.10 = 10%%).",
    )
    parser.addoption(
        "--perf-warmup-iters",
        type=int,
        default=3,
        help="Warmup iterations before measurement (default: 3).",
    )
    parser.addoption(
        "--perf-measure-iters",
        type=int,
        default=5,
        help="Measurement iterations for averaging (default: 5).",
    )
    parser.addoption(
        "--perf-baseline-path",
        type=str,
        default=DEFAULT_BASELINE_PATH,
        help="Path to the baseline JSON file.",
    )


# ------------------------------------------------------------------ #
#  Baseline store                                                     #
# ------------------------------------------------------------------ #


class BaselineStore:
    """Thin wrapper around a JSON file that holds perf baselines."""

    def __init__(self, path):
        self.path = path
        self._data = self._load()

    # -- persistence ------------------------------------------------ #

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"metadata": {}, "benchmarks": {}}

    def flush(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self._data, f, indent=2)

    # -- read / write ----------------------------------------------- #

    def get(self, key):
        return self._data["benchmarks"].get(key)

    def put(self, key, entry):
        self._data["benchmarks"][key] = entry

    def set_metadata(self, **kwargs):
        self._data["metadata"].update(kwargs)


# ------------------------------------------------------------------ #
#  Fixtures                                                           #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module", autouse=True)
def cleanup_dist():
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


@pytest.fixture(scope="session")
def model_id(request):
    return request.config.getini("model")


@pytest.fixture(scope="session")
def dtype(request):
    return request.config.getini("dtype").lower()


@pytest.fixture(scope="session")
def attn_backend(request):
    return request.config.getini("attn_backend").lower()


@pytest.fixture(scope="module")
def perf_engine(model_id, dtype, attn_backend):
    """LLMEngine configured for performance measurement."""
    model_config = ModelConfig(model_name=model_id, precision=dtype)
    inference_config = InferenceConfig(
        max_batch_size=8,
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend,
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
    )
    return LLMEngine(
        model_config=model_config,
        inference_config=inference_config,
    )


@pytest.fixture(scope="session")
def tokenizer(model_id):
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="session")
def alpaca_dataset():
    return AlpacaDataset(random_seed=42)


@pytest.fixture(scope="session")
def baseline_store(request, model_id, dtype, attn_backend):
    """Load (or create) the baseline store and flush on teardown."""
    path = request.config.getoption("--perf-baseline-path")
    store = BaselineStore(path)

    update = request.config.getoption("--perf-update-baselines")
    if update:
        git_sha = _git_sha()
        store.set_metadata(
            model=model_id,
            attention_backend=attn_backend,
            precision=dtype,
            updated_at=datetime.now(timezone.utc).isoformat(),
            git_commit=git_sha,
        )

    yield store

    if update:
        store.flush()


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #


def _git_sha():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"
