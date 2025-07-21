import pytest
import torch.distributed as dist
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis import ModelConfig, InferenceConfig, LLMEngine, SpeculativeLLMEngine
from types import SimpleNamespace
from tests.sample_dataset import AlpacaDataset


def pytest_addoption(parser):
    parser.addini(
        "model",
        "Model to use for the test",
        type="string",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    parser.addini(
        "dtype", "Data type to use for the test", type="string", default="bf16"
    )
    parser.addini(
        "attn_backend",
        "Attention backend to use for the test",
        type="string",
        default="sdpa",
    )
    parser.addini(
        "draft_model",
        "Draft model to use for Speculative Decoding tests",
        type="string",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )


@pytest.fixture(scope="module", autouse=True)
def cleanup_dist():
    yield
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception as e:
            print(f"[conftest]: Error in barrier: {e}")
        dist.destroy_process_group()


@pytest.fixture(scope="session", autouse=True)
def device():
    return "cuda"


@pytest.fixture(scope="module")
def model_id(request):
    """Get the model ID from pytest config."""
    return request.config.getini("model")


@pytest.fixture(scope="module")
def draft_model_id(request):
    """Get the draft model ID from pytest config."""
    return request.config.getini("draft_model")


@pytest.fixture(scope="module")
def dtype(request):
    """Get the data type configuration for both Yalis and HF."""
    dt = request.config.getini("dtype").lower()

    yalis_dt = dt

    hf_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    hf_dt = hf_map[dt]

    return SimpleNamespace(yalis=yalis_dt, hf=hf_dt)


@pytest.fixture(scope="module")
def attn_backend(request):
    """Get the attention backend configuration for both Yalis and HF."""
    attnb = request.config.getini("attn_backend").lower()
    yalis_attnb = attnb

    hf_map = {
        "sdpa": "sdpa",
        "flash": "flash_attention_2",
        # For some reason, flex does not work with in hf right now
        "flex": "flash_attention_2",
    }

    hf_attnb = hf_map[attnb]

    return SimpleNamespace(yalis=yalis_attnb, hf=hf_attnb)


# Dataset and tokenizer fixtures
@pytest.fixture(scope="module")
def tokenizer(model_id):
    """Create a tokenizer for the test model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(scope="module")
def alpaca_dataset():
    """Create an Alpaca dataset for testing."""
    dataset = AlpacaDataset(random_seed=42)
    return dataset


# Model fixtures
@pytest.fixture(scope="module")
def hf_model(model_id, dtype, attn_backend, device):
    """Create a HuggingFace model for comparison testing."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_backend.hf,
        dtype=dtype.hf,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def yalis_engine(model_id, dtype, attn_backend):
    """Create a standard Yalis LLMEngine."""
    model_config = ModelConfig(model_name=model_id, precision=dtype.yalis)
    inference_config = InferenceConfig(
        # max batch size for dynamic batching
        max_batch_size=8,  # Set to max batch size from test
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend.yalis,
        use_paged_kv_caching=False,
    )
    return LLMEngine(
        model_config=model_config, inference_config=inference_config
    )


@pytest.fixture(scope="module")
def speculative_engine(model_id, draft_model_id, dtype, attn_backend):
    """Create a SpeculativeLLMEngine for testing."""
    target_model_config = ModelConfig(
        model_name=model_id, precision=dtype.yalis
    )
    draft_model_config = ModelConfig(
        model_name=draft_model_id, precision=dtype.yalis
    )
    inference_config = InferenceConfig(
        # initial batch size, will be changed with reset_kv_cache
        max_batch_size=8,
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend.yalis,
        use_paged_kv_caching=False,
    )
    return SpeculativeLLMEngine(
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        inference_config=inference_config,
    )
