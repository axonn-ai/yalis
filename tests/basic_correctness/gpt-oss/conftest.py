import os
import sys
from pathlib import Path
import pytest
import torch.distributed as dist
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis import ModelConfig, InferenceConfig, LLMEngine, SpeculativeLLMEngine
from types import SimpleNamespace
from tests.sample_dataset import AlpacaDataset

# Assume offline mode by default unless otherwise specified
HF_DATASETS_OFFLINE = os.environ.get("HF_DATASETS_OFFLINE", "1") == "1"

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

logger = logging.getLogger(__name__)

# Add parent test directory to sys.path so imports like 'from utils import ...' work
parent_test_dir = Path(__file__).parent.parent
if str(parent_test_dir) not in sys.path:
    sys.path.insert(0, str(parent_test_dir))


def pytest_addoption(parser):
    parser.addini(
        "model",
        "Model to use for the test",
        type="string",
        default="yalis/external/checkpoints/openai/gpt-oss-20b",
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
        default="yalis/external/checkpoints/openai/gpt-oss-20b",
    )


@pytest.fixture(scope="module", autouse=True)
def cleanup_dist():
    yield
    if dist.is_initialized():
        dist.barrier()
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
        "sdpa": "eager",  # Use eager as fallback for models that don't support SDPA yet (e.g., GptOssForCausalLM)
    }

    hf_attnb = hf_map[attnb]

    return SimpleNamespace(yalis=yalis_attnb, hf=hf_attnb)


# Dataset and tokenizer fixtures
@pytest.fixture(scope="function")
def tokenizer(model_id):
    """Create a tokenizer for the test model."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=HF_DATASETS_OFFLINE)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% endif %}{% endfor %}"
    
    return tokenizer


@pytest.fixture(scope="function")
def alpaca_dataset():
    """Create an Alpaca dataset for testing."""
    dataset = AlpacaDataset(random_seed=42)
    return dataset


# Model fixtures
@pytest.fixture(scope="function")
def hf_model(model_id, dtype, attn_backend, device):
    """Create a HuggingFace model for comparison testing."""
    # Disable MXFP4 CUDA kernels to prevent GPU-side dequantization attempts.
    os.environ["MXFP4_DISABLE_CUDA_KERNELS"] = "1"
    # Dequantize on CPU where memory is abundant, then move to rank-specific GPU.
    logger.info("Loading HF model on CPU for safe MXFP4 dequantization...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_backend.hf,
        dtype=dtype.hf,
        device_map="cpu",
        local_files_only=HF_DATASETS_OFFLINE,
        trust_remote_code=True,
    )
    # Move dequantized model to rank-specific GPU for inference.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    target_device = f"cuda:{local_rank}"
    logger.info(f"Moving model to {target_device} for inference...")
    model = model.to(target_device)
    model.eval()
    return model


@pytest.fixture(scope="function")
def yalis_engine(model_id, dtype, attn_backend):
    """Create a standard Yalis LLMEngine."""
    # Resolve model_path: if model_id is a relative path, make it absolute relative to repo root
    if not os.path.isabs(model_id):
        model_path = os.path.abspath(model_id)
    else:
        model_path = model_id
    model_name_for_config = os.path.basename(model_path)
    model_config = ModelConfig(model_name_for_config, model_path=model_path, precision=dtype.yalis)
    inference_config = InferenceConfig(
        max_batch_size=2,
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


@pytest.fixture(scope="function")
def speculative_engine(model_id, draft_model_id, dtype, attn_backend):
    """Create a SpeculativeLLMEngine for testing."""
    # Resolve model paths: if relative, make absolute relative to repo root
    if not os.path.isabs(model_id):
        target_model_path = os.path.abspath(model_id)
    else:
        target_model_path = model_id
    
    if not os.path.isabs(draft_model_id):
        draft_model_path = os.path.abspath(draft_model_id)
    else:
        draft_model_path = draft_model_id
    
    target_model_name = os.path.basename(target_model_path)
    draft_model_name = os.path.basename(draft_model_path)
    target_model_config = ModelConfig(target_model_name, model_path=target_model_path, precision=dtype.yalis)
    draft_model_config = ModelConfig(draft_model_name, model_path=draft_model_path, precision=dtype.yalis)
    inference_config = InferenceConfig(
        max_batch_size=2,
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
