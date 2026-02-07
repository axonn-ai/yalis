import os
import pytest
import torch.distributed as dist
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis import ModelConfig, InferenceConfig, LLMEngine, SpeculativeLLMEngine
from types import SimpleNamespace
from tests.sample_dataset import AlpacaDataset

# Assume offline mode by default unless otherwise specified
HF_DATASETS_OFFLINE = os.environ.get("HF_DATASETS_OFFLINE", "1") == "1"

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
        "sdpa": "eager", # Use eager as fallback for GptOssForCausalLM
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=HF_DATASETS_OFFLINE,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if tokenizer.chat_template is None:
        template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% endif %}{% endfor %}"
        )
        tokenizer.chat_template = template

    return tokenizer


@pytest.fixture(scope="module")
def alpaca_dataset():
    """Create an Alpaca dataset for testing."""
    dataset = AlpacaDataset(random_seed=42)
    return dataset


# Model fixtures
@pytest.fixture(scope="module")
def yalis_engine(model_id, dtype, attn_backend):
    """Create a standard Yalis LLMEngine with serialized loading.
    
    In distributed mode, ranks serialize their checkpoint loading to avoid
    concurrent memory contention. Each rank independently loads and shards
    its weights via tensor parallelism.
    """
    rank = 0
    world_size = 1

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # Resolve model_path: if model_id is a relative path, make it
    # absolute relative to repo root
    if not os.path.isabs(model_id):
        model_path = os.path.abspath(model_id)
    else:
        model_path = model_id
    model_name_for_config = os.path.basename(model_path)
    model_config = ModelConfig(
        model_name_for_config, model_path=model_path, precision=dtype.yalis
    )
    inference_config = InferenceConfig(
        max_batch_size=4,
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend.yalis,
        use_paged_kv_caching=False,
    )
    
    # Serialize checkpoint loading: each rank loads and shards sequentially
    if world_size > 1 and dist.is_initialized():
        for r in range(world_size):
            if rank == r:
                engine = LLMEngine(
                    model_config=model_config, inference_config=inference_config
                )
            dist.barrier()  # Ensure rank r finishes before rank r+1 starts
    else:
        engine = LLMEngine(
            model_config=model_config, inference_config=inference_config
        )

    return engine


@pytest.fixture(scope="module")
def hf_model(model_id, dtype, attn_backend, device):
    """Create a HuggingFace model for comparison testing (rank 0 only).

    Only rank 0 loads the HF model since it's only used for reference outputs
    in single-GPU mode. Other ranks return None.
    """
    rank = 0
    world_size = 1

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if rank != 0:
        # Only rank 0 loads the HF model; other ranks return None
        return None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation=attn_backend.hf,
        torch_dtype=dtype.hf,
        device_map="auto",
        local_files_only=HF_DATASETS_OFFLINE,
        trust_remote_code=True,
    )
    model.eval()

    return model


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
