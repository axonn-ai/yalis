import pytest
import torch.distributed as dist


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


@pytest.fixture(scope="module", autouse=True)
def cleanup_dist():
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
