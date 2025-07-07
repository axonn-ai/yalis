import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="module", autouse=True)
def dist_group():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
