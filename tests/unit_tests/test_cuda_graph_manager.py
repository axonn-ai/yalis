"""Tests for CUDA Graph Manager infrastructure."""

import torch
import pytest
from yalis.cuda_graph_manager import CUDAGraphManager


def test_cuda_graph_manager_initialization():
    """Test CUDAGraphManager initialization with various configs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test 1: Default powers-of-2 behavior
    manager = CUDAGraphManager(max_batch_size=16, device=device)
    assert manager.capture_sizes == [1, 2, 4, 8, 16]
    assert manager.enabled is True

    # Test 2: Empty list (lazy mode)
    manager = CUDAGraphManager(
        max_batch_size=16, device=device, cuda_graph_capture_sizes=[]
    )
    assert manager.capture_sizes == []

    # Test 3: Custom batch sizes
    custom_sizes = [1, 2, 3, 4, 5, 7, 8]
    manager = CUDAGraphManager(
        max_batch_size=16, device=device, cuda_graph_capture_sizes=custom_sizes
    )
    assert manager.capture_sizes == custom_sizes

    # Test 4: Filtering oversized batch sizes
    custom_sizes = [1, 2, 4, 8, 16, 32, 64]  # 32, 64 > max_batch_size
    manager = CUDAGraphManager(
        max_batch_size=16, device=device, cuda_graph_capture_sizes=custom_sizes
    )
    assert manager.capture_sizes == [1, 2, 4, 8, 16]


def test_find_suitable_batch_size():
    """Test finding suitable batch size for graph routing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manager = CUDAGraphManager(
        max_batch_size=32,
        device=device,
        cuda_graph_capture_sizes=[1, 2, 4, 8, 16],
    )

    # Manually add some graph entries for testing (mock)
    from yalis.cuda_graph_manager import CUDAGraphEntry

    dummy_tensor = torch.zeros(1)

    for bs in [1, 2, 4, 8]:
        manager.graph_pool[bs] = CUDAGraphEntry(
            graph=None,
            batch_size=bs,
            static_tokens=dummy_tensor,
            static_block_table=dummy_tensor,
            static_token_counter=dummy_tensor,
            static_output_token=dummy_tensor,
            static_output_logits=None,
        )

    # Test exact match
    assert manager.find_suitable_batch_size(1) == 1
    assert manager.find_suitable_batch_size(4) == 4

    # Test rounding up to next available
    assert manager.find_suitable_batch_size(3) == 4
    assert manager.find_suitable_batch_size(5) == 8

    # Test when no suitable graph available
    assert manager.find_suitable_batch_size(16) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
