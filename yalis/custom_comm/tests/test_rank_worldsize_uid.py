#!/usr/bin/env python3
"""
Simple test to check if NVSHMEMCommWrapper constructor works and can get rank/world size.
Run with: mpirun -np 4 python test_rank_worldsize.py
"""

import sys
import os
import torch
import numpy as np


# Add the build directory to the Python path so we can import the extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

from yalis_nvshmem_collectives import HAS_NVSHMEM
if HAS_NVSHMEM:
  try:
      from yalis_nvshmem_collectives import nvshmem_comm_cuda
      print("✓ Successfully imported nvshmem_comm_cuda extension")
  except ImportError as e:
        print(f"✗ Failed to import nvshmem_comm_cuda extension: {e}")
        sys.exit(1)
else:
    print("✗ NVSHMEM is not available")
    sys.exit(1)

def test_allreduce():
    """Test creating NVSHMEMCommWrapper and getting rank/world size."""
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4

    print(f"\n=== Testing NVSHMEMCommWrapper Constructor (Rank {rank}/{world_size-1}) ===")
    
    try:
        unique_id = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()

        uid_gpu = unique_id.to("cuda")
        torch.distributed.broadcast(uid_gpu, 0)
        torch.distributed.barrier()
        unique_id = uid_gpu.to("cpu")

        # Create an NVSHMEMCommWrapper instance using the shared unique id
        comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_rank, unique_id)
        print(f"✓ Successfully created NVSHMEMCommWrapper instance for rank {rank}")
        
        # Test getting rank
        wrapper_rank = comm_wrapper.get_rank()
        print(f"✓ get_rank() returned: {wrapper_rank}")
        
        # Test getting world size
        wrapper_world_size = comm_wrapper.get_world_size()
        print(f"✓ get_world_size() returned: {wrapper_world_size}")
        
        # Verify correctness
        if wrapper_rank == rank and wrapper_world_size == world_size:
            print("✓ Wrapper values match MPI values")
        else:
            print(f"✗ Wrapper values don't match MPI: wrapper({wrapper_rank}, {wrapper_world_size}) vs MPI({rank}, {world_size})")
            return False


        torch.distributed.barrier()
        
        # Test All Reduce with all 1s
        local_rank = rank % 4
        tensor, tensor_id = comm_wrapper.allocate_tensor(4096, torch.float16, torch.device(f"cuda:{local_rank}"), nvshmem_comm_cuda.Protocol.SIMPLE)
        tensor.fill_(1)
        
        num_chunks = 1024 // 32 // 4
        comm_wrapper.set_kernel_params(nvshmem_comm_cuda.Protocol.SIMPLE, 1, 512, 4096)

        for i in range(10000):
            tensor.fill_(1)
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, 0, "recursive")


        torch.cuda.synchronize()

        # # Check if the tensor is all reduced
        if not torch.allclose(tensor, torch.ones(4096, dtype=torch.float16, device=f"cuda:{local_rank}") * world_size):
            print(f"✗ All reduce (all 1s) failed on rank {rank}")
            print(f"Tensor: {tensor}")
            return False

        print(f"✓ All reduce (all 1s) completed on rank {rank}")
        print(f"Tensor: {tensor}")
        torch.distributed.barrier()

        # Test All Reduce with random values
        tensor_random_local = torch.randn(4096, dtype=torch.float16, device=f"cuda:{local_rank}")
        tensor_random_global = torch.zeros(4096, dtype=torch.float16, device=f"cuda:{local_rank}")
        tensor_random_global.copy_(tensor_random_local)
        torch.distributed.all_reduce(tensor_random_global, op=torch.distributed.ReduceOp.SUM)


        for i in range(10000):
            tensor.copy_(tensor_random_local)
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, 0, "recursive")
        
        torch.cuda.synchronize()

        if not torch.allclose(tensor, tensor_random_global, rtol=1e-2, atol=2e-2):
            print(f"✗ All reduce (random values) failed on rank {rank}")
            print(f"Tensor: {tensor}")
            print(f"Tensor random global: {tensor_random_global}")
            return False

        print(f"✓ All reduce (random values) completed on rank {rank}")
        print(f"Tensor: {tensor}")
        return True
        
    except Exception as e:
        print(f"✗ Error during testing on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Initialize MPI
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4
    torch.cuda.set_device(torch.device(f"cuda:{local_rank}"))
    
    if rank == 0:
        print("Testing NVSHMEMCommWrapper constructor with Torch Process groups")
        print(f"Running with {world_size} processes")
        print("=" * 60)
    
    # Synchronize all processes
    torch.distributed.barrier()
    
    success = test_allreduce()

    success_t = torch.tensor(int(success), device=f"cuda:{local_rank}", dtype=torch.int32)
    torch.distributed.all_reduce(success_t, op=torch.distributed.ReduceOp.MIN)  # MIN==1 only if everyone had 1
    if success_t.item() == 1:
        print("\n🎉 All tests passed successfully on all ranks!")
    else:
        print("\n❌ Some tests failed!")    

    # Non-root processes wait for root to exit
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


