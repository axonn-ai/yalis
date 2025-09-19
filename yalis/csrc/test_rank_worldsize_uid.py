#!/usr/bin/env python3
"""
Simple test to check if NVSHMEMCommWrapper constructor works and can get rank/world size.
Run with: mpirun -np 4 python test_rank_worldsize.py
"""

import sys
import os
import torch
import nvshmem.core
import numpy as np



# Add the build directory to the Python path so we can import the extension
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'build'))

try:
    import nvshmem_comm_cuda
    print("✓ Successfully imported nvshmem_comm_cuda extension")
except ImportError as e:
    print(f"✗ Failed to import nvshmem_comm_cuda extension: {e}")
    sys.exit(1)

def test_constructor():
    """Test creating NVSHMEMCommWrapper and getting rank/world size."""
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = rank % 4

    print(f"\n=== Testing NVSHMEMCommWrapper Constructor (Rank {rank}/{world_size-1}) ===")
    
    try:
        unique_id = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()

        # Create an empty uniqueid for all ranks
        uniqueid = nvshmem.core.get_unique_id(empty=True)
        if rank == 0:
            # Rank 0 gets a real uniqueid
            #uniqueid = nvshmem.core.get_unique_id()
            uniqueid = unique_id
            broadcast_objects = [uniqueid]
        else:
            broadcast_objects = [None]

        ## We use torch.distributed.broadcast_object_list to send the UID to all ranks
        torch.distributed.broadcast_object_list(broadcast_objects, src=0)
        torch.distributed.barrier()

        ##uid_tensor = torch.from_numpy(broadcast_objects[0]._data.view(np.int8))
        uid_tensor = broadcast_objects[0]

        #print(f"Unique ID: {type(uid_tensor)}")
        #print(f"Unique ID: {broadcast_objects[0]._data.view(np.int8)}")


        uid_gpu = unique_id.to("cuda")
        torch.distributed.broadcast(uid_gpu, 0)
        torch.distributed.barrier()
        unique_id = uid_gpu.to("cpu")

        #print(f"Unique ID: {unique_id}")

        # Create an NVSHMEMCommWrapper instance using MPI rank and size

        # Parameters: (rank, world_size, device)
        comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, world_size, local_rank, unique_id)
        print(f"✓ Successfully created NVSHMEMCommWrapper instance for rank {rank}")
        
        # Test getting rank
        wrapper_rank = comm_wrapper.get_rank()
        print(f"✓ get_rank() returned: {wrapper_rank}")
        
        # Test getting world size
        wrapper_world_size = comm_wrapper.get_world_size()
        print(f"✓ get_world_size() returned: {wrapper_world_size}")
        
        # Verify values match MPI
        if wrapper_rank == rank and wrapper_world_size == world_size:
            print("✓ Wrapper values match MPI values")
        else:
            print(f"✗ Wrapper values don't match MPI: wrapper({wrapper_rank}, {wrapper_world_size}) vs MPI({rank}, {world_size})")
            return False


        torch.distributed.barrier()
        
        # Test All Reduce
        local_rank = rank % 4
        tensor, tensor_id = comm_wrapper.allocate_tensor(4096, torch.bfloat16, torch.device(f"cuda:{local_rank}"), nvshmem_comm_cuda.Protocol.SIMPLE)
        tensor.fill_(1)
        #tensor = torch.ones(1024 * 1024, dtype=torch.int32, device=f"cuda:{local_rank}")
        
        # # Print tensor
        # #print(f"{tensor_id} Tensor: {tensor}")

        num_chunks = 1024 // 32 // 4
        comm_wrapper.set_kernel_params(nvshmem_comm_cuda.Protocol.SIMPLE, 1, 512, 4096)

        for i in range(4000):
            tensor.fill_(1)
            comm_wrapper.allreduce_preallocated(tensor, tensor_id, 0, "recursive")


        torch.cuda.synchronize()

        # # Check if the tensor is all reduced
        if not torch.allclose(tensor, torch.ones(4096, dtype=torch.bfloat16, device=f"cuda:{local_rank}") * world_size):
            print(f"✗ All reduce failed on rank {rank}")
            print(f"Tensor: {tensor}")
            return False

        print(f"✓ All reduce completed on rank {rank}")
        print(f"Tensor: {tensor}")
        torch.distributed.barrier()
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
    
    success = test_constructor()
   
    # Gather results from all processes
    # all_success = torch.distributed.all_gather(success)
    
    # if rank == 0:
    #     if all(all_success):
    #         print("\n🎉 All tests passed successfully on all ranks!")
    #     else:
    #         print(f"\n❌ Some tests failed!")
    #         print(f"Results: {all_success}")
    
    # Non-root processes wait for root to exit
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


