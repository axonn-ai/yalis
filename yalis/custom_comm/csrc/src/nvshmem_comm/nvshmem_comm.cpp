// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "nvshmem_comm/nvshmem_comm.h"
#include "nvshmem_comm/coll_factory.h"
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <tuple>
#include <type_traits>
#include <ctime>

NVSHMEMCommWrapper::NVSHMEMCommWrapper(int rank, int world_size, int device) 
    : rank_(rank), world_size_(world_size), device_(device), initialized_(false) {
    
    // Initialize MPI if not already initialized
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char **argv = nullptr;
        MPI_Init(&argc, &argv);
    }

    // Set device
    CUDA_CHECK(cudaSetDevice(device_));

    // Initialize NVSHMEM with MPI
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Get PE information
    mype_ = nvshmem_my_pe();
    npes_ = nvshmem_n_pes();

    // Verify rank consistency
    if (mype_ != rank_ || npes_ != world_size_) {
        throw std::runtime_error("MPI rank/world_size mismatch with NVSHMEM PE info");
    }

    // Initialize default protocol
    initialize_coll(Protocol::LL8);

    initialized_ = true;
    std::cout << "NVSHMEM initialized for PE " << mype_ << " on " << npes_ << " PEs" << std::endl;
}


// Unique ID-based initialization
NVSHMEMCommWrapper::NVSHMEMCommWrapper(int rank, int world_size, int device, const torch::Tensor& unique_id_tensor)
    : rank_(rank), world_size_(world_size), device_(device), initialized_(false) {
    CUDA_CHECK(cudaSetDevice(device_));

    nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;

    if (unique_id_tensor.numel() != sizeof(nvshmemx_uniqueid_t)) {
        throw std::runtime_error("unique_id_tensor has wrong size for nvshmemx_uniqueid_t");
    }
    memcpy(&uid, unique_id_tensor.data_ptr(), sizeof(uid));

    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

    mype_ = nvshmem_my_pe();
    npes_ = nvshmem_n_pes();

    if (mype_ != rank_ || npes_ != world_size_) {
        throw std::runtime_error("Rank/world_size mismatch with NVSHMEM PE info");
    }

    // Initialize default protocol
    initialize_coll(Protocol::LL8);

    initialized_ = true;
    std::cout << "NVSHMEM initialized for PE " << mype_ << " on " << npes_ << " PEs" << std::endl;
}

NVSHMEMCommWrapper::~NVSHMEMCommWrapper() {
    if (initialized_) {
        std::cout << "NVSHMEMCommWrapper destructor called" << std::endl; destroy();
    }
}

void NVSHMEMCommWrapper::destroy() {
    if (initialized_) {
        std::cout << "NVSHMEMCommWrapper destroying" << std::endl;
        nvshmem_barrier_all();
        coll_map_.clear();  // This will automatically delete all unique_ptr objects
        nvshmem_finalize();
        initialized_ = false;
    }
}

void NVSHMEMCommWrapper::initialize_coll(Protocol protocol) {
    if (coll_map_.find(protocol) == coll_map_.end()) {
        // Create the coll object using factory
        coll_map_[protocol] = std::unique_ptr<IColl>(CollFactory::create_coll(protocol));

        // Initialize with default kernel parameters
        coll_map_[protocol]->init(32, 512, 262144);
    }
}

std::tuple<torch::Tensor, uint64_t> NVSHMEMCommWrapper::allocate_tensor(size_t size, torch::Dtype dtype, torch::Device device, Protocol protocol) {
    // Ensure the protocol-specific coll object exists
    initialize_coll(protocol);

    // Allocate tensor using the appropriate coll object
    auto& coll = coll_map_[protocol];
    auto ret = coll->allocate_tensor(size, dtype, device);
    auto& [tensor, id] = ret;

    // Record which protocol this tensor uses
    tensor_to_protocol_map_[id] = protocol;

    return ret;
}

void NVSHMEMCommWrapper::free_tensor(uint64_t id) {
    if (tensor_to_protocol_map_.find(id) == tensor_to_protocol_map_.end()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    Protocol protocol = tensor_to_protocol_map_[id];
    tensor_to_protocol_map_.erase(id);

    // Free tensor using the appropriate coll object
    auto& coll = coll_map_[protocol];
    coll->free_tensor(id);
}

void NVSHMEMCommWrapper::allreduce_preallocated(torch::Tensor& tensor, uint64_t id, uint64_t stream_ptr, std::string alg) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    // Get the protocol for this tensor
    if (tensor_to_protocol_map_.find(id) == tensor_to_protocol_map_.end()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    Protocol protocol = tensor_to_protocol_map_[id];

    // Get the appropriate coll object
    auto& coll = coll_map_[protocol];
    coll->dispatch_allreduce_preallocated(tensor, id, stream, alg);
}

void NVSHMEMCommWrapper::set_kernel_params(Protocol protocol, int num_blocks, int threads_per_block, size_t chunk_size) {
    // Apply kernel parameters to the specified protocol
    if (coll_map_.find(protocol) == coll_map_.end()) {
        throw std::runtime_error("Protocol is not initialized");
    }
    auto& coll = coll_map_[protocol];
    coll->set_kernel_params(num_blocks, threads_per_block, chunk_size);
}

torch::Tensor NVSHMEMCommWrapper::get_unique_id_bytes() {
    nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
    nvshmemx_get_uniqueid(&uid);

    auto uid_tensor = torch::empty({sizeof(uid)}, torch::dtype(torch::kInt8).device(torch::kCPU));
    std::memcpy((void*)uid_tensor.data_ptr(), (void*)&uid, sizeof(uid));
    return uid_tensor;
}


