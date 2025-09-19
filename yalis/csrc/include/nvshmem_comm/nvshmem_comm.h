#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <atomic>

// Include NVSHMEM headers to get proper type definitions
#include <nvshmem.h>
#include <nvshmemx.h>

#include "nvshmem_comm/nvshmem_utils.h"
#include "nvshmem_comm/ll8_coll.cuh"

class NVSHMEMCommWrapper {
public:
    NVSHMEMCommWrapper(int rank, int world_size, int device);
    // Initialize using NVSHMEM unique id based attributes; unique_id is the raw bytes returned by nvshmemx_get_uniqueid
    NVSHMEMCommWrapper(int rank, int world_size, int device, const torch::Tensor& unique_id_bytes);
    ~NVSHMEMCommWrapper();

    // Disable copy constructor and assignment operator
    NVSHMEMCommWrapper(const NVSHMEMCommWrapper&) = delete;
    NVSHMEMCommWrapper& operator=(const NVSHMEMCommWrapper&) = delete;

    void destroy();

    std::tuple<torch::Tensor, uint64_t> allocate_tensor(size_t size, torch::Dtype dtype, torch::Device device);
    void free_tensor(uint64_t id);

    // Collective operations
    void allreduce_preallocated(torch::Tensor& tensor, uint64_t id, uint64_t stream_ptr, std::string alg = "recursive");

    // Configuration methods
    void set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size);

    // Getter methods
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }

    int get_mype() const { return mype_; }
    int get_npes() const { return npes_; }

    static torch::Tensor get_unique_id_bytes();

private:

    void initialize_coll();

    int rank_;
    int world_size_;
    int mype_;
    int npes_;
    int device_;
    bool initialized_;

    nvshmemx_uniqueid_t uid_ = NVSHMEMX_UNIQUEID_INITIALIZER;

    std::unordered_map<uint64_t, Protocol> tensor_to_protocol_map_;
    std::unique_ptr<RecursiveLL8Coll> coll_;
}; 