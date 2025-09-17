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

// Include our utilities and types
#include "nvshmem_utils.h"

// Recursive-doubling all-reduce using NVSHMEM with 16B payloads (templated)
template <typename T>
__global__ extern void recursive_allreduce_kernel_payload(Payload16B<T> *dst, Payload16B<T> *src, size_t nreduce,
                                                   size_t chunk_elems, size_t stride_size, uint32_t* seq_num, uint64_t* signal);
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

    // Main communication methods
    void allreduce(torch::Tensor& tensor, uint64_t stream_ptr, std::string alg = "ring");

    void allreduce_preallocated(torch::Tensor& tensor, uint64_t id, uint64_t stream_ptr, std::string alg = "ring");

    // Configuration methods
    void set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size);
    
    // Getter methods
    int get_rank() const { return rank_; }
    int get_world_size() const { return world_size_; }
    int get_mype() const { return mype_; }
    int get_npes() const { return npes_; }

    static torch::Tensor get_unique_id_bytes();

    template <typename T>
    struct StepBuffer {
      // base of the symmetric allocation
      unsigned char* base;
      size_t payload_elems;   // = size/npes_node_
      size_t stride_bytes;    // bytes between step i and i+1

      __host__ __device__ inline T* payload(int step) const {
        return reinterpret_cast<T*>(base + step * stride_bytes);
      }
      __host__ __device__ inline volatile uint16_t* flag(int step) const {
        // flag is immediately after payload
        return reinterpret_cast<volatile uint16_t*>(
          base + step * stride_bytes + payload_elems * sizeof(T));
      }
    };

protected:
    // Templated implementation for different data types and algorithms
template <typename T>
    void allreduce_preallocated_impl(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg);
    

private:
    int rank_;
    int world_size_;
    int device_;
    int mype_;
    int npes_;
    int mype_node_;
    int npes_node_;
    size_t signal_size_;
    bool initialized_;
    
    // Kernel parameters
    int num_blocks_;
    int threads_per_block_;
    size_t chunk_elems_;

    // Recursive-doubling steps
    int steps_world_;
    int steps_intra_;
    int steps_inter_;

    nvshmemx_uniqueid_t uid_ = NVSHMEMX_UNIQUEID_INITIALIZER;

    // Memory Pools
    std::unordered_map<uint64_t, void *> allocated_tensors_;
    std::unordered_map<uint64_t, void *> allocated_scratch_;
    std::unordered_map<uint64_t, uint32_t*> seq_nums_;
    std::unordered_map<uint64_t, size_t> stride_sizes_;
    std::atomic<uint64_t> next_id_;
    uint64_t* signal_;
}; 