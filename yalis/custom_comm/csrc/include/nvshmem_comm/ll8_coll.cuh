// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "nvshmem_comm/coll.h"
#include "nvshmem_comm/nvshmem_utils.h"
#define POLL_TIMEOUT 100

// 8B payload structure for inter-node communication with dirty/clean flag
template <typename T>
struct alignas(8) Payload8B {
  // 8B cell: [data[N] | flag(u32)]
  static constexpr size_t kFlagBytes      = 4;
  static constexpr size_t kCellBytes      = 8;
  static constexpr size_t kMaxDataBytes   = kCellBytes - kFlagBytes;

  static_assert(sizeof(T) <= kMaxDataBytes,
                "T is too large to fit any element alongside a 4-byte flag in 16 bytes.");

  // Max N that fits alongside the 4B flag
  static constexpr size_t N               = kMaxDataBytes / sizeof(T);
  static constexpr size_t kDataBytes      = N * sizeof(T);
  static constexpr size_t kPadBytes       = kCellBytes - (kDataBytes + kFlagBytes);

  T         data[N];
  uint32_t  flag;                         // 0 = not ready, 1 = ready (or any scheme)


  // TODO: Remove these
  __host__ __device__ inline void set_dirty(bool ready = true) { flag = ready ? 1u : 0u; }
  __host__ __device__ inline bool is_dirty() const             { return flag != 0u; }

  // Pack/unpack helpers (assumes vals has at least N elems)
  __host__ __device__ inline void pack(const T* vals) {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) data[i] = vals[i];
  }

  __host__ __device__ inline void unpack(T* out) const {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) out[i] = data[i];
  }

  // Generic in-place add: this += rhs
  __device__ __forceinline__ void add_from(const Payload8B& __restrict__ rhs) {
    #pragma unroll
    for (size_t j = 0; j < N; ++j) {
      data[j] = add_op(data[j], rhs.data[j]);  // your add_op<T>(a,b) specialization
    }
  }

  __device__ __forceinline__ void add_update(const Payload8B& __restrict__ op1, const Payload8B& __restrict__ op2) {
    #pragma unroll
    for (size_t j = 0; j < N; ++j) {
      data[j] = add_op(op1.data[j], op2.data[j]);  // your add_op<T>(a,b) specialization
    }
    flag = op1.flag;
  }

};

template <>
__device__ __forceinline__ void Payload8B<__nv_bfloat16>::add_from(const Payload8B<__nv_bfloat16>& __restrict__ rhs) {
  // process 2 elements per iteration using __nv_bfloat162
  size_t vecN = N & ~size_t(1); // even count
  for (size_t i = size_t(0); i < vecN; i += size_t(2)) {
    __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(rhs.data + i);
    __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(data + i);

    float2 af = __bfloat1622float2(a);
    float2 bf = __bfloat1622float2(b);
    __nv_bfloat162 c = __floats2bfloat162_rn(af.x + bf.x, af.y + bf.y);
    *reinterpret_cast<__nv_bfloat162*>(data + i) = c;
  }
}

template <>
__device__ __forceinline__ void Payload8B<__nv_bfloat16>::add_update(const Payload8B<__nv_bfloat16>& __restrict__ op1, const Payload8B<__nv_bfloat16>& __restrict__ op2) {
  // process 2 elements per iteration using __nv_bfloat162
  size_t vecN = N & ~size_t(1); // even count
  for (size_t i = size_t(0); i < vecN; i += size_t(2)) {
    __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(op1.data + i);
    __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(op2.data + i);
    float2 af = __bfloat1622float2(a);
    float2 bf = __bfloat1622float2(b);
    __nv_bfloat162 c = __floats2bfloat162_rn(af.x + bf.x, af.y + bf.y);
    *reinterpret_cast<__nv_bfloat162*>(data + i) = c;
  }
  flag = op1.flag;
}

// Sanity: total size is exactly 8
static_assert(sizeof(Payload8B<float>)        == 8, "float payload should be 8B");       // N=3
static_assert(sizeof(Payload8B<__nv_bfloat16>)== 8, "bf16 payload should be 8B");        // N=6

inline size_t elems_per_cell(c10::ScalarType st) {
  switch (st) {
    case c10::kBFloat16: return Payload8B<__nv_bfloat16>::N;
    case c10::kHalf:     return Payload8B<__half>::N;
    case c10::kFloat:    return Payload8B<float>::N;
    case c10::kInt:      return Payload8B<int>::N;
    default: throw std::runtime_error("Unsupported dtype for 8B payload");
  }
}

__global__ void bump_seq(uint32_t* seq, uint64_t* signal, int steps_inter);
__global__ void init_signal_kernel(uint64_t *signal, size_t n);

template <typename T>
__global__ void pack_payloads_kernel(Payload8B<T>* __restrict__ payloads,
                                     const T* __restrict__ data,
                                     size_t num_elements,
                                     uint32_t* seq_num);

template <typename T>
__global__ void unpack_payloads_kernel(T* __restrict__ data,
                                       const Payload8B<T>* __restrict__ payloads,
                                       size_t num_elements,
                                       uint32_t* seq_num,
                                       uint64_t* seq_num_signal,
                                       int steps, int steps_intra, int steps_inter);

template <typename T>
__global__ extern void recursive_allreduce_kernel_payload(Payload8B<T> *dst, Payload8B<T> *src, size_t nreduce,
                                                   int steps, int steps_intra, int steps_inter,
                                                   size_t chunk_elems, size_t stride_size, uint32_t* seq_num, 
                                                   uint64_t* seq_num_signal);

class RecursiveLL8Coll: public CollBase<RecursiveLL8Coll> {
  public:
    using Base = CollBase<RecursiveLL8Coll>;


    RecursiveLL8Coll() = default;

    ~RecursiveLL8Coll() noexcept override { 
        cleanup(); 
    }

    void initialize(int num_blocks, int threads_per_block, size_t chunk_size);
    void cleanup();
    void register_tensor(uint64_t id, size_t size, torch::Dtype dt, torch::Device dev);
    void deregister_tensor(uint64_t id);

    void set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size) {
      num_blocks_ = num_blocks;
      threads_per_block_ = threads_per_block;
      chunk_size_ = chunk_size;
    }

    template <typename T>
    void allreduce_preallocated_impl(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg);

  private:
    template <typename T>
    void pack_payloads(Payload8B<T>* payloads, const T* data, size_t num_elements, uint32_t* seq_num, cudaStream_t stream);

    template <typename T>
    void unpack_payloads(T* data, const Payload8B<T>* payloads, size_t num_elements, uint32_t* seq_num, uint64_t* seq_num_signal, int steps, int steps_intra, int steps_inter, cudaStream_t stream);

    template <typename T>
    void __forceinline__ dispatch_intra_reducescatter(T *dst, const T *src, size_t num_elements, cudaStream_t stream) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            nvshmemx_bfloat16_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, __half>) {
            nvshmemx_half_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, float>) {
            nvshmemx_float_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, int>) {
            nvshmemx_int_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for intra-node reduce scatter");
        }
    }

    template <typename T>
    void __forceinline__ dispatch_inter_allgather(T *dst, const T *src, size_t num_elements, cudaStream_t stream) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            nvshmemx_bfloat16_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, __half>) {
            nvshmemx_half_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, float>) {
            nvshmemx_float_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, int>) {
            nvshmemx_int_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for inter-node allgather");
        }
    }

    int num_blocks_;
    int threads_per_block_;
    size_t chunk_size_;

    int steps_;
    int steps_intra_;
    int steps_inter_;

    std::unordered_map<uint64_t, void *> allocated_scratch_send_;
    std::unordered_map<uint64_t, void *> allocated_scratch_recv_;
    std::unordered_map<uint64_t, uint32_t*> seq_nums_;
    std::unordered_map<uint64_t, uint64_t*> seq_num_signals_;
    std::unordered_map<uint64_t, size_t> stride_sizes_;
};


