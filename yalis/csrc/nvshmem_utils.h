#pragma once

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>  // for std::max_align_t

// CUDA error checking macro
#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                          \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (cudaSuccess != result) {                                              \
            throw std::runtime_error(std::string("CUDA failed: ") + cudaGetErrorString(result)); \
        }                                                                         \
    } while (0)

// Template functions for addition operations
template <typename T>
__device__ inline T add_op(T a, T b) { return a + b; }

template <>
__device__ inline __half add_op(__half a, __half b) { return __hadd(a, b); }

template <>
__device__ inline __nv_bfloat16 add_op(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = __bfloat162float(a);
    float fb = __bfloat162float(b);
    return __float2bfloat16(fa + fb);
}

// ---- generic reducer: scalar stride (works for all T) ----
template <typename T>
__device__ inline void reduce_add_vec(
    T* __restrict__ dst, const T* __restrict__ src,
    size_t n, int compute_threads, int compute_rank)
{
    for (size_t i = compute_rank; i < n; i += compute_threads) {
        dst[i] = add_op(src[i], dst[i]);
    }
}

// ---- bf16 specialization: vectorized with __nv_bfloat162 ----
template <>
__device__ inline void reduce_add_vec<__nv_bfloat16>(
    __nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src,
    size_t n, int compute_threads, int compute_rank)
{
    // process 2 elements per iteration using __nv_bfloat162
    size_t vecN = n & ~size_t(1); // even count
    for (size_t i = size_t(compute_rank) * 2; i < vecN; i += size_t(compute_threads) * 2) {
        __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(src + i);
        __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(dst + i);

        float2 af = __bfloat1622float2(a);
        float2 bf = __bfloat1622float2(b);
        __nv_bfloat162 c = __floats2bfloat162_rn(af.x + bf.x, af.y + bf.y);

        *reinterpret_cast<__nv_bfloat162*>(dst + i) = c;
    }

    // tail (odd element) handled once
    if ((vecN < n) && (compute_rank == 0)) {
        __nv_bfloat16 a = src[vecN];
        __nv_bfloat16 b = dst[vecN];
        dst[vecN] = __float2bfloat16(__bfloat162float(a) + __bfloat162float(b));
    }
}

// Generic signaling put using byte-size for portability across types
__device__ inline void put_signal_block_bytes(void *dst, const void *src, size_t bytes,
                                              uint64_t *signal, uint64_t val, int op, int pe) {
    // Note: This function declaration is kept for compatibility
    // Implementation would depend on the specific NVSHMEM version being used
}

// 16B payload structure for inter-node communication with dirty/clean flag
template <typename T>
struct alignas(8) Payload16B {
  // 16B cell: [ data[N] | flag(u32) | pad ]
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

  __host__ __device__ inline void set_dirty(bool ready = true) { flag = ready ? 1u : 0u; }
  __host__ __device__ inline bool is_dirty() const             { return flag != 0u; }

  // Pack/unpack helpers (assumes vals has at least N elems)
  __host__ __device__ inline void pack(const T* vals) {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) data[i] = vals[i];
    set_dirty(true);
  }
  __host__ __device__ inline void unpack(T* out) const {
    #pragma unroll
    for (size_t i = 0; i < N; ++i) out[i] = data[i];
  }

  // Generic in-place add: this += rhs
  __device__ __forceinline__ void add_from(const Payload16B& __restrict__ rhs) {
    #pragma unroll
    for (size_t j = 0; j < N; ++j) {
      data[j] = add_op(data[j], rhs.data[j]);  // your add_op<T>(a,b) specialization
    }
  }


};

template <>
__device__ __forceinline__ void Payload16B<__nv_bfloat16>::add_from(const Payload16B<__nv_bfloat16>& __restrict__ rhs) {
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


// Sanity: total size is exactly 16
// static_assert(sizeof(Payload16B<float>)        == 16, "float payload should be 16B");       // N=3
// static_assert(sizeof(Payload16B<__nv_bfloat16>)== 16, "bf16 payload should be 16B");        // N=6
// static_assert(sizeof(Payload16B<double>)       == 16, "double payload should be 16B");      // N=1

// Utility functions
static inline __host__ __device__
size_t align_up(size_t x, size_t a) { return (x + a - 1) & ~(a - 1); }

inline size_t element_align(torch::ScalarType st) {
    switch (st) {
        case c10::kFloat:    return alignof(float);
        case c10::kHalf:     return alignof(__half);
        case c10::kBFloat16: return alignof(__nv_bfloat16);
        case c10::kInt:      return alignof(int);
        case c10::kLong:     return alignof(long);
        case c10::kChar:     return alignof(char);
        // case c10::kDouble:   return alignof(double);
        default: return alignof(std::max_align_t);
    }
}

inline size_t elems_per_cell(c10::ScalarType st) {
  switch (st) {
    // case c10::kDouble:   return Payload16B<double>::N;
    case c10::kBFloat16: return Payload16B<__nv_bfloat16>::N;
    case c10::kHalf:     return Payload16B<__half>::N;
    case c10::kFloat:    return Payload16B<float>::N;
    case c10::kInt:      return Payload16B<int>::N;
    default: throw std::runtime_error("Unsupported dtype for 16B payload");
  }
}
