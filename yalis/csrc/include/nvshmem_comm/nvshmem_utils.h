#pragma once

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>  

#define WARP_SIZE 32

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
        default: return alignof(std::max_align_t);
    }
}
