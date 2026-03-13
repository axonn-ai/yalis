/*  SPDX-License-Identifier: Apache-2.0
    SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    This file is adapted from
       https://github.com/vllm-project/vllm/blob/main/csrc/activation_kernels.cu
*/
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

// Activation and gating kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const int64_t base = token_idx * 2 * d;
    const scalar_t x = VLLM_LDG(&input[base + 2 * idx]);
    const scalar_t y = VLLM_LDG(&input[base + 2 * idx + 1]);
    out[token_idx * d + idx] = ACT_FN(x) * y;
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

}  // namespace vllm

// Launch activation and gating kernel.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL)                            \
  int d = input.size(-1) / 2;                                            \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  dim3 grid(num_tokens);                                                 \
  dim3 block(std::min(d, 1024));                                         \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  VLLM_DISPATCH_FLOATING_TYPES(                                          \
      input.scalar_type(), "act_and_mul_kernel", [&] {                   \
        vllm::act_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel);
}

namespace vllm {

__device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul_fn(const T& gate, const T& up,
                                                  float alpha, float limit) {
  const float g = fminf((float)gate, limit);
  const float u = fmaxf(fminf((float)up, limit), -limit);
  return (T)((u + 1.0f) * g / (1.0f + expf(-g * alpha)));
}

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2 * d] interleaved
    const int d, const float alpha, const float limit) {
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int PAIRS = VEC_SIZE / 2;
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_ptr = input + token_idx * 2 * d;
  scalar_t* out_ptr = out + token_idx * d;

  const bool in_aligned = is_16byte_aligned(in_ptr);
  const bool out_aligned =
      (reinterpret_cast<uintptr_t>(out_ptr) & 7) == 0;

  if (in_aligned && out_aligned && d >= PAIRS) {
    const int4* in_vec = reinterpret_cast<const int4*>(in_ptr);
    int2* out_vec = reinterpret_cast<int2*>(out_ptr);
    const int num_vecs = d / PAIRS;
    const int vec_end = num_vecs * PAIRS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 v = VLLM_LDG(&in_vec[i]);
      int2 r;
      auto* vp = reinterpret_cast<scalar_t*>(&v);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < PAIRS; j++) {
        rp[j] = ACT_FN(vp[2 * j], vp[2 * j + 1], alpha, limit);
      }
      out_vec[i] = r;
    }
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = ACT_FN(VLLM_LDG(&in_ptr[2 * i]),
                           VLLM_LDG(&in_ptr[2 * i + 1]), alpha, limit);
    }
  } else {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t gate = VLLM_LDG(&in_ptr[2 * idx]);
      const scalar_t up = VLLM_LDG(&in_ptr[2 * idx + 1]);
      out_ptr[idx] = ACT_FN(gate, up, alpha, limit);
    }
  }
}

}  // namespace vllm

#define LAUNCH_SWIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                         \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "swigluoai_and_mul_kernel", [&] {                   \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),             \
                                         input.data_ptr<scalar_t>(), d, ALPHA, \
                                         LIMIT);                               \
      });

void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SWIGLUOAI_AND_MUL(vllm::swigluoai_and_mul_fn, alpha, limit);
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_tanh_kernel);
}

namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = VLLM_LDG(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                       \
  int d = input.size(-1);                                                      \
  int64_t num_tokens = input.numel() / d;                                      \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "activation_kernel", [&] { \
    vllm::activation_kernel<scalar_t, KERNEL<scalar_t>>                        \
        <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),                 \
                                     input.data_ptr<scalar_t>(), d);           \
  });

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}
