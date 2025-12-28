#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
using namespace nvcuda;
using namespace nvcuda::wmma;

extern "C" __global__ void decode_attn_two_pass_wmma(
    const half*  __restrict__ Q,       // [B*H, D]
    const half*  __restrict__ K,       // [B*H, T, D]
    const half*  __restrict__ V,       // [B*H, T, D]
    const float* __restrict__ Bias,    // [B*H, T]
          half*  __restrict__ Out,     // [B*H, D]
    const float* __restrict__ Thr,     // [B*H]
    float        scale,
    int B, int H, int T, int D
) {
  constexpr int NUM_WARPS = 4;

  int idx = blockIdx.x;
  if (idx >= B*H) return;

  // Base pointers into Q/K/V/Bias/Out
  const half*  q_ptr    = Q    + idx*(size_t)D;
  const half*  k_base   = K    + idx*(size_t)T*D;
  const half*  v_base   = V    + idx*(size_t)T*D;
  const float* bias_ptr = Bias + idx*(size_t)T;
        half*  out_ptr  = Out  + idx*(size_t)D;
  float thr_scalar      = Thr[idx];

  int tid = threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid % 32;

  // Shared tiles and softmax stats.
  __shared__ __align__(32) half a_tile[8][16];
  __shared__ __align__(32) float C_tile[NUM_WARPS][16];
  __shared__ float m_arr[NUM_WARPS], l_arr[NUM_WARPS];
  __shared__ float m, l;
  __shared__ int valid_cnt;

  // Dynamic shared: logits buffer + valid index list + Q scratch.
  extern __shared__ uint8_t _smem[];
  half* exp_buf = reinterpret_cast<half*>(_smem);
  uint16_t* valid_idx = reinterpret_cast<uint16_t*>(_smem + T * sizeof(half));
  half* q_half = reinterpret_cast<half*>(_smem + (T * sizeof(half)) + (T * sizeof(uint16_t)));

  // Load Q and stage into the WMMA A tile.
  if (tid < D) {
    q_half[tid] = __ldg(q_ptr + tid);
  }
  __syncthreads();

  if (tid < D) {
    a_tile[tid / 16][tid % 16] = q_half[tid];
  }
  __syncthreads();

  m_arr[warp_id] = -1e9f;
  l_arr[warp_id] = 0.0f;
  m = -1e9f;
  l = 0.0f;

  // WMMA fragments.
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,       16,16,16, half, row_major>  Afrag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,       16,16,16, half, col_major>  Bfrag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,    16,16,16, float>            Cfrag;

  constexpr int BLOCK_N = 16;
  int warp_offset = warp_id * BLOCK_N;
  for (int t0 = warp_offset; t0 < T; t0 += BLOCK_N * 4) {
    int bs = min(BLOCK_N, T - t0);
    
    wmma::fill_fragment(Cfrag, 0);
    
    // PASS 1: Compute logits and track running (m,l).
    for (int d0 = 0; d0 < D; d0 += 16) {
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);

      wmma::load_matrix_sync(Afrag, &a_tile[d0 / 16][0], 0);
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }

    // Store the C tile to shared memory
    wmma::store_matrix_sync(&C_tile[warp_id][0], Cfrag, 0, nvcuda::wmma::mem_row_major);

    if (lane < bs) {
      C_tile[warp_id][lane] = C_tile[warp_id][lane] * scale + __ldg(bias_ptr + t0 + lane);
    }

    // Update warp-local softmax stats and store logits.
    if (lane == 0) {
      for (int i = 0; i < bs; ++i) {
        float x  = C_tile[warp_id][i];
        float nm = fmaxf(m_arr[warp_id], x);
        float em = expf(m_arr[warp_id]  - nm);
        float eb = expf(x  - nm);
        l_arr[warp_id] = fmaf(l_arr[warp_id], em, eb);
        m_arr[warp_id] = nm;
        exp_buf[t0 + i] = __float2half(x);
      }
    }
  }

  // Reduce across warps to get block max and sum
  __syncthreads();
  if (tid == 0) {
    float m_final = m_arr[0];
    float l_final = l_arr[0];
  
    for (int i = 1; i < NUM_WARPS; ++i) {
      float m_i = m_arr[i];
      float l_i = l_arr[i];
  
      float nm = fmaxf(m_final, m_i);
      float em1 = expf(m_final - nm);
      float em2 = expf(m_i - nm);
      l_final = l_final * em1 + l_i * em2;
      m_final = nm;
    }
  
    m = m_final;
    l = l_final;
  }
  __syncthreads();


  // Compute softmax using block max/sum and apply threshold.
  for (int i = threadIdx.x; i < T; i += blockDim.x) {
      exp_buf[i] = __float2half(expf(__half2float(exp_buf[i]) - m) / l);
      exp_buf[i] = __half2float(exp_buf[i]) < thr_scalar ? __float2half(0.0f) : exp_buf[i];
  }
  __syncthreads();

  // Pack the valid indices
  if (threadIdx.x == 0) { 
      valid_cnt = 0;
      for (int i = 0; i < T; ++i) {
          // Pack the indices that are greater than 0 into valid_idx
          if (__half2float(exp_buf[i]) > 0.0f) {
              valid_idx[valid_cnt] = i;
              valid_cnt++;
          }
      }
  }

  // Weighted sum with V
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t < valid_cnt; ++t) {
        int idx = valid_idx[t];
        float s = __half2float(exp_buf[idx]);
        float v = __half2float(__ldg(v_base + idx * D + d));
        sum += s * v;
    }
    out_ptr[d] = __float2half(sum);
  }
}

// Launcher (called from C++ wrapper)
extern "C" void decode_attn_cuda_launcher(
    const void* Qp, const void* Kp, const void* Vp,
    const float* Bias, void* Outp,
    const float* Thr, float scale,
    int B, int H, int T, int D
) {
  // cast back to __half*
  const __half* Qh = reinterpret_cast<const __half*>(Qp);
  const __half* Kh = reinterpret_cast<const __half*>(Kp);
  const __half* Vh = reinterpret_cast<const __half*>(Vp);
        __half* Oh = reinterpret_cast<      __half*>(Outp);

  int blocks  = B*H;
  int threads = 128;
  size_t dyn_shm = size_t(T)*sizeof(__half)
                 + size_t(T)*sizeof(uint16_t)
                 + size_t(D)*sizeof(__half);


  decode_attn_two_pass_wmma
    <<<blocks, threads, dyn_shm>>>(
      Qh, Kh, Vh, Bias, Oh, Thr, scale, B, H, T, D
    );

  // Error check for async launch.
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
}
