#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math.h>
#include <iostream>
using namespace nvcuda;
using namespace nvcuda::wmma;

// Doubling-region update cadence.
__device__ __forceinline__ int compute_step(int t, int base_cutoff, int base_step) {
  if (t < base_cutoff) {
    return base_step;
  }
  int ratio = t / base_cutoff;
  int k = 31 - __clz((unsigned)ratio);
  return base_step << (k + 1);
}

extern "C" __global__ void nowmp_decode_attn_gmem_logits_cache(
    const half*  __restrict__ Q,       // [B*H, D]
    const half*  __restrict__ K,       // [B*H, T, D]
    const half*  __restrict__ V,       // [B*H, T, D]
    const float* __restrict__ Bias,    // [B*H, T]
          half*  __restrict__ Out,     // [B*H, D]
          float* __restrict__ alpha,   // [B*H]
          float* __restrict__ b,       // [B*H]
          float* __restrict__ kept_cum,// [B*H]
          float* __restrict__ total_cum,//[B*H]
          float* __restrict__ lr_a,    // [B*H]
          float* __restrict__ lr_b,    // [B*H]
    const int*   __restrict__ token_counter, // [B]
    float r_target,
    float log_a_min,
    float log_a_max,
    float b_min,
    float b_max,
    int base_cutoff,
    int base_step,
    float anl_a,
    float anl_b,
    float scale,
    half* __restrict__ logits_gmem,    // [B*H, T]
    float* __restrict__ keep_counts,   // [B*H]
    int B, int H, int T, int D
) {
  constexpr int NUM_WARPS = 4;
  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_N = 16;
  constexpr unsigned ACTIVE_MASK = (1u << BLOCK_N) - 1u;

  int bh = blockIdx.x;
  if (bh >= B * H) {
    return;
  }

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  int b_idx = bh / H;

  const half*  q_ptr    = Q    + bh * (size_t)D;
  const half*  k_base   = K    + bh * (size_t)T * D;
  const half*  v_base   = V    + bh * (size_t)T * D;
  const float* bias_ptr = Bias + bh * (size_t)T;
        half*  out_ptr  = Out  + bh * (size_t)D;

  __shared__ __align__(32) half  a_tile[8][16];
  __shared__ __align__(32) float C_tile[NUM_WARPS][16];
  __shared__ float m_arr[NUM_WARPS], l_arr[NUM_WARPS];
  __shared__ float m, l;
  __shared__ int keep_arr[NUM_WARPS];
  __shared__ int t_shared;
  __shared__ int last_idx_shared;
  __shared__ float thr_shared;

  if (tid == 0) {
    int t = token_counter[b_idx] + 1;
    if (t < 1) {
      t = 1;
    }
    if (t > T) {
      t = T;
    }
    t_shared = t;
    last_idx_shared = t - 1;

    // Threshold curve: theta(t) = exp(alpha + b * log(t)), clamped for stability.
    float alpha_val = alpha[bh];
    float b_val = b[bh];
    float log_t = logf((float)t);
    float x = alpha_val + b_val * log_t;
    x = fminf(fmaxf(x, -20.0f), 0.0f);
    thr_shared = expf(x);
  }
  __syncthreads();

  extern __shared__ uint8_t _smem[];
  // Dynamic shared: q tile + per-warp partial output.
  half* q_half = reinterpret_cast<half*>(_smem);
  size_t q_bytes = size_t(D) * sizeof(half);
  size_t off = (q_bytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
  float* out_partial = reinterpret_cast<float*>(_smem + off);

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

  if (lane == 0) {
    keep_arr[warp_id] = 0;
  }
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,    16,16,16, half, row_major>  Afrag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,    16,16,16, half, col_major>  Bfrag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16, float>            Cfrag;

  int warp_offset = warp_id * BLOCK_N;
  int t_max = t_shared;

  for (int t0 = warp_offset; t0 < t_max; t0 += BLOCK_N * NUM_WARPS) {
    int bs = min(BLOCK_N, t_max - t0);

    wmma::fill_fragment(Cfrag, 0);
    for (int d0 = 0; d0 < D; d0 += 16) {
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);
      wmma::load_matrix_sync(Afrag, &a_tile[d0 / 16][0], 0);
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }

    wmma::store_matrix_sync(&C_tile[warp_id][0], Cfrag, 0, nvcuda::wmma::mem_row_major);

    if (lane < bs) {
      float x = C_tile[warp_id][lane] * scale + __ldg(bias_ptr + t0 + lane);
      logits_gmem[bh * (size_t)T + t0 + lane] = __float2half(x);
    }

    if (lane == 0) {
      // Numerically stable softmax stats per warp tile.
      for (int i = 0; i < bs; ++i) {
        float x = C_tile[warp_id][i] * scale + __ldg(bias_ptr + t0 + i);
        float nm = fmaxf(m_arr[warp_id], x);
        float em = expf(m_arr[warp_id] - nm);
        float eb = expf(x - nm);
        l_arr[warp_id] = fmaf(l_arr[warp_id], em, eb);
        m_arr[warp_id] = nm;
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    // Reduce per-warp m/l into block-wide softmax stats.
    float m_local = (lane < NUM_WARPS) ? m_arr[lane] : -1e9f;
    float l_local = (lane < NUM_WARPS) ? l_arr[lane] : 0.0f;

    for (int offset = 2; offset > 0; offset >>= 1) {
      float m_other = __shfl_down_sync(0xffffffff, m_local, offset);
      float l_other = __shfl_down_sync(0xffffffff, l_local, offset);
      if (lane + offset < NUM_WARPS) {
        float nm = fmaxf(m_local, m_other);
        float em1 = expf(m_local - nm);
        float em2 = expf(m_other - nm);
        l_local = l_local * em1 + l_other * em2;
        m_local = nm;
      }
    }

    if (lane == 0) {
      m = m_local;
      l = l_local;
    }
  }
  __syncthreads();

  for (int d = lane; d < D; d += WARP_SIZE) {
    out_partial[warp_id * D + d] = 0.0f;
  }

  float thr_scalar = thr_shared;
  int last_idx = last_idx_shared;
  for (int t0 = warp_offset; t0 < t_max; t0 += BLOCK_N * NUM_WARPS) {
    int bs = min(BLOCK_N, t_max - t0);

    float p = 0.0f;
    int keep = 0;
    if (lane < bs) {
      int token_idx = t0 + lane;
      float logit = __half2float(logits_gmem[bh * (size_t)T + token_idx]);
      float p_f = expf(logit - m) / l;
      // Always keep diagonal; threshold compare on post-softmax weights.
      if (token_idx == last_idx) {
        keep = 1;
      } else if (p_f >= thr_scalar) {
        keep = 1;
      } else {
        p_f = 0.0f;
      }
      // Quantize p to fp16 to match kernel accumulation path.
      half p_h = __float2half(p_f);
      float p_q = __half2float(p_h);
      p = p_q;
    }

    unsigned mask = __ballot_sync(ACTIVE_MASK, keep);
    if (lane == 0) {
      keep_arr[warp_id] += __popc(mask);
    }

    while (mask) {
      int bit = __ffs(mask) - 1;
      float p_i = __shfl_sync(ACTIVE_MASK, p, bit);
      int token_idx = t0 + bit;

      for (int d = lane; d < D; d += WARP_SIZE) {
        float v = __half2float(__ldg(v_base + (size_t)token_idx * D + d));
        out_partial[warp_id * D + d] += p_i * v;
      }

      mask &= mask - 1;
    }
  }

  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int w = 0; w < NUM_WARPS; ++w) {
      sum += out_partial[w * D + d];
    }
    out_ptr[d] = __float2half(sum);
  }

  __syncthreads();
  if (tid == 0) {
    int keep_total = 0;
    for (int w = 0; w < NUM_WARPS; ++w) {
      keep_total += keep_arr[w];
    }
    keep_counts[bh] = (float)keep_total;

    float kept = kept_cum[bh] + (float)keep_total;
    float total = total_cum[bh] + (float)t_max;
    kept_cum[bh] = kept;
    total_cum[bh] = total;

    float r_cum = kept / (total + 1e-6f);
    float e = r_cum - r_target;

    int step = compute_step(t_max, base_cutoff, base_step);
    if (step > 0 && (t_max % step) == 0) {
      float lr_a_val = lr_a[bh];
      float lr_b_val = lr_b[bh];
      float alpha_val = alpha[bh] + lr_a_val * e;
      float b_val = b[bh] + lr_b_val * e;
      alpha_val = fminf(fmaxf(alpha_val, log_a_min), log_a_max);
      b_val = fminf(fmaxf(b_val, b_min), b_max);
      alpha[bh] = alpha_val;
      b[bh] = b_val;

      if (t_max >= base_cutoff) {
        lr_a_val *= anl_a;
        lr_b_val *= anl_b;
        lr_a[bh] = lr_a_val;
        lr_b[bh] = lr_b_val;
      }
    }
  }
}

extern "C" void nowmp_attn_cuda_launcher_gmem(
    const void* Qp, const void* Kp, const void* Vp,
    const float* Bias, void* Outp,
    float* alpha, float* b,
    float* kept_cum, float* total_cum,
    float* lr_a, float* lr_b,
    const int* token_counter,
    float r_target,
    float log_a_min,
    float log_a_max,
    float b_min,
    float b_max,
    int base_cutoff,
    int base_step,
    float anl_a,
    float anl_b,
    float scale,
    void* logits_gmem,
    float* keep_counts,
    int B, int H, int T, int D,
    cudaStream_t stream
) {
  const __half* Qh = reinterpret_cast<const __half*>(Qp);
  const __half* Kh = reinterpret_cast<const __half*>(Kp);
  const __half* Vh = reinterpret_cast<const __half*>(Vp);
        __half* Oh = reinterpret_cast<      __half*>(Outp);
        __half* Lh = reinterpret_cast<      __half*>(logits_gmem);

  int blocks = B * H;
  int threads = 128;
  size_t q_bytes = size_t(D) * sizeof(__half);
  size_t off = (q_bytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
  constexpr int NUM_WARPS = 4;
  size_t out_bytes = size_t(NUM_WARPS) * D * sizeof(float);
  size_t dyn_shm = off + out_bytes;

  nowmp_decode_attn_gmem_logits_cache
    <<<blocks, threads, dyn_shm, stream>>>(
      Qh,
      Kh,
      Vh,
      Bias,
      Oh,
      alpha,
      b,
      kept_cum,
      total_cum,
      lr_a,
      lr_b,
      token_counter,
      r_target,
      log_a_min,
      log_a_max,
      b_min,
      b_max,
      base_cutoff,
      base_step,
      anl_a,
      anl_b,
      scale,
      Lh,
      keep_counts,
      B, H, T, D
    );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}
