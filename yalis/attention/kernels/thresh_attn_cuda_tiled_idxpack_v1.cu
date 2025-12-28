#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
using namespace nvcuda;
using namespace nvcuda::wmma;

extern "C" __global__ void decode_attn_two_pass_tile_fused_idxpack_v1(
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
  constexpr int WARP_SIZE = 32;
  constexpr int BLOCK_N   = 16;
  constexpr unsigned ACTIVE_MASK = (1u << BLOCK_N) - 1u;

  int idx = blockIdx.x;
  if (idx >= B * H) return;

  // Base pointers into Q/K/V/Bias/Out
  const half*  q_ptr    = Q    + idx * (size_t)D;
  const half*  k_base   = K    + idx * (size_t)T * D;
  const half*  v_base   = V    + idx * (size_t)T * D;
  const float* bias_ptr = Bias + idx * (size_t)T;
        half*  out_ptr  = Out  + idx * (size_t)D;
  float thr_scalar      = Thr[idx];

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;

  // STATIC shared for the 16x16 A-tile and per-warp C tiles
  __shared__ __align__(32) half a_tile[8][16];
  __shared__ __align__(32) float C_tile[NUM_WARPS][16];
  __shared__ float m_arr[NUM_WARPS], l_arr[NUM_WARPS];
  __shared__ float m, l;
  __shared__ float inv_l;
  // Tile-local idx-pack buffers: O(NUM_WARPS * BLOCK_N), no O(T) storage.
  __shared__ __align__(16) uint16_t tile_idx[NUM_WARPS][BLOCK_N];
  __shared__ __align__(16) half tile_p[NUM_WARPS][BLOCK_N];

  // DYNAMIC shared: only O(D) scratch (no O(T) buffers)
  extern __shared__ uint8_t _smem[];
  half* q_half = reinterpret_cast<half*>(_smem);
  size_t q_bytes = size_t(D) * sizeof(half);
  size_t off = (q_bytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
  float* out_partial = reinterpret_cast<float*>(_smem + off);

  // Load Q into shared scratch.
  if (tid < D) {
    q_half[tid] = __ldg(q_ptr + tid);   // fp16 -> fp16
  }
  __syncthreads();

  // WMMA fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,       16,16,16, half, row_major>  Afrag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,       16,16,16, half, col_major>  Bfrag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,    16,16,16, float>            Cfrag;

  // Load a_tile once from shared Q.
  if (tid < D) {
    int i = tid / 16;
    int k = tid % 16;
    a_tile[i][k] = q_half[tid];
  }
  __syncthreads();

  m_arr[warp_id] = -1e9f;
  l_arr[warp_id] = 0.0f;
  m = -1e9f;
  l = 0.0f;

  int warp_offset = warp_id * BLOCK_N;

  // PASS 1: compute online softmax stats (m,l) without storing logits
  for (int t0 = warp_offset; t0 < T; t0 += BLOCK_N * NUM_WARPS) {
    int bs = min(BLOCK_N, T - t0);

    wmma::fill_fragment(Cfrag, 0);
    for (int d0 = 0; d0 < D; d0 += 16) {
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);
      wmma::load_matrix_sync(Afrag, &a_tile[d0 / 16][0], 0);
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }

    wmma::store_matrix_sync(&C_tile[warp_id][0], Cfrag, 0, nvcuda::wmma::mem_row_major);

    if (lane < bs) {
      C_tile[warp_id][lane] = C_tile[warp_id][lane] * scale + __ldg(bias_ptr + t0 + lane);
    }

    if (lane == 0) {
      for (int i = 0; i < bs; ++i) {
        float x  = C_tile[warp_id][i];
        float nm = fmaxf(m_arr[warp_id], x);
        float em = expf(m_arr[warp_id] - nm);
        float eb = expf(x - nm);
        l_arr[warp_id] = fmaf(l_arr[warp_id], em, eb);
        m_arr[warp_id] = nm;
      }
    }
  }

  // Reduce across warps to get block max and sum (match base kernel)
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
    inv_l = 1.0f / l_final;
  }
  __syncthreads();

  // PASS 2: recompute logits, threshold, and immediately accumulate V
  // No O(T) shared buffers: only O(D) partial sums per warp + per-tile idx-pack.
  for (int d = lane; d < D; d += WARP_SIZE) {
    out_partial[warp_id * D + d] = 0.0f;
  }
  __syncwarp();

  for (int t0 = warp_offset; t0 < T; t0 += BLOCK_N * NUM_WARPS) {
    int bs = min(BLOCK_N, T - t0);

    wmma::fill_fragment(Cfrag, 0);
    for (int d0 = 0; d0 < D; d0 += 16) {
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);
      wmma::load_matrix_sync(Afrag, &a_tile[d0 / 16][0], 0);
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }

    wmma::store_matrix_sync(&C_tile[warp_id][0], Cfrag, 0, nvcuda::wmma::mem_row_major);

    float p = 0.0f;
    int keep = 0;
    if (lane < bs) {
      float logit = C_tile[warp_id][lane] * scale + __ldg(bias_ptr + t0 + lane);
      p = expf(logit - m) * inv_l;
      if (p < thr_scalar) {
        p = 0.0f;
      }
      keep = (p > 0.0f);
    }

    // Tile-local idx packing: ballot + rank into compact per-warp list.
    unsigned mask = __ballot_sync(ACTIVE_MASK, keep);
    int nnz = __popc(mask);
    if (keep && lane < BLOCK_N) {
      int rank = __popc(mask & ((1u << lane) - 1u));
      tile_idx[warp_id][rank] = static_cast<uint16_t>(t0 + lane);
      tile_p[warp_id][rank] = __float2half(p);
    }
    __syncwarp();

    // Accumulate in registers per d, write back once per tile (reduces shared traffic).
    if ((D & 1) == 0) {
      for (int d = lane * 2; d < D; d += WARP_SIZE * 2) {
        float acc0 = out_partial[warp_id * D + d];
        float acc1 = out_partial[warp_id * D + d + 1];
        for (int j = 0; j < nnz; ++j) {
          int token_idx = 0;
          float p_j = 0.0f;
          if (lane == 0) {
            token_idx = tile_idx[warp_id][j];
            p_j = __half2float(tile_p[warp_id][j]);
          }
          token_idx = __shfl_sync(0xffffffff, token_idx, 0);
          p_j = __shfl_sync(0xffffffff, p_j, 0);

          const half2* v2_ptr = reinterpret_cast<const half2*>(
              v_base + (size_t)token_idx * D + d);
          half2 v2 = *v2_ptr;
          float2 v2f = __half22float2(v2);
          acc0 += p_j * v2f.x;
          acc1 += p_j * v2f.y;
        }
        out_partial[warp_id * D + d] = acc0;
        out_partial[warp_id * D + d + 1] = acc1;
      }
    } else {
      for (int d = lane; d < D; d += WARP_SIZE) {
        float acc = out_partial[warp_id * D + d];
        for (int j = 0; j < nnz; ++j) {
          int token_idx = 0;
          float p_j = 0.0f;
          if (lane == 0) {
            token_idx = tile_idx[warp_id][j];
            p_j = __half2float(tile_p[warp_id][j]);
          }
          token_idx = __shfl_sync(0xffffffff, token_idx, 0);
          p_j = __shfl_sync(0xffffffff, p_j, 0);

          float v = __half2float(__ldg(v_base + (size_t)token_idx * D + d));
          acc += p_j * v;
        }
        out_partial[warp_id * D + d] = acc;
      }
    }
  }

  // Reduce across warps to get final output
  __syncthreads();
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int w = 0; w < NUM_WARPS; ++w) {
      sum += out_partial[w * D + d];
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
  const __half* Qh = reinterpret_cast<const __half*>(Qp);
  const __half* Kh = reinterpret_cast<const __half*>(Kp);
  const __half* Vh = reinterpret_cast<const __half*>(Vp);
        __half* Oh = reinterpret_cast<      __half*>(Outp);

  int blocks  = B * H;
  int threads = 128;
  size_t q_bytes = size_t(D) * sizeof(__half);
  size_t off = (q_bytes + sizeof(float) - 1) & ~(sizeof(float) - 1);
  size_t out_bytes = size_t(4) * D * sizeof(float);
  size_t dyn_shm = off + out_bytes;

  decode_attn_two_pass_tile_fused_idxpack_v1
    <<<blocks, threads, dyn_shm>>>(
      Qh, Kh, Vh, Bias, Oh, Thr, scale, B, H, T, D
    );

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
}
