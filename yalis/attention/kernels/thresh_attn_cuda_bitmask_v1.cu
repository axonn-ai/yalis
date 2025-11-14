#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
using namespace nvcuda;
using namespace nvcuda::wmma;

struct __align__(4) PackedHalfIndex {
  __half score;     // 16 bits
  uint16_t index;   // 16 bits
};
static_assert(sizeof(PackedHalfIndex) == 4, "Size must be 4 bytes");

__device__ void write_entry(PackedHalfIndex* buf, int i, __half val, uint16_t idx) {
  buf[i].score = val;
  buf[i].index = idx;
}

__device__ void read_entry(const PackedHalfIndex* buf, int i, __half& val, uint16_t& idx) {
  val = buf[i].score;
  idx = buf[i].index;
}

__device__ __forceinline__ bool bit_is_set(const uint32_t* mask_words, int i) {
    int w   = i >> 5;          // i / 32
    int bit = i & 31;          // i % 32
    uint32_t mword = mask_words[w];
    return (mword >> bit) & 1u;
}

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

  // STATIC shared for the 16×16 A-tile
  __shared__ __align__(32) half a_tile[8][16];
  __shared__ __align__(32) float C_tile[NUM_WARPS][16];
  __shared__ float m_arr[NUM_WARPS], l_arr[NUM_WARPS];
  __shared__ float m, l;
  __shared__ int valid_cnt;

  // DYNAMIC shared: a single byte array
  extern __shared__ uint8_t _smem[];
  // carve it up
  half* exp_buf = reinterpret_cast<half*>(_smem);

  int num_mask_words = (T + 31) / 32;
  uint32_t* mask_words = reinterpret_cast<uint32_t*>(_smem + T * sizeof(half));

  half* q_half = reinterpret_cast<half*>(
      _smem + T * sizeof(half) + num_mask_words * sizeof(uint32_t)
  );

  //INIT
  if (tid < D) {
    q_half[tid] = __ldg(q_ptr + tid);   // fp16 → fp16
  }

  m_arr[warp_id] = -1e9f;
  l_arr[warp_id] = 0.0f;
  m = -1e9f;
  l = 0.0f;

  // WMMA fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,       16,16,16, half, row_major>  Afrag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,       16,16,16, half, col_major>  Bfrag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,    16,16,16, float>            Cfrag;


  // Load a_tile once, each thread writes one value of q_half to a_tile 
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
      for (int k = 0; k < 16; ++k) {
        a_tile[i][k] = q_half[i * 16 + k];
      }
  }


  constexpr int BLOCK_N = 16;
  int warp_offset = warp_id * BLOCK_N;
  for (int t0 = warp_offset; t0 < T; t0 += BLOCK_N * 4) {
    int bs = min(BLOCK_N, T - t0);
    
    wmma::fill_fragment(Cfrag, 0);
    
    // PASS 1: Compute scaled dot-product attention with WMMA
    for (int d0 = 0; d0 < D; d0 += 16) {
      // load into WMMA
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);

      wmma::load_matrix_sync(Afrag, &a_tile[d0 / 16][0], 0);
      //half const* k_tile = k_base + (size_t)(d0 * T + t0);
      // Perform matrix multiply
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);
    }

    // Store the C tile to shared memory
    wmma::store_matrix_sync(&C_tile[warp_id][0], Cfrag, 0, nvcuda::wmma::mem_row_major);

    if (lane < bs) {
      C_tile[warp_id][lane] = C_tile[warp_id][lane] * scale + __ldg(bias_ptr + t0 + lane);
    }

    // Pass 2: Compute max and sum for softmax
    if (lane == 0) {
      // Compute the running max and sum of the exp(x - max)
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

  // Compute softmax using global m,l and construct bitmask (no atomics)
  for (int w = warp_id; w < num_mask_words; w += NUM_WARPS) {
      int i = w * 32 + lane;
      float p = 0.0f;

      int   keep = 0;  // needs to be defined for all lanes
      if (i < T) {
          float logit = __half2float(exp_buf[i]);     // logits
          p = expf(logit - m) / l;                    // softmax

          // exp_buf[i] currently holds logits x_i from the WMMA pass
          // exp_buf[i] = __float2half(expf(__half2float(exp_buf[i]) - m) / l);
          // exp_buf[i] = __half2float(exp_buf[i]) < thr_scalar ? __float2half(0.0f) : exp_buf[i];
          if (p < thr_scalar) {
            p = 0.0f;
          }
          exp_buf[i] = __float2half(p); 
          keep = p > 0.0f;
      }

      // Warp-wide ballot gives us the 32-bit mask for this word
      unsigned int bits = __ballot_sync(0xffffffff, keep);

      // One lane writes the mask word; no atomics needed
      if (lane == 0 && w < num_mask_words) {
          mask_words[w] = bits;
      }
  }
  __syncthreads();

  // for (int i = threadIdx.x; i < T; i += blockDim.x) {
  //   // exp_buf[i] currently holds logits in half from pass 1
  //   float logit = __half2float(exp_buf[i]);   // logits (float)
  //   float p     = expf(logit - m) / l;        // softmax in float

  //   // Threshold in float32 space (matches your Python ref behavior)
  //   if (p < thr_scalar) {
  //       p = 0.0f;
  //   }

  //   // Store final prob (0 or >0) as half
  //   exp_buf[i] = __float2half(p);
  // }
  // __syncthreads();

  // // === Build bitmask from exp_buf (no new math) ===
  // for (int w = warp_id; w < num_mask_words; w += NUM_WARPS) {
  //     int i = w * 32 + lane;

  //     int keep = 0;
  //     if (i < T) {
  //         // At this point, exp_buf[i] is final prob (0 or >= thr), in half
  //         keep = __half2float(exp_buf[i]) > 0.0f;
  //     }

  //     // Each warp builds its 32-bit word
  //     unsigned int bits = __ballot_sync(0xffffffff, keep);

  //     if (lane == 0 && w < num_mask_words) {
  //         mask_words[w] = bits;
  //     }
  // }
  // __syncthreads();

  // Weighted sum with V
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int i = 0; i < T; ++i) {
      // cheap mask test + early continue
      if (!bit_is_set(mask_words, i)) continue;

      float p = __half2float(exp_buf[i]);
      float v = __half2float(__ldg(v_base + i * D + d));
      sum += p * v;
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
  int num_mask_words = (T + 31) / 32;
  size_t dyn_shm = size_t(T)*sizeof(__half)
                 + size_t(num_mask_words)*sizeof(uint32_t)
                 + size_t(D)*sizeof(__half);


  decode_attn_two_pass_wmma
    <<<blocks, threads, dyn_shm>>>(
      Qh, Kh, Vh, Bias, Oh, Thr, scale, B, H, T, D
    );

  // error‐check for async launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
}