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
  constexpr int NUM_WARPS = 32;

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

  __shared__ float m_arr[NUM_WARPS], l_arr[NUM_WARPS];
  __shared__ float m, l;
  __shared__ int valid_cnt;

  // DYNAMIC shared: a single byte array
  extern __shared__ uint8_t _smem[];
  // carve it up:
  half* exp_buf = reinterpret_cast<half*>(_smem);                               // T floats
  uint16_t* valid_idx = reinterpret_cast<uint16_t*>(_smem +     T * sizeof(half));     // T floats
  half*  q_half  = reinterpret_cast<half* >( _smem + (T * sizeof(half)) + (T * sizeof(uint16_t)) );   // D halves
  float*  acc     = reinterpret_cast<float* >( _smem + (T * sizeof(half)) + (T * sizeof(uint16_t)) + (D * sizeof(half)) );   // D halves

  // --- 1) INIT ---
  if (tid < D) {
    q_half[tid] = __ldg(q_ptr + tid);   // fp16 → fp16
  }

  m_arr[warp_id] = -1e9f;
  l_arr[warp_id] = 0.0f;
  m = -1e9f;
  l = 0.0f;

  // Initialize accumulator
  for (int d = tid; d < D; d += blockDim.x) {
      acc[d] = 0.0f;
  }

  for (int t = warp_id; t < T; t += NUM_WARPS) {
    float dot = 0.0f;

    for (int d = lane; d < D; d += 32) {
        float q_val = __half2float(q_half[d]);
        float k_val = __half2float(k_base[t * D + d]);
        dot += q_val * k_val;
    }

    // Reduce within warp
    dot += __shfl_xor_sync(0xFFFFFFFF, dot, 16);
    dot += __shfl_xor_sync(0xFFFFFFFF, dot, 8);
    dot += __shfl_xor_sync(0xFFFFFFFF, dot, 4);
    dot += __shfl_xor_sync(0xFFFFFFFF, dot, 2);
    dot += __shfl_xor_sync(0xFFFFFFFF, dot, 1);

    if (lane == 0) {
        float logits = dot * scale + __ldg(bias_ptr + t);
        exp_buf[t] = __float2half(logits);

        float nm = fmaxf(m_arr[warp_id], logits);
        float em = expf(m_arr[warp_id] - nm);
        float eb = expf(logits - nm);
        l_arr[warp_id] = fmaf(l_arr[warp_id], em, eb);
        m_arr[warp_id] = nm;
    }
  }

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


  // Compute the softmax using max and sum 
  for (int i = threadIdx.x; i < T; i += blockDim.x) {
      exp_buf[i] = __float2half(expf(__half2float(exp_buf[i]) - m) / l);
      exp_buf[i] = __half2float(exp_buf[i]) < thr_scalar ? __float2half(0.0f) : exp_buf[i];
  }
  __syncthreads();

  //if (tid == 0) {
  //  for (int i = 0; i < T; ++i) {
  //    printf("%6.3f ", exp_buf[i]);
  //  }
  //  printf("\n");
  //}
  
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

  __syncthreads();

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t < valid_cnt; ++t) {
        int idx = valid_idx[t];
        float s = __half2float(exp_buf[idx]);                    // s ∈ [1 x T]
        float v = __half2float(__ldg(v_base + idx * D + d)); // v ∈ [T x D]
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
    int B, int H, int T, int D,
    cudaStream_t stream
) {
  // cast back to __half*
  const __half* Qh = reinterpret_cast<const __half*>(Qp);
  const __half* Kh = reinterpret_cast<const __half*>(Kp);
  const __half* Vh = reinterpret_cast<const __half*>(Vp);
        __half* Oh = reinterpret_cast<      __half*>(Outp);

  int blocks  = B*H;
  int threads = 1024;
  size_t dyn_shm = size_t(T)*sizeof(__half)
                 + size_t(T)*sizeof(uint16_t)
                 + size_t(D)*sizeof(__half)
                 + size_t(D)*sizeof(float);


  decode_attn_two_pass_wmma
    <<<blocks, threads, dyn_shm, stream>>>(
      Qh, Kh, Vh, Bias, Oh, Thr, scale, B, H, T, D
    );

  // error‐check for async launch
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    printf("kernel launch failed: %s\n", cudaGetErrorString(err));
}




/// Claude's code
// extern "C" __global__ void decode_attn_two_pass_optimized( 
// const half*  __restrict__ Q,       // [B*H, D] 
// const half*  __restrict__ K,       // [B*H, T, D] 
// const half*  __restrict__ V,       // [B*H, T, D] 
// const float* __restrict__ Bias,    // [B*H, T] 
// half*  __restrict__ Out,     // [B*H, D] 
// const float* __restrict__ Thr,     // [B*H] 
// float        scale, int B, int H, int T, int D) { 
//   int idx = blockIdx.x; 
//   if (idx >= B*H) return; 
//   // Base pointers 
//   const half*  q_ptr    = Q    + idx*(size_t)D; 
//   const half*  k_base   = K    + idx*(size_t)T*D; 
//   const half*  v_base   = V    + idx*(size_t)T*D; 
//   const float* bias_ptr = Bias + idx*(size_t)T; 
//   half*  out_ptr  = Out  + idx*(size_t)D; 
//   float thr_scalar      = Thr[idx]; 
//   int tid = threadIdx.x; 
//   int warp_id = tid / 32; 
//   int lane_id = tid % 32; 
// 
//   // Shared memory layout - single buffer for scores/probabilities 
//   extern __shared__ uint8_t _smem[]; 
//   float* score_buf = reinterpret_cast<float*>(_smem);                                                        // T floats (reused) 
//   int*   valid_idx = reinterpret_cast<int*>(_smem + T * sizeof(float));                                     // T ints 
//   float* acc       = reinterpret_cast<float*>(_smem + T * sizeof(float) + T * sizeof(int));                 // D floats 
//   
//   // validity_mask will be placed after acc: (_smem + T * sizeof(float) + T * sizeof(int) + D * sizeof(float)) 
//   // Shared memory for reductions 
//   __shared__ float s_max[32]; 
//   __shared__ float s_sum[32]; 
//   __shared__ int s_valid_count;
//   
//   // Initialize accumulator
//   for (int d = tid; d < D; d += blockDim.x) {
//       acc[d] = 0.0f;
//   }
//   
//   // --- PASS 1: Compute Q·K^T with direct vector operations ---
//   
//   // Each thread processes multiple tokens to ensure coalesced access
//   constexpr int TOKENS_PER_THREAD = 4;
//   
//   for (int t_base = 0; t_base < T; t_base += blockDim.x * TOKENS_PER_THREAD) {
//       
//       // Each thread handles TOKENS_PER_THREAD tokens
//       #pragma unroll
//       for (int i = 0; i < TOKENS_PER_THREAD; i++) {
//           int t = t_base + tid * TOKENS_PER_THREAD + i;
//           
//           if (t < T) {
//               // Direct vector dot product Q·K[t] using cooperative loading
//               float dot_product = 0.0f;
//               
//               // Vectorized dot product with cooperative loading
//               for (int d = 0; d < D; d += 32) {
//                   // Cooperative load Q values (broadcast across warp)
//                   float q_val = (d + lane_id < D) ? __half2float(q_ptr[d + lane_id]) : 0.0f;
//                   
//                   // Load K values with coalesced access
//                   float k_val = (d + lane_id < D) ? __half2float(k_base[t * D + d + lane_id]) : 0.0f;
//                   
//                   // Warp-level reduction for dot product
//                   float prod = q_val * k_val;
//                   
//                   #pragma unroll
//                   for (int offset = 16; offset > 0; offset /= 2) {
//                       prod += __shfl_down_sync(0xFFFFFFFF, prod, offset);
//                   }
//                   
//                   if (lane_id == 0) {
//                       dot_product += prod;
//                   }
//                   
//                   // Broadcast result to all threads in warp
//                   dot_product = __shfl_sync(0xFFFFFFFF, dot_product, 0);
//               }
//               
//               // Apply scale and bias and store
//               score_buf[t] = dot_product * scale + bias_ptr[t];
//           }
//       }
//   }
//   
//   __syncthreads();
//   
//   // --- Online softmax with parallel reduction ---
//   
//   // Initialize per-warp max and sum
//   if (tid < 32) {
//       s_max[tid] = -1e9f;
//       s_sum[tid] = 0.0f;
//   }
//   __syncthreads();
//   
//   // Compute max across all elements (parallel reduction)
//   float local_max = -1e9f;
//   for (int t = tid; t < T; t += blockDim.x) {
//       local_max = fmaxf(local_max, score_buf[t]);
//   }
//   
//   // Warp-level reduction for max
//   #pragma unroll
//   for (int offset = 16; offset > 0; offset /= 2) {
//       local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
//   }
//   
//   if (lane_id == 0) {
//       s_max[warp_id] = local_max;
//   }
//   __syncthreads();
//   
//   // Final reduction across warps
//   if (tid < 32) {
//       local_max = (tid < (blockDim.x + 31) / 32) ? s_max[tid] : -1e9f;
//       #pragma unroll
//       for (int offset = 16; offset > 0; offset /= 2) {
//           local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
//       }
//       if (tid == 0) {
//           s_max[0] = local_max;
//       }
//   }
//   __syncthreads();
//   
//   float global_max = s_max[0];
//   
//   // Compute exp and sum (parallel) - reuse score_buf for probabilities
//   float local_sum = 0.0f;
//   for (int t = tid; t < T; t += blockDim.x) {
//       float exp_val = expf(score_buf[t] - global_max);
//       score_buf[t] = exp_val; // Reuse buffer: raw scores -> exp values
//       local_sum += exp_val;
//   }
//   
//   // Warp-level reduction for sum
//   #pragma unroll
//   for (int offset = 16; offset > 0; offset /= 2) {
//       local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
//   }
//   
//   if (lane_id == 0) {
//       s_sum[warp_id] = local_sum;
//   }
//   __syncthreads();
//   
//   // Final reduction across warps
//   if (tid < 32) {
//       local_sum = (tid < (blockDim.x + 31) / 32) ? s_sum[tid] : 0.0f;
//       #pragma unroll
//       for (int offset = 16; offset > 0; offset /= 2) {
//           local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
//       }
//       if (tid == 0) {
//           s_sum[0] = local_sum;
//       }
//   }
//   __syncthreads();
//   
//   float global_sum = s_sum[0];
//   
//   // Apply softmax normalization and thresholding (parallel)
//   for (int t = tid; t < T; t += blockDim.x) {
//       float prob = score_buf[t] / global_sum;
//       score_buf[t] = (prob >= thr_scalar) ? prob : 0.0f; // Final probabilities
//   }
//   __syncthreads();
//   
//   // --- Parallel valid index collection using prefix sum ---
//   
//   // Step 1: Create validity mask in shared memory
//   int* validity_mask = reinterpret_cast<int*>(_smem + T * sizeof(float) + T * sizeof(int) + D * sizeof(float));
//   for (int t = tid; t < T; t += blockDim.x) {
//       validity_mask[t] = (score_buf[t] > 0.0f) ? 1 : 0;
//   }
//   __syncthreads();
//   
//   // Step 2: Parallel prefix sum to get indices
//   // Simple version - for larger T, use a more sophisticated parallel scan
//   if (tid == 0) {
//       int count = 0;
//       for (int t = 0; t < T; t++) {
//           if (validity_mask[t]) {
//               valid_idx[count++] = t;
//           }
//       }
//       s_valid_count = count;
//   }
//   __syncthreads();
//   
//   int valid_count = s_valid_count;
//   
//   // --- PASS 2: Compute attention output with optimized V multiplication ---
//   
//   // Parallel computation of attention output
//   for (int d = tid; d < D; d += blockDim.x) {
//       float sum = 0.0f;
//       
//       // Vectorized accumulation with better memory access patterns
//       for (int i = 0; i < valid_count; i += 4) { // Process 4 at a time
//           #pragma unroll
//           for (int j = 0; j < 4 && (i + j) < valid_count; j++) {
//               int t_idx = valid_idx[i + j];
//               float prob = score_buf[t_idx]; // Now contains final probabilities
//               float v_val = __half2float(v_base[t_idx * D + d]);
//               sum += prob * v_val;
//           }
//       }
//       
//       out_ptr[d] = __float2half(sum);
//   }
// }
// 
// // Updated launcher with adjusted shared memory size
// extern "C" void decode_attn_cuda_launcher(
//   const void* Qp, const void* Kp, const void* Vp,
//   const float* Bias, void* Outp,
//   const float* Thr, float scale,
//   int B, int H, int T, int D
// ) {
//   const __half* Qh = reinterpret_cast<const __half*>(Qp);
//   const __half* Kh = reinterpret_cast<const __half*>(Kp);
//   const __half* Vh = reinterpret_cast<const __half*>(Vp);
//         __half* Oh = reinterpret_cast<      __half*>(Outp);
// 
//   int blocks  = B * H;
//   int threads = 256; // Increased thread count for better parallelism
//   
//   size_t dyn_shm = size_t(T) * sizeof(float)      // score_buf (reused)
//                  + size_t(T) * sizeof(int)        // valid_idx
//                  + size_t(D) * sizeof(float)      // acc
//                  + size_t(T) * sizeof(int);       // validity_mask
// 
//   decode_attn_two_pass_optimized
//       <<<blocks, threads, dyn_shm>>>(
//           Qh, Kh, Vh, Bias, Oh, Thr, scale, B, H, T, D
//       );
// 
//   cudaError_t err = cudaGetLastError();
//   if (err != cudaSuccess)
//       printf("kernel launch failed: %s\n", cudaGetErrorString(err));
// } 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
using namespace nvcuda;
using namespace nvcuda::wmma;

// nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,       16,16,16, half, row_major>  Vfrag;
//  // --- 3) PASS 2: threshold & accumulate V ---
//  for (int d0 = 0; d0 < D; d0 += BLOCK_N) {
//    int bs = min(BLOCK_N, D - d0);
//
//    wmma::fill_fragment(Cfrag, 0);
//
//    // tile over T in chunks of 16
//    for (int t0 = 0; t0 < T; t0 += BLOCK_N) {
//
//      // replicate the 1×16 slice of V
//      if (lane < 16) {
//        half s = __float2half(exp_buf[t0 + lane]);
//        #pragma unroll
//        for (int i = 0; i < 16; ++i) {
//          a_tile[i][lane] = s;
//        }
//      }
//
//      // load into WMMA
//      wmma::load_matrix_sync(Afrag, &a_tile[0][0], 16);
//      half const* v_tile = v_base + (size_t)(t0 * D + d0);
//      wmma::load_matrix_sync(Vfrag, (const half*)v_tile, D);
//      // Perform matrix multiply
//      wmma::mma_sync(Cfrag, Afrag, Vfrag, Cfrag);
//
//    }
//    
//    // TODO: Is this needed?
//    // __syncthreads();
//
//    wmma::store_matrix_sync(&C_tile[0][0], Cfrag, 16, nvcuda::wmma::mem_row_major);
//
//    if (lane < bs) {
//      acc[d0 + lane] = C_tile[0][lane];
//    }
//    __syncthreads();
//  }
