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

  int idx = blockIdx.x;
  if (idx >= B*H) return;

  // Base pointers into Q/K/V/Bias/Out
  const half*  q_ptr    = Q    + idx*(size_t)D;
  const half*  k_base   = K    + idx*(size_t)T*D;
  const half*  v_base   = V    + idx*(size_t)T*D;
  const float* bias_ptr = Bias + idx*(size_t)T;
        half*  out_ptr  = Out  + idx*(size_t)D;
  float thr_scalar      = Thr[idx];

  // STATIC shared for the 16×16 A-tile
  __shared__ __align__(32) half a_tile[16][16];
  __shared__ float C_tile[16][16];
  __shared__ float qk_blk[16];
  __shared__ float m, l;
  __shared__ int valid_cnt;
  if (threadIdx.x == 0) {
    m = -1e9f;
    l = 0.0f;
  }

  // DYNAMIC shared: a single byte array
  extern __shared__ uint8_t _smem[];
  // carve it up:
  float* exp_buf = reinterpret_cast<float*>(_smem);                               // T floats
  float* valid_idx = reinterpret_cast<float*>(_smem +     T * sizeof(float));     // T floats
  float* acc     = reinterpret_cast<float*>(_smem +     2 * T * sizeof(float));       // D floats
  half*  q_half  = reinterpret_cast<half* >( _smem + (2 * T + D) * sizeof(float) );   // D halves

  int lane = threadIdx.x;  // 0..31

  // --- 1) INIT ---
  for (int d = lane; d < D; d += 32) {
    if (d < D) {
      q_half[d] = q_ptr[d];   // fp16 → fp16
      acc[d]    = 0.0f;       // accumulator in fp32
    }
  }
  //if (lane < min(32, D)) {
  //float v = __half2float(q_half[lane]);
  //printf("[block %d lane %2d] q_half[%2d] = %8.5f\n",
  //       blockIdx.x, lane, lane, v);
  //}

  //if (threadIdx.x == 0) {
  //  for (int i = 0; i < D; i++) {
  //    printf("q_half[%d] = %f\n", i, __half2float(q_half[i]));
  //  }
  //}
  // ------ Correct upto this point ------

  // WMMA fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,       16,16,16, half, row_major> Afrag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,       16,16,16, half, col_major>  Bfrag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator,    16,16,16, float>            Cfrag;
  //if (threadIdx.x == 0 && blockIdx.x == 0) {
  //  printf("Fragment sizes: Afrag=%d elements, Bfrag=%d elements, Cfrag=%d elements\n", 
  //         Afrag.num_elements, Bfrag.num_elements, Cfrag.num_elements);
  //}
  

  // --- 2) PASS 1: compute (m,l) & buffer exp() ---
  constexpr int BLOCK_N = 16;
  for (int t0 = 0; t0 < T; t0 += BLOCK_N) {
    int bs = min(BLOCK_N, T - t0);

    // Debug: Print basic block information
    //if (threadIdx.x == 0 && blockIdx.x == 0) {
    //  printf("Processing t0=%d, bs=%d, block=%d, thread=%d\n", 
    //         t0, bs, blockIdx.x, threadIdx.x);
    //}

    wmma::fill_fragment(Cfrag, 0);

    // tile over D in chunks of 16
    for (int d0 = 0; d0 < D; d0 += 16) {
      // Debug: Print current D chunk
      // if (threadIdx.x == 0 && blockIdx.x == 0) {
      //   printf("Processing D chunk starting at d0=%d\n", d0);
      // }

      // Debug: Print Q values being loaded
      //if (lane < 16 && threadIdx.x == 0 && blockIdx.x == 0) {
      //  printf("Thread 0, loading q_half[%d] = %f\n", 
      //         d0 + lane, __half2float(q_half[d0 + lane]));
      //}

      // replicate the 1×16 slice of Q
      if (lane < 16) {
        half v = q_half[d0 + lane];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
          a_tile[i][lane] = v;
        }
      }
      __syncthreads();

      // Debug: Print a_tile contents
      //if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0 && d0 == 0) {
      //  printf("a_tile[0-3][0-3] sample:\n");
      //  for (int i = 0; i < 16; ++i) {
      //    for (int j = 0; j < 16; ++j) {
      //      printf("%6.3f, ", __half2float(a_tile[i][j]));
      //    }
      //    printf("\n");
      //  }
      //}

      // Debug: Print K values being loaded
      //if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0 && d0 == 0) {
      //  half const* k_tile = k_base + (size_t)(t0 * D + d0);
      //  printf("k_tile address = %p, k_base = %p, offset = %zu\n", 
      //         k_tile, k_base, (size_t)(t0 * D + d0));
      //  printf("k_tile[0-3][0-3] sample:\n");
      //  for (int i = 0; i < 16; ++i) {
      //    for (int j = 0; j < 16; ++j) {
      //      printf("%6.3f, ", __half2float(k_tile[i*D + j]));
      //    }
      //    printf("\n");
      //  }
      //}

      // load into WMMA
      wmma::load_matrix_sync(Afrag, &a_tile[0][0], 16);
      half const* k_tile = k_base + (size_t)(t0 * D + d0);
      wmma::load_matrix_sync(Bfrag, (const half*)k_tile, D);

      // Debug: Check if WMMA load succeeded
      //if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0 && d0 == 0) {
      //  bool afrag_ok = true, bfrag_ok = true;
      //  // Check for NaN or Inf in fragments
      //  for (int i = 0; i < Afrag.num_elements; ++i) {
      //    float val = __half2float(Afrag.x[i]);
      //    if (isnan(val) || isinf(val)) {
      //      afrag_ok = false;
      //      printf("ERROR: Afrag[%d] = %f is invalid\n", i, val);
      //      break;
      //    }
      //  }
      //  for (int i = 0; i < Bfrag.num_elements; ++i) {
      //    float val = __half2float(Bfrag.x[i]);
      //    if (isnan(val) || isinf(val)) {
      //      bfrag_ok = false;
      //      printf("ERROR: Bfrag[%d] = %f is invalid\n", i, val);
      //      break;
      //    }
      //  }
      //  printf("WMMA fragment load - Afrag: %s, Bfrag: %s\n", 
      //         afrag_ok ? "OK" : "ERROR", bfrag_ok ? "OK" : "ERROR");
      //}

      __syncthreads();

      //if (blockIdx.x == 0 && t0 == 0 && d0 == 0) {
      //  //if (lane < 16) {
      //  //  for (int i = 0; i < Afrag.num_elements; ++i) {
      //  //    printf("lane %2d: Afrag[%2d] %f\n", lane, i, __half2float(Afrag.x[i]));
      //  //  }
      //  //}

      //  //if (lane < 32) {
      //  //  for (int i = 0; i < Bfrag.num_elements; ++i) {
      //  //    printf("lane %2d: Bfrag[%2d] %f\n", lane, i, __half2float(Bfrag.x[i]));
      //  //  }
      //  //}
      //}

      // Perform matrix multiply
      wmma::mma_sync(Cfrag, Afrag, Bfrag, Cfrag);

    }
    __syncthreads();

    wmma::store_matrix_sync(&C_tile[0][0], Cfrag, 16, nvcuda::wmma::mem_row_major);

    __syncthreads();
    // if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0) {
    //    for (int i = 0; i < 16; ++i) {
    //      for (int j = 0; j < 16; ++j) {
    //        printf("%6.3f ", C_tile[i][j]);
    //      }
    //      printf("\n");
    //    }
    //}

    // Debug: Check extraction indices
    // if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0) {
    //   printf("Extracting from Cfrag - bs=%d, Cfrag has %lu elements\n", bs, Cfrag.num_elements);
    // }

    if (lane < bs) {
      // Debug: Check extraction access pattern
      // if (blockIdx.x == 0 && t0 == 0 && lane < 4) {
      //   printf("Extracting Cfrag[%d] = %f for lane %d\n", lane, C_tile[0][lane], lane);
      // }
      qk_blk[lane] = C_tile[0][lane] * scale + bias_ptr[t0 + lane];

    }
    __syncthreads();

    // Debug: Print extracted values
    // if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0 && lane < 4) {
    //   for (int i = 0; i < 4; ++i) {
    //     printf("Extracted: qk_blk[%d] = %f (scale=%f, bias=%f)\n", 
    //            i, qk_blk[i], scale, bias_ptr[t0 + i]);
    //     }
    // }

    if (lane == 0) {
      // Debug: Show softmax calculation
      //if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0) {
      //  printf("Lane 0 processing softmax for block t0=%d:\n", t0);
      //  for (int i = 0; i < min(4, bs); ++i) {
      //    printf("  qk_blk[%d] = %f\n", i, qk_blk[i]);
      //  }
      //}

      // Compute the running max and sum of the exp(x - max)
      for (int i = 0; i < bs; ++i) {
        float x  = qk_blk[i];
        float nm = fmaxf(m, x);
        float em = expf(m  - nm);
        float eb = expf(x  - nm);
        l = l*em + eb;
        m = nm;
        exp_buf[t0 + i] = x;

        // Debug: Show softmax values
        //if (threadIdx.x == 0 && blockIdx.x == 0 && t0 == 0 && i < 4) {
        //  printf("  softmax[%d]: x=%f, max=%f, exp(x-max)=%f, running_sum=%f\n", 
        //         i, x, m, eb, l);
        //}
      }
    }
    __syncthreads();
  }

  // Compute the softmax using max and sum 
  for (int i = threadIdx.x; i < T; i += 32) {
      exp_buf[i] = expf(exp_buf[i] - m) / l;
      exp_buf[i] = exp_buf[i] < thr_scalar ? 0.0f : exp_buf[i];
  }
  
  if (threadIdx.x == 0) {
      valid_cnt = 0;
      for (int i = 0; i < T; ++i) {
          // Pack the indices that are greater than 0 into valid_idx
          if (exp_buf[i] > 0.0f) {
              valid_idx[valid_cnt] = i;
              valid_cnt++;
          }
      }
  }



  // Debug: Print final softmax values
  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //  printf("Final values: max=%f, sum=%f\n", m, l);
  //  printf("First few exp_buf values:\n");
  //  for (int i = 0; i < min(16, T); ++i) {
  //    printf("  exp_buf[%d] = %f\n", i, exp_buf[i]);
  //  }
  //}

  __syncthreads();

  // --- 3) PASS 2: threshold & accumulate V ---
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float sum = 0.0f;
    for (int t = 0; t < valid_cnt; ++t) {
        int idx = valid_idx[t];
        float s = exp_buf[idx];                    // s ∈ [1 x T]
        float v = __half2float(v_base[idx * D + d]); // v ∈ [T x D]
        sum += s * v;
    }
    acc[d] = sum;
  }

//  for (int t0 = 0; t0 < T; t0 += BLOCK_N) {
//    int bs = min(BLOCK_N, T - t0);
//    if (lane < bs) {
//      float e = exp_buf[t0 + lane];
//      if (e >= thresh) {
//        for (int d = lane; d < D; d += 32) {
//          float v = __half2float( v_base[(size_t)(t0+lane)*D + d] );
//          acc[d] += e * v;
//        }
//      }
//    }
//    __syncthreads();
//  }

  // --- 4) FINALIZE ---
  for (int d = lane; d < D; d += 32) {
    float out_f = acc[d];
    out_ptr[d]  = __float2half(out_f);
  }
}

// our two‐pass + WMMA kernel from before:
//extern "C"
//__global__ void decode_attn_two_pass_wmma(
//    const float* __restrict__ Q,         // [B*H*D]
//    const float* __restrict__ K,         // [B*H*T*D]
//    const float* __restrict__ V,         // [B*H*T*D]
//    const float* __restrict__ Bias,      // [B*H*T]
//    float*       __restrict__ Out,       // [B*H*D]
//    const float* __restrict__ Threshold, // [B*H]
//    float        scale,                  // sm_scale
//    int B, int H, int T, int D
//) {
//  // one warp per (batch,head)
//  int idx = blockIdx.x;  
//  if (idx >= B*H) return;
//  const float* q_ptr    = Q    + idx*D;
//  const float* k_base   = K    + idx*T*D;
//  const float* v_base   = V    + idx*T*D;
//  const float* bias_ptr = Bias + idx*T;
//        float* out_ptr  = Out  + idx*D;
//  float thr_scalar      = Threshold[idx];
//
//  __shared__ half a_tile[16][16];
//
//  extern __shared__ float smem[];
//  float* exp_buf = smem;            // [T]
//  float* acc     = smem + T;        // [D]
//  __half*  q_half  = (__half*)(smem + T + D); // [D]
//
//
//  int lane = threadIdx.x;           // 0..31
//
//  // load Q into half & zero‐init acc[]
//  for (int d = lane; d < D; d += 32) {
//    q_half[d] = __float2half(q_ptr[d]);
//    acc[d]    = 0.0f;
//  }
//  __syncthreads();
//
//  // PASS 1: build (m,l) and buffer exp(...)
//  float m = -1e9f, l = 0.0f;
//  constexpr int BLOCK_N = 16;
//  for (int t0 = 0; t0 < T; t0 += BLOCK_N) {
//    // WMMA fragments
//      // correctly declare the WMMA fragments:
//    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> A;
//    wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> Bm;
//    wmma::fragment<wmma::accumulator,16,16,16,float> C;
//    wmma::fill_fragment(C, 0.0f);
//
//    // tile over D in chunks of 16
//    for (int d0 = 0; d0 < D; d0 += 16) {
//      // replicate 1×16 q_half[d0..d0+15] into a_tile[0..15][0..15]
//      if (lane < 16) {
//        half v = q_half[d0 + lane];
//        #pragma unroll
//        for (int i = 0; i < 16; ++i)
//          a_tile[i][lane] = v;
//      }
//      __syncthreads();
//
//      // load A and B tiles
//      wmma::load_matrix_sync(A, &a_tile[0][0], 16);
//      half const* k_tile = reinterpret_cast<half const*>(
//        k_base + (t0*D + d0)
//      );
//      wmma::load_matrix_sync(Bm, k_tile, D);
//
//      // Cfrag += Afrag @ Bfrag
//      wmma::mma_sync(C, A, Bm, C);
//      __syncthreads();
//    }
//
//    // extract qk[t0..t0+15]
//    float qk_blk[BLOCK_N];
//    if (lane < BLOCK_N) {
//      qk_blk[lane] = C.x[lane]*scale + bias_ptr[t0+lane];
//    }
//    __syncthreads();
//
//    // thread 0 reduces + buffers
//    if (lane == 0) {
//      #pragma unroll
//      for (int i = 0; i < BLOCK_N; ++i) {
//        float x  = qk_blk[i];
//        float nm = fmaxf(m, x);
//        float em = expf(m - nm);
//        float eb = expf(x - nm);
//        l = l*em + eb;
//        m = nm;
//        exp_buf[t0 + i] = eb;
//      }
//    }
//    __syncthreads();
//
//
//    // tile D in chunks of 16
//    //for (int d0 = 0; d0 < D; d0 += 16) {
//    //  wmma::load_matrix_sync(A, reinterpret_cast<half*>(q_half + d0), 16);
//    //  half const* k_tile = reinterpret_cast<half const*>(k_base + (t0*D + d0));
//    //  wmma::load_matrix_sync(Bm, k_tile, D);
//    //  wmma::mma_sync(C, A, Bm, C);
//    //}
//    //__syncthreads();
//
//    //// extract the 16 results and add bias
//    //float qk_blk[BLOCK_N];
//    //if (lane < BLOCK_N)
//    //  qk_blk[lane] = C.x[lane]*scale + bias_ptr[t0 + lane];
//    //__syncthreads();
//
//    //// thread-0 reduces and buffers
//    //if (lane == 0) {
//    //  for (int i = 0; i < BLOCK_N; ++i) {
//    //    float x    = qk_blk[i];
//    //    float nm   = fmaxf(m, x);
//    //    float em   = expf(m - nm);
//    //    float eb   = expf(x - nm);
//    //    l = l*em + eb;   
//    //    m = nm;
//    //    exp_buf[t0 + i] = eb;
//    //  }
//    //}
//    //__syncthreads();
//  }
//
//  // --- PASS 2: exact threshold & accumulate V ---
//  float thresh = thr_scalar * l;
//  for (int t0 = 0; t0 < T; t0 += BLOCK_N) {
//    if (lane < BLOCK_N) {
//      float e = exp_buf[t0 + lane];
//      if (e >= thresh) {
//        for (int d = lane; d < D; d += 32) {
//          // load V in fp16, convert to fp32
//          float v = v_base[(t0+lane)*D + d];
//          acc[d] += e * v;
//        }
//      }
//    }
//    __syncthreads();
//  }
//
//  // --- FINALIZE: divide by l, cast to fp16, store ---
//  for (int d = lane; d < D; d += 32) {
//    float o = acc[d] / l;
//    out_ptr[d] = o;
//  }
//}

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
  int threads = 32;
  size_t dyn_shm = size_t(T)*sizeof(float)
                 + size_t(T)*sizeof(float)
                 + size_t(D)*sizeof(float)
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