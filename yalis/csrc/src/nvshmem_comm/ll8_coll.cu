#include "nvshmem_comm/ll8_coll.cuh"

__global__ void bump_seq(uint32_t* seq, uint64_t* signal) {
  int mype  = nvshmem_my_pe();
  int npes  = nvshmem_n_pes();
  int steps = 0;
  while ((1u << steps) < (unsigned)npes) ++steps;
  const uint32_t seq_local = *seq;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
      (*seq)++;
  }

  // Make sure partners are ready
  for (int step = 2; step < steps; ++step) {
    int inter_node_step = step - 2;
    nvshmem_signal_wait_until(signal + inter_node_step, NVSHMEM_CMP_EQ, seq_local + 1);
  }
}

__global__ void init_signal_kernel(uint64_t *signal, size_t n) {
  int tid = threadIdx.x;
  if (tid < n) {
      signal[tid] = 1ULL;
  }
  __threadfence_system();
}


template <typename T>
__global__ void pack_payloads_kernel(Payload8B<T>* __restrict__ payloads,
                                     const T* __restrict__ data,
                                     size_t num_elements,
                                     uint32_t* seq_num)
{
    const size_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = cell_idx * Payload8B<T>::N;
    if (base >= num_elements) return;

    // how many elements go in this cell (tail-aware)
    const size_t ncopy = min((size_t)Payload8B<T>::N, num_elements - base);

    // fill data; zero-pad the tail (optional)
    #pragma unroll
    for (size_t i = 0; i < Payload8B<T>::N; ++i) {
        payloads[cell_idx].data[i] = (i < ncopy) ? data[base + i] : T(0);
    }

    // mark ready inside the same 8B cell
    // (since the entire 8B cell will be sent with put128 as a single "word")
    payloads[cell_idx].flag = *seq_num;
}

// Unpack 8B cells back to a flat array
template <typename T>
__global__ void unpack_payloads_kernel(T* __restrict__ data,
                                       const Payload8B<T>* __restrict__ payloads,
                                       size_t num_elements,
                                       uint32_t* seq_num,
                                       uint64_t* seq_num_signal)
{

    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();
    int steps = 0;
    while ((1u << steps) < (unsigned)npes) ++steps;

    uint64_t local_seq_num = *seq_num;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int partner;
      for (int step = 2; step < steps; ++step) {
          int inter_node_step = step - 2;
          partner = mype ^ (1 << step);
          nvshmem_uint64_atomic_set(seq_num_signal + inter_node_step, local_seq_num + 1, partner);
      }
    }

    const size_t cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t base = cell_idx * Payload8B<T>::N;
    if (base >= num_elements) return;

    const size_t ncopy = min((size_t)Payload8B<T>::N, num_elements - base);

    #pragma unroll
    for (size_t i = 0; i < ncopy; ++i) {
        data[base + i] = payloads[cell_idx].data[i];
    }
}

__device__ __forceinline__ unsigned long long read_globaltimer() {
  #if __CUDA_ARCH__ >= 700
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
  #else
    // Fallback: you’ll need a host-supplied cycle budget on pre-Volta.
    return 0ull;
  #endif
  }

__device__ __forceinline__ uint32_t ld_flag_cg(const volatile uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

template <typename T>
__device__ void process_chunk_warp_all_ready(Payload8B<T>* __restrict__ dst_chunk,
                                             const Payload8B<T>* __restrict__ my_chunk,
                                             size_t chunk_size,
                                             uint32_t seq_num_local,
                                             unsigned long long timeout_ns)
{
  const int lane  = threadIdx.x & 31;
  const int warp  = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;

  for (size_t tile = warp * 32; tile < chunk_size; tile += warps * 32) {
    const size_t idx   = tile + lane;
    const bool   valid = idx < chunk_size;

    // Mask of lanes that actually own a cell in this tile
    unsigned full = __activemask();
    unsigned mask = __ballot_sync(full, valid);

    Payload8B<T>*       dst_cell = valid ? (dst_chunk + idx) : nullptr;
    const Payload8B<T>* src_cell = valid ? (my_chunk  + idx) : nullptr;
    volatile uint32_t*   flag_ptr = valid ? &dst_cell->flag : (volatile uint32_t*)NULL;

    // Cooperative spin until *all valid lanes* see their flag == seq
    const unsigned long long start    = read_globaltimer();
    const unsigned long long deadline = start + timeout_ns; // already in ns

    int backoff = 64;
    while (true) {
      // Each valid lane does an acquire load of *its* flag
      bool ready_lane = !valid || (ld_flag_cg(flag_ptr) == seq_num_local);

      // Did *all* valid lanes report ready?
      unsigned ready_mask = __ballot_sync(full, ready_lane);
      if ((ready_mask & mask) == mask) break;   // all valid lanes ready

      __nanosleep(backoff);
      // backoff = min(backoff << 1, 512);

      if (lane == 0 && read_globaltimer() >= deadline) {
        // (optional) record a rare timeout; avoid heavy printf here
        // e.g., write to a debug buffer
        printf("[pe %d][block %d][thread %d] timeout seq_num=%u got=%u\n",
        (int)nvshmem_my_pe(),
        (int)blockIdx.x,
        (int)threadIdx.x,
        (int)seq_num_local,
        (int)dst_cell->flag);
        break;  // don’t hang forever in production
      }
    }

    // Now it’s safe for each lane to read/update its cell
    if (valid) {
        dst_cell->add_from(*src_cell);
    }
  }
}


// Recursive-doubling all-reduce using NVSHMEM with 16B payloads and flags.
// Templated over data type T.
template <typename T>
__global__ void recursive_allreduce_kernel_payload(Payload8B<T> *dst, Payload8B<T> *src, size_t nreduce,
                                                   size_t chunk_elems, size_t stride_size, uint32_t* seq_num, 
                                                   uint64_t* seq_num_signal) 
  {
    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    constexpr size_t ELEMS_PER_PAYLOAD = Payload8B<T>::N;

    // Compute number of doubling steps = ceil(log2(npes))
    int steps = 0;
    while ((1u << steps) < (unsigned)npes) ++steps;

    // Block‐wise partitioning
    int  block_idx    = blockIdx.x;
    int  thread_id    = threadIdx.x;
    int  num_threads  = blockDim.x;
    int  num_blocks   = gridDim.x;


    const size_t total_cells = (nreduce + ELEMS_PER_PAYLOAD - 1) / ELEMS_PER_PAYLOAD;
    const size_t cells_per_block = total_cells / num_blocks;
    if (cells_per_block * (block_idx) >= total_cells) return;

    const int  lane    = thread_id & (WARP_SIZE - 1);
    const int  warp_id = thread_id >> 5;

    // Slide dst/src pointers to this block’s slice
    size_t dst_offset = block_idx * cells_per_block;
    size_t src_offset = block_idx * cells_per_block;

    // how many cells this block actually owns (handle tail)
    const size_t cells_this_block = min(cells_per_block, total_cells - dst_offset);


    // Chunking inside each step
    size_t chunk_cells = max((size_t)32, (chunk_elems + ELEMS_PER_PAYLOAD - 1) / ELEMS_PER_PAYLOAD);
    size_t num_chunks  = (cells_this_block + chunk_cells - 1) / chunk_cells;

    const uint32_t seq_num_local = *seq_num;


    // if (thread_id == 0 && block_idx == 0) {
      // printf("[%d, %d, %d] seq_num: %u\n", mype, block_idx, thread_id, seq_num_local);
    // }


    int partner;
    // --- recursive-doubling reduce ---
    for (int step = 2; step < steps ; ++step) {
        partner = mype ^ (1 << step);
        int inter_node_step = step - 2;



        // Inform my partner that I am ready to receive their data and wait for them to do the same
        // This is important to ensure that if I have completed my partial sum from the previous partner, 
        // my partner might not have completed their partial sum from the previous partner, and I would
        // overwrite their data with my data.
        // if (thread_id == 0) {
        //     nvshmem_signal_wait_until(signal + inter_node_step, NVSHMEM_CMP_EQ, seq_num_local);
        // }
        // __syncthreads();
        //if (thread_id == 0) {
        //    // Signal my partner that I am ready to receive their data
        //    nvshmem_uint64_atomic_set(signal + step, 1, partner);

        //    // Wait for my partner to signal me that they are ready to receive my data
        //    nvshmem_signal_wait_until(signal + step, NVSHMEM_CMP_GE, 1);
        //}
        //__syncthreads();
        // exchange & reduce each chunk

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
          const size_t chunk_size = min(chunk_cells, cells_this_block - chunk_idx * chunk_cells);
          Payload8B<T>* my_chunk = src + src_offset + chunk_idx * chunk_cells;
          Payload8B<T>* dst_chunk = reinterpret_cast<Payload8B<T>*>(
                  reinterpret_cast<uint8_t*>(dst) + (inter_node_step + 1) * stride_size
              ) + dst_offset + chunk_idx * chunk_cells;
      
          // if (warp_id == 0) {
          //     // Send exactly THIS CHUNK: chunk_size cells × 16 B each.
          //     // Prefer put128 so each element == one 16B “word”
          //     nvshmemx_put128_nbi_block(
          //         (void*)dst_chunk,
          //         (const void*)my_chunk,
          //         chunk_size,
          //         partner);
          //     continue; // comm warp advances to next chunk
          // }
          nvshmemx_put64_nbi_block(
              (void*)dst_chunk,
              (const void*)my_chunk,
              chunk_size,
              partner);

          // if (warp_id == 0) {
          //     nvshmemx_put64_nbi_warp(
          //         (void*)dst_chunk,
          //         (const void*)my_chunk,
          //         chunk_size,
          //         partner);
          //     continue;
          // }
      
          // if (warp_id == 1) {
          //     // All lanes cooperatively poll; each lane takes a strided subset
          //     volatile const Payload8B<T>* rcv = dst_chunk;
      
          //     for (size_t i = lane; i < chunk_cells; i += WARP_SIZE) {
          //         // Spin until THIS cell is ready
          //         while (rcv[i].flag != seq_num) { __nanosleep(64); }
          //     }
          //     // Ensure payload writes from the NIC are visible to all threads
          //     __threadfence_system();
      
          //     if (lane == 0) {
          //         ready_seq = (int)chunk_idx;
          //         __threadfence_block();
          //     }
          //     __syncwarp();
          //     continue;
          // }
          // __syncwarp();
      
          // --- COMPUTE WARPS ---
          // const int compute_threads = num_threads - WARP_SIZE;
          // const int compute_rank    = thread_id - WARP_SIZE;
      
          // Wait until comm warp marks this chunk ready
          // while ((int)ready_seq < (int)chunk_idx) { /* spin */ }

          process_chunk_warp_all_ready(dst_chunk, my_chunk, chunk_size, seq_num_local, POLL_TIMEOUT * 1000000000ULL);
      
          // Reduce cell-wise: unpack → add → repack
          // for (size_t i = thread_id; i < chunk_size; i += num_threads) {
          //     // Check if the flag is correct
          //     Payload8B<T> *dst_cell = dst_chunk + i;
          //     const unsigned long long start = read_globaltimer();
          //     const unsigned long long deadline = start + POLL_TIMEOUT * 1000000000ULL;
          //     volatile uint32_t* flag = &dst_cell->flag;
          //     while (ld_flag_cg(flag) != seq_num_local) { 
          //       __nanosleep(64); 
          //       if (read_globaltimer() >= deadline) {
          //         printf("[pe %d][block %d][thread %d] timeout step=%d partner=%d chunk=%d i=%llu got=%u exp=%u total_cells=%llu signal=%llu\n",
          //         (int)mype,
          //         (int)block_idx,
          //         (int)thread_id,
          //         (int)step,
          //         (int)partner,
          //         (int)chunk_idx,
          //         (unsigned long long)i,                 // size_t -> %llu
          //         (unsigned int)dst_cell->flag,          // uint32_t -> %u
          //         (unsigned int)seq_num_local,           // uint32_t -> %u
          //         (unsigned long long)total_cells,      // size_t -> %llu
          //         (unsigned long long)signal[inter_node_step]);      // size_t -> %llu
          //       }
          //     }
          //     __threadfence_system();
          //     dst_cell->add_from(my_chunk[i]);
          // }
          __syncwarp();
        }

        // The source pointer should be updated to the next step to the current latest buffer
        src = (Payload8B<T>*)((uint8_t*)dst + (inter_node_step + 1) * stride_size);

        __syncthreads();
    }
    __syncthreads();
}


// Class Method implementations
void RecursiveLL8Coll::initialize(int num_blocks, int threads_per_block, size_t chunk_size) {
    steps_ = 0;
    while ((1u << steps_) < (unsigned)npes_) ++steps_;
    steps_intra_ = 0;
    while ((1u << steps_intra_) < (unsigned)npes_node_) ++steps_intra_;
    steps_inter_ = steps_ - steps_intra_;

    set_kernel_params(num_blocks, threads_per_block, chunk_size);
}

void RecursiveLL8Coll::cleanup() {
    for (auto [id, ptr] : allocated_scratch_) {
        nvshmem_free(ptr);
    }
    for (auto [id, ptr] : seq_nums_) {
        cudaFree(ptr);
    }
    for (auto [id, ptr] : seq_num_signals_) {
        nvshmem_free(ptr);
    }
    allocated_scratch_.clear();
    seq_nums_.clear();
    seq_num_signals_.clear();
    stride_sizes_.clear();
}

void RecursiveLL8Coll::register_tensor(uint64_t id, size_t size, torch::Dtype dt, torch::Device dev) {
    size_t local_elems = size / npes_node_;
    size_t per_cell    = elems_per_cell(dt);
    size_t num_cells   = (local_elems + per_cell - 1) / per_cell;
    size_t stride_size = num_cells * sizeof(Payload8B<char>); // always 8

    // Create scratch memory for inter-node communication
    void *scratch = nvshmem_malloc((steps_inter_ + 1) * stride_size);
    if (!scratch) {
        throw std::runtime_error("Failed to allocate scratch memory");
    }
    // Important to zero out the scratch memory
    cudaMemset(scratch, 0, (steps_inter_ + 1) * stride_size);

    // Create sequence number for inter-node communication per tensor
    uint32_t *seq_num;
    cudaMalloc(&seq_num, sizeof(uint32_t));
    cudaMemset(seq_num, 0, sizeof(uint32_t));

    nvshmem_barrier_all();

    // Create signal tensors that hold the peer sequence numbers to wait on
    uint64_t *seq_num_signal = (uint64_t *)nvshmem_calloc(steps_inter_, sizeof(uint64_t));
    if (!seq_num_signal) {
        throw std::runtime_error("Failed to allocate signal memory");
    }

    init_signal_kernel<<<1, steps_inter_>>>(seq_num_signal, steps_inter_);
    cudaDeviceSynchronize();
    nvshmem_barrier_all();

    allocated_scratch_[id] = scratch;
    seq_nums_[id] = seq_num;
    seq_num_signals_[id] = seq_num_signal;
    stride_sizes_[id] = stride_size;
}

void RecursiveLL8Coll::deregister_tensor(uint64_t id) {
    // TODO: Implement
    if (allocated_scratch_.find(id) == allocated_scratch_.end()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    nvshmem_free(allocated_scratch_[id]);
    nvshmem_free(seq_num_signals_[id]);
    cudaFree(seq_nums_[id]);
    allocated_scratch_.erase(id);
    seq_nums_.erase(id);
    seq_num_signals_.erase(id);
    stride_sizes_.erase(id);
}

template <typename T>
void RecursiveLL8Coll::allreduce_preallocated_impl(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg) {
    size_t numel = tensor.numel();

    // Size of each tensor after intra-node reduce scatter
    size_t numel_reduced = numel / npes_node_;
    size_t chunk_elems_reduced = chunk_size_ / npes_node_;

    void *src_sym = tensor.data_ptr();
    Payload8B<T> *payload_scratch = (Payload8B<T>*)allocated_scratch_[id];
    size_t stride_size = stride_sizes_[id];
    uint32_t *seq_num = seq_nums_[id];
    uint64_t *seq_num_signal = seq_num_signals_[id];

    void *payload_args[] = {&payload_scratch, &payload_scratch, &numel_reduced,
                    &chunk_elems_reduced, &stride_size, &seq_num, &seq_num_signal};
    dim3 gridDim(num_blocks_), blockDim(threads_per_block_);

    // TODO: Make this architecture specific
    int pack_threads = 512;
    int pack_blocks = (numel_reduced + pack_threads - 1) / pack_threads;

    // Intra-node reduce scatter
    dispatch_intra_reducescatter<T>((T*)src_sym, (const T*)src_sym, numel_reduced, stream);

    // Bump sequence number and wait on peers to have completed the last sequence number
    bump_seq<<<1, 1, 0, stream>>>(seq_num, seq_num_signal);

    // Launch kernel to properly pack data into Payload8B structures
    pack_payloads<T>(payload_scratch, (T*)src_sym, numel_reduced, seq_num, stream);

    // Inter-node recursive all-reduce
    nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_payload<T>, gridDim, blockDim, payload_args, 0, stream);

    // Unpack the final payload results and all-gather back to source tensor
    Payload8B<T> *final_payload = (Payload8B<T>*)((uint8_t*)payload_scratch + (steps_inter_ * stride_size));
    unpack_payloads<T>((T*)src_sym, final_payload, numel_reduced, seq_num, seq_num_signal, stream);


    // Intra-node all-gather
    dispatch_inter_allgather<T>((T*)src_sym, (const T*)src_sym, numel_reduced, stream);
    return;
}

// Private helper method implementations
template <typename T>
void RecursiveLL8Coll::pack_payloads(Payload8B<T>* payloads, const T* data, size_t num_elements, uint32_t* seq_num, cudaStream_t stream) {
    const size_t num_cells = (num_elements + Payload8B<T>::N - 1) / Payload8B<T>::N;
    const int block = 512;
    const int grid  = (int)((num_cells + block - 1) / block);
    pack_payloads_kernel<<<grid, block, 0, stream>>>(payloads, data, num_elements, seq_num);
}

template <typename T>
void RecursiveLL8Coll::unpack_payloads(T* data, const Payload8B<T>* payloads, size_t num_elements, uint32_t* seq_num, uint64_t* seq_num_signal, cudaStream_t stream) {
    const size_t num_cells = (num_elements + Payload8B<T>::N - 1) / Payload8B<T>::N;
    const int block = 512;
    const int grid  = (int)((num_cells + block - 1) / block);
    unpack_payloads_kernel<<<grid, block, 0, stream>>>(data, payloads, num_elements, seq_num, seq_num_signal);
}


#define INSTANTIATE_LL8_COLL_TYPE(T) \
    template __global__ void pack_payloads_kernel<T>(Payload8B<T>* __restrict__ payloads, const T* __restrict__ data, size_t num_elements, uint32_t* seq_num); \
    template __global__ void unpack_payloads_kernel<T>(T* __restrict__ data, const Payload8B<T>* __restrict__ payloads, size_t num_elements, uint32_t* seq_num, uint64_t* signal); \
    template __global__ void recursive_allreduce_kernel_payload<T>(Payload8B<T> *dst, Payload8B<T> *src, size_t nreduce, size_t chunk_elems, size_t stride_size, uint32_t* seq_num, uint64_t* seq_num_signal); \
    template void RecursiveLL8Coll::allreduce_preallocated_impl<T>(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg); \
    template void RecursiveLL8Coll::pack_payloads<T>(Payload8B<T>* payloads, const T* data, size_t num_elements, uint32_t* seq_num, cudaStream_t stream); \
    template void RecursiveLL8Coll::unpack_payloads<T>(T* data, const Payload8B<T>* payloads, size_t num_elements, uint32_t* seq_num, uint64_t* signal, cudaStream_t stream);

// Generate instantiations for all supported types using the macro
INSTANTIATE_LL8_COLL_TYPE(float)
INSTANTIATE_LL8_COLL_TYPE(__nv_bfloat16)
INSTANTIATE_LL8_COLL_TYPE(__half)
INSTANTIATE_LL8_COLL_TYPE(int)