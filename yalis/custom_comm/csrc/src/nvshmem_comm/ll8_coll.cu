// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "nvshmem_comm/ll8_coll.cuh"

__global__ void bump_seq(uint32_t* seq, uint64_t* signal, int steps_inter) {
  const uint32_t seq_local = *seq;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
      (*seq)++;
  }

  // Make sure partners are ready
  for (int step = 0; step < steps_inter; ++step) {
    nvshmem_signal_wait_until(signal + step, NVSHMEM_CMP_EQ, seq_local + 1);
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
                                       uint64_t* seq_num_signal, 
                                       int steps, 
                                       int steps_intra,
                                       int steps_inter)
{

    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();

    uint64_t local_seq_num = *seq_num;

    // This is okay here because we are copying from the source buffer and so the recv buffer can be overwritten
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int partner;
      for (int step = steps_intra; step < steps; ++step) {
          int inter_node_step = step - steps_intra;
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
__device__ void __forceinline__ process_chunk_warp_all_ready(int lane, int warp, int warps, 
                                            Payload8B<T>* __restrict__ acc_chunk,
                                            const Payload8B<T>* __restrict__ dst_chunk,
                                            const Payload8B<T>* __restrict__ src_chunk,
                                            size_t chunk_size,
                                            uint32_t seq_num_local,
                                            unsigned long long timeout_ns)
{
  for (size_t tile = warp * 32; tile < chunk_size; tile += warps * 32) {
    const size_t idx   = tile + lane;
    const bool   valid = idx < chunk_size;

    // Mask of lanes that actually own a cell in this tile
    unsigned full = __activemask();
    unsigned mask = __ballot_sync(full, valid);

    Payload8B<T>* acc_cell = valid ? (acc_chunk + idx) : nullptr;
    const Payload8B<T>* dst_cell = valid ? (dst_chunk + idx) : nullptr;
    const Payload8B<T>* src_cell = valid ? (src_chunk  + idx) : nullptr;
    const volatile uint32_t* flag_ptr = valid ? &dst_cell->flag : (const volatile uint32_t*)NULL;

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
        // my_cell = src_cell + dst_cell
        acc_cell->add_update(*dst_cell, *src_cell);
    }
  }
}


// Recursive-doubling all-reduce using NVSHMEM with 16B payloads and flags.
// Templated over data type T.
template <typename T>
__global__ void recursive_allreduce_kernel_payload(Payload8B<T> *dst, Payload8B<T> *src, size_t nreduce,
                                                   int steps, int steps_intra, int steps_inter,
                                                   size_t chunk_elems, size_t stride_size, uint32_t* seq_num, 
                                                   uint64_t* seq_num_signal) 
  {
    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();

    constexpr size_t ELEMS_PER_PAYLOAD = Payload8B<T>::N;

    // Block‐wise partitioning
    const int block_idx    = blockIdx.x;
    const int num_blocks   = gridDim.x;

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;


    const size_t total_cells = (nreduce + ELEMS_PER_PAYLOAD - 1) / ELEMS_PER_PAYLOAD;
    const size_t cells_per_block = total_cells / num_blocks;
    if (cells_per_block * (block_idx) >= total_cells) return;

    // Slide pointers to this block’s slice
    size_t offset = block_idx * cells_per_block;

    // how many cells this block actually owns (handle tail)
    const size_t cells_this_block = min(cells_per_block, total_cells - offset);


    // Chunking inside each step
    size_t chunk_cells = max((size_t)32, (chunk_elems + ELEMS_PER_PAYLOAD - 1) / ELEMS_PER_PAYLOAD);
    size_t num_chunks  = (cells_this_block + chunk_cells - 1) / chunk_cells;

    const uint32_t seq_num_local = *seq_num;

    int partner, inter_node_step;
    Payload8B<T>* src_chunk;
    Payload8B<T>* dst_chunk;
    Payload8B<T>* acc_chunk;
    // --- recursive-doubling reduce ---
    for (int step = steps_intra; step < steps ; ++step) {
        partner = mype ^ (1 << step);
        inter_node_step = step - steps_intra;

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
          const size_t chunk_size = min(chunk_cells, cells_this_block - chunk_idx * chunk_cells);
          src_chunk = src + offset + chunk_idx * chunk_cells;
          dst_chunk = reinterpret_cast<Payload8B<T>*>(
                  reinterpret_cast<uint8_t*>(dst) + (inter_node_step + 1) * stride_size
              ) + offset + chunk_idx * chunk_cells;

          // Accumulator for this step is the next src chunk
          acc_chunk= reinterpret_cast<Payload8B<T>*>(
                  reinterpret_cast<uint8_t*>(src) + stride_size
              ) + offset + chunk_idx * chunk_cells;
      
          nvshmemx_put64_nbi_block(
              (void*)dst_chunk,
              (const void*)src_chunk,
              chunk_size,
              partner);

          process_chunk_warp_all_ready(lane, warp, warps, acc_chunk, dst_chunk, src_chunk, chunk_size, seq_num_local, POLL_TIMEOUT * 1000000000ULL);
      
          __syncwarp();
        }

        // The source pointer should be updated to the next step to the current latest buffer
        src = (Payload8B<T>*)((uint8_t*)src + stride_size);

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
    for (auto [id, ptr] : allocated_scratch_send_) {
        nvshmem_free(ptr);
    }
    for (auto [id, ptr] : allocated_scratch_recv_) {
        nvshmem_free(ptr);
    }
    for (auto [id, ptr] : seq_nums_) {
        cudaFree(ptr);
    }
    for (auto [id, ptr] : seq_num_signals_) {
        nvshmem_free(ptr);
    }
    allocated_scratch_send_.clear();
    allocated_scratch_recv_.clear();
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
    // TODO: We need to allocate 2x the size of the scratch memory so that 
    // we can separate the source and destination scratch memory
    // This is important because if dest(p1) == src(p2),
    // then we need extra synchronization to ensure that partner p1 does not 
    // overwrite src(p2) untill p2 has completed the last sequence number
    // This is only possible with a quiet barrier or by separating the source 
    // and destination scratch memory
    void *send_scratch = nvshmem_malloc((steps_inter_ + 1) * stride_size);
    if (!send_scratch) {
        throw std::runtime_error("Failed to allocate scratch memory");
    }
    // Important to zero out the scratch memory
    cudaMemset(send_scratch, 0, (steps_inter_ + 1) * stride_size);
    
    void *recv_scratch = nvshmem_malloc((steps_inter_ + 1) * stride_size);
    if (!recv_scratch) {
        throw std::runtime_error("Failed to allocate scratch memory");
    }
    // Important to zero out the scratch memory
    cudaMemset(recv_scratch, 0, (steps_inter_ + 1) * stride_size);

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

    allocated_scratch_send_[id] = send_scratch;
    allocated_scratch_recv_[id] = recv_scratch;
    seq_nums_[id] = seq_num;
    seq_num_signals_[id] = seq_num_signal;
    stride_sizes_[id] = stride_size;
}

void RecursiveLL8Coll::deregister_tensor(uint64_t id) {
    // TODO: Implement
    if (allocated_scratch_send_.find(id) == allocated_scratch_send_.end()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    nvshmem_free(allocated_scratch_send_[id]);
    nvshmem_free(allocated_scratch_recv_[id]);
    nvshmem_free(seq_num_signals_[id]);
    cudaFree(seq_nums_[id]);
    allocated_scratch_send_.erase(id);
    allocated_scratch_recv_.erase(id);
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
    Payload8B<T> *payload_scratch_send = (Payload8B<T>*)allocated_scratch_send_[id];
    Payload8B<T> *payload_scratch_recv = (Payload8B<T>*)allocated_scratch_recv_[id];
    size_t stride_size = stride_sizes_[id];
    uint32_t *seq_num = seq_nums_[id];
    uint64_t *seq_num_signal = seq_num_signals_[id];

    void *payload_args[] = {&payload_scratch_recv, &payload_scratch_send, &numel_reduced, &steps_, &steps_intra_, &steps_inter_,
                    &chunk_elems_reduced, &stride_size, &seq_num, &seq_num_signal};
    dim3 gridDim(num_blocks_), blockDim(threads_per_block_);


    // Intra-node reduce scatter
    dispatch_intra_reducescatter<T>((T*)src_sym, (const T*)src_sym, numel_reduced, stream);

    // Bump sequence number and wait on peers to have completed the last sequence number
    bump_seq<<<1, 1, 0, stream>>>(seq_num, seq_num_signal, steps_inter_);

    // Launch kernel to properly pack data into Payload8B structures
    pack_payloads<T>(payload_scratch_send, (T*)src_sym, numel_reduced, seq_num, stream);

    // Inter-node recursive all-reduce
    nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_payload<T>, gridDim, blockDim, payload_args, 0, stream);

    // Unpack the final payload results and all-gather back to source tensor
    // TODO: Unpacking and final reduction can be done in a single kernel avoiding the extra copy
    Payload8B<T> *final_payload = (Payload8B<T>*)((uint8_t*)payload_scratch_send + (steps_inter_ * stride_size));
    unpack_payloads<T>((T*)src_sym, final_payload, numel_reduced, seq_num, seq_num_signal, steps_, steps_intra_, steps_inter_, stream);

    // Intra-node all-gather
    dispatch_inter_allgather<T>((T*)src_sym, (const T*)src_sym, numel_reduced, stream);
    return;
}

// Private helper method implementations
template <typename T>
void RecursiveLL8Coll::pack_payloads(Payload8B<T>* payloads, const T* data, size_t num_elements, uint32_t* seq_num, cudaStream_t stream) {
    const size_t num_cells = (num_elements + Payload8B<T>::N - 1) / Payload8B<T>::N;
    // TODO: Make this architecture specific
    const int block = 1024;
    const int grid  = (int)((num_cells + block - 1) / block);
    pack_payloads_kernel<<<grid, block, 0, stream>>>(payloads, data, num_elements, seq_num);
}

template <typename T>
void RecursiveLL8Coll::unpack_payloads(T* data, const Payload8B<T>* payloads, size_t num_elements, uint32_t* seq_num, uint64_t* seq_num_signal, int steps, int steps_intra, int steps_inter, cudaStream_t stream) {
    const size_t num_cells = (num_elements + Payload8B<T>::N - 1) / Payload8B<T>::N;
    const int block = 1024;
    const int grid  = (int)((num_cells + block - 1) / block);
    unpack_payloads_kernel<<<grid, block, 0, stream>>>(data, payloads, num_elements, seq_num, seq_num_signal, steps, steps_intra, steps_inter);
}


#define INSTANTIATE_LL8_COLL_TYPE(T) \
    template __global__ void pack_payloads_kernel<T>(Payload8B<T>* __restrict__ payloads, const T* __restrict__ data, size_t num_elements, uint32_t* seq_num); \
    template __global__ void unpack_payloads_kernel<T>(T* __restrict__ data, const Payload8B<T>* __restrict__ payloads, size_t num_elements, uint32_t* seq_num, uint64_t* signal, int steps, int steps_intra, int steps_inter); \
    template __global__ void recursive_allreduce_kernel_payload<T>(Payload8B<T> *dst, Payload8B<T> *src, size_t nreduce, int steps, int steps_intra, int steps_inter, size_t chunk_elems, size_t stride_size, uint32_t* seq_num, uint64_t* seq_num_signal); \
    template void RecursiveLL8Coll::allreduce_preallocated_impl<T>(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg); \
    template void RecursiveLL8Coll::pack_payloads<T>(Payload8B<T>* payloads, const T* data, size_t num_elements, uint32_t* seq_num, cudaStream_t stream); \
    template void RecursiveLL8Coll::unpack_payloads<T>(T* data, const Payload8B<T>* payloads, size_t num_elements, uint32_t* seq_num, uint64_t* signal, int steps, int steps_intra, int steps_inter, cudaStream_t stream);

// Generate instantiations for all supported types using the macro
INSTANTIATE_LL8_COLL_TYPE(float)
INSTANTIATE_LL8_COLL_TYPE(__nv_bfloat16)
INSTANTIATE_LL8_COLL_TYPE(__half)
INSTANTIATE_LL8_COLL_TYPE(int)


