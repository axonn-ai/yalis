#include "nvshmem_comm/simple_coll.cuh"

// Templated over data type T.
template <typename T>
__global__ void recursive_allreduce_kernel_t(T *dst, T *src, size_t nreduce, int steps, int steps_intra, int steps_inter,
                                             uint64_t *chunk_signal, size_t chunk_size_bytes) {
    int mype  = nvshmem_my_pe();
    int npes  = nvshmem_n_pes();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

    // Block‐wise partitioning (same as before)
    int  block_idx    = blockIdx.x;
    int  thread_id    = threadIdx.x;
    int  num_threads  = blockDim.x;
    int  num_blocks   = gridDim.x;
    const size_t elems_per_block   = nreduce / num_blocks;
    if (elems_per_block * (block_idx) >= nreduce) return;


    // Slide dst/src pointers to this block’s slice
    dst = dst + block_idx * elems_per_block;
    src = src + block_idx * elems_per_block;                // scratch lives here
    chunk_signal = chunk_signal + block_idx * steps_inter;  // one signal word per block

    // Chunking inside each step
    size_t chunk_elems = max((size_t)32, chunk_size_bytes / sizeof(T));
    size_t num_chunks  = (elems_per_block + chunk_elems - 1) / chunk_elems;

    int partner;
    int inter_node_step;
    // --- recursive-doubling reduce ---
    for (int step = steps_intra; step < steps ; ++step) {
        inter_node_step = step - steps_intra;
        partner = mype ^ (1 << step);

        // exchange & reduce each chunk
        for (size_t chunk_idx = 0; chunk_idx < num_chunks ; chunk_idx++) {
            size_t chunk_size = chunk_elems;
            T *my_chunk = src + chunk_idx * chunk_elems;
            T *scratch_chunk = (T*)dst + inter_node_step * nreduce + chunk_idx * chunk_elems;

            nvshmemx_putmem_signal_nbi_block(
                (void*)scratch_chunk,
                (const void*)my_chunk,
                chunk_size * sizeof(T),
                chunk_signal + inter_node_step,
                1,
                NVSHMEM_SIGNAL_ADD,
                partner);

            // wait for partner’s data to arrive
            if (thread_id == 0) {
                // The +1 is to account for the fact that the signal starts at 1, not 0
                nvshmem_signal_wait_until(chunk_signal + inter_node_step, NVSHMEM_CMP_GE, chunk_idx + 1);
                //printf("[%d, %d] step: %d, chunk_idx: %lu, signal: %lu\n", mype, block_idx, step, chunk_idx, *(signal + step));
            }
            __syncthreads();

            // accumulate: dst += scratch
            for (size_t i = thread_id; i < chunk_size; i += num_threads) {
                scratch_chunk[i] = add_op(my_chunk[i], scratch_chunk[i]);
            }
        }
        __syncthreads();

        // The source pointer should be updated to the next step to the current latest buffer
        src = (T*)dst + inter_node_step * nreduce;

      	//nvshmem_fence();
        if (thread_id == 0) {
             chunk_signal[inter_node_step] = 0;
             __threadfence_system();
        }
    }
}

// Class Method implementations
void RecursiveSimpleColl::initialize(int num_blocks, int threads_per_block, size_t chunk_size) {
    steps_ = 0;
    while ((1u << steps_) < (unsigned)npes_) ++steps_;
    steps_intra_ = 0;
    while ((1u << steps_intra_) < (unsigned)npes_node_) ++steps_intra_;
    steps_inter_ = steps_ - steps_intra_;

    set_kernel_params(num_blocks, threads_per_block, chunk_size);
}

void RecursiveSimpleColl::cleanup() {
    for (auto [id, ptr] : allocated_scratch_) {
        nvshmem_free(ptr);
    }
    if (chunk_signal_) {
        nvshmem_free(chunk_signal_);
    }

    allocated_scratch_.clear();
}

void RecursiveSimpleColl::register_tensor(uint64_t id, size_t size, torch::Dtype dt, torch::Device dev) {
    // Create scratch memory for inter-node communication
    // TODO: Can reduce the size of the scratch memory by npes_node_
    size_t size_inter = size / npes_node_;
    void *scratch = nvshmem_malloc(steps_inter_ * size_inter * torch::elementSize(dt));
    if (!scratch) {
        throw std::runtime_error("Failed to allocate scratch memory");
    }
    // Important to zero out the scratch memory
    cudaMemset(scratch, 0, steps_inter_ * size_inter * torch::elementSize(dt));
    allocated_scratch_[id] = scratch;
}

void RecursiveSimpleColl::deregister_tensor(uint64_t id) {
    if (allocated_scratch_.find(id) == allocated_scratch_.end()) {
        throw std::runtime_error("Invalid tensor ID");
    }
    nvshmem_free(allocated_scratch_[id]);
    allocated_scratch_.erase(id);
}

void RecursiveSimpleColl::set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size) {
    num_blocks_ = num_blocks;
    threads_per_block_ = threads_per_block;
    chunk_size_ = chunk_size;

    if (steps_ == 0) {
       throw std::runtime_error("Steps is 0");
    }

    if (chunk_signal_) {
        nvshmem_free(chunk_signal_);
    }

    chunk_signal_size_ = num_blocks_ * steps_inter_;
    chunk_signal_ = (uint64_t*)nvshmem_calloc(chunk_signal_size_, sizeof(uint64_t));
    if (!chunk_signal_) {
        throw std::runtime_error("Failed to allocate chunk signal memory");
    }
}

template <typename T>
void RecursiveSimpleColl::allreduce_preallocated_impl(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg) {
    size_t numel = tensor.numel();
    size_t numel_reduced = numel / npes_node_;

    size_t size_bytes = numel * tensor.element_size();
    size_t size_bytes_reduced = size_bytes / npes_node_;

    size_t chunk_size_reduced = chunk_size_ / npes_node_;

    void *src_sym = tensor.data_ptr();
    void *dst_sym = allocated_scratch_[id];

    dim3 gridDim(num_blocks_), blockDim(threads_per_block_);
    void *args[] = {&dst_sym, &src_sym, &numel_reduced, &steps_, &steps_intra_, &steps_inter_, &chunk_signal_, &chunk_size_reduced};

    dispatch_intra_reducescatter((T*)src_sym, (const T*)src_sym, numel_reduced, stream);
    nvshmemx_collective_launch((const void *)recursive_allreduce_kernel_t<T>, gridDim, blockDim, args, 0, stream);
    dispatch_intra_allgather((T*)src_sym, (const T*)dst_sym + (steps_inter_ - 1) * numel_reduced, numel_reduced, stream);

    // TODO: Remove this by employing sequence numbers
    //nvshmemx_quiet_on_stream(stream);

    return;
}

#define INSTANTIATE_SIMPLE_COLL_TYPE(T) \
    template void RecursiveSimpleColl::allreduce_preallocated_impl<T>(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg); \

// Generate instantiations for all supported types using the macro
INSTANTIATE_SIMPLE_COLL_TYPE(float)
INSTANTIATE_SIMPLE_COLL_TYPE(__nv_bfloat16)
INSTANTIATE_SIMPLE_COLL_TYPE(__half)
INSTANTIATE_SIMPLE_COLL_TYPE(int)


