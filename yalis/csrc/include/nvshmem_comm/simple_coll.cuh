#include "nvshmem_comm/coll.h"
#include "nvshmem_comm/nvshmem_utils.h"
#define POLL_TIMEOUT 10

template <typename T>
__global__ extern void recursive_allreduce_kernel_t(T *dst, T *src, size_t nreduce, int steps, int steps_intra, int steps_inter,
                                             uint64_t *chunk_signal, size_t chunk_size_bytes);

class RecursiveSimpleColl: public CollBase<RecursiveSimpleColl> {
  public:
    using Base = CollBase<RecursiveSimpleColl>;


    RecursiveSimpleColl() = default;

    ~RecursiveSimpleColl() noexcept override { 
        cleanup(); 
    }

    void initialize(int num_blocks, int threads_per_block, size_t chunk_size);
    void cleanup();
    void register_tensor(uint64_t id, size_t size, torch::Dtype dt, torch::Device dev);
    void deregister_tensor(uint64_t id);

    void set_kernel_params(int num_blocks, int threads_per_block, size_t chunk_size);

    template <typename T>
    void allreduce_preallocated_impl(torch::Tensor& tensor, uint64_t id, cudaStream_t stream, const std::string& alg);

  private:
    template <typename T>
    void __forceinline__ dispatch_intra_reducescatter(T *dst, const T *src, size_t num_elements, cudaStream_t stream) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            nvshmemx_bfloat16_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, __half>) {
            nvshmemx_half_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, float>) {
            nvshmemx_float_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, int>) {
            nvshmemx_int_sum_reducescatter_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for intra-node reduce scatter");
        }
    }

    template <typename T>
    void __forceinline__ dispatch_intra_allgather(T *dst, const T *src, size_t num_elements, cudaStream_t stream) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            nvshmemx_bfloat16_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, __half>) {
            nvshmemx_half_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, float>) {
            nvshmemx_float_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else if constexpr (std::is_same_v<T, int>) {
            nvshmemx_int_fcollect_on_stream(NVSHMEMX_TEAM_NODE,
                                                         dst,
                                                         src,
                                                         num_elements, stream);
        } else {
            throw std::runtime_error("Unsupported dtype for inter-node allgather");
        }
    }

    int num_blocks_;
    int threads_per_block_;
    size_t chunk_size_;
    int steps_;
    int steps_intra_;
    int steps_inter_;

    std::unordered_map<uint64_t, void *> allocated_scratch_;
    uint64_t* chunk_signal_;
    size_t chunk_signal_size_;
};