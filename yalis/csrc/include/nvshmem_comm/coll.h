#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <nvshmem.h>
#include <nvshmemx.h>

enum class Protocol : uint8_t {
  SIMPLE,
  LL8,
};

// Abstract base class for protocol-specific collective operations
template <class Derived>
class CollBase {
  public:

    virtual ~CollBase() noexcept {
      for (auto [id, ptr] : allocated_tensors_) {
        nvshmem_free(ptr);
      }
      allocated_tensors_.clear();
    }

    template <typename... Args>
    void init(Args&&... args) {
      // Check if nvshmem is initialized
      // TODO: Currently MPG is not supported
      if (nvshmemx_init_status() != NVSHMEM_STATUS_IS_INITIALIZED) {
        throw std::runtime_error("NVSHMEM is not initialized");
      }
      mype_ = nvshmem_my_pe();
      npes_ = nvshmem_n_pes();
      mype_node_ = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
      npes_node_ = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);

      derived()->initialize(std::forward<Args>(args)...);
    }


    std::tuple<torch::Tensor, uint64_t> allocate_tensor(size_t size, torch::Dtype dt, torch::Device dev) {
        void *ptr = nvshmem_malloc(size * torch::elementSize(dt));

        if (!ptr) {
            throw std::runtime_error("Failed to allocate tensor memory");
        }

        uint64_t id = next_id_.fetch_add(1);
        allocated_tensors_[id] = ptr;

        // Register the tensor with the derived class which can maintain its own scratch memory
        derived()->register_tensor(id, size, dt, dev);

        auto tensor = torch::from_blob(ptr, {static_cast<long>(size)}, torch::dtype(dt).device(dev));
        return std::make_tuple(tensor, id);
    }

    void free_tensor(uint64_t id) {
        if (allocated_tensors_.find(id) == allocated_tensors_.end()) {
            throw std::runtime_error("Invalid tensor ID");
        }
        nvshmem_free(allocated_tensors_[id]);
        derived()->deregister_tensor(id);
    }

    template <typename T>
    void allreduce_preallocated(torch::Tensor& t, uint64_t id,
                                cudaStream_t s, const std::string& alg) {
        derived()->template allreduce_preallocated_impl<T>(t, id, s, alg);
    }

    template <typename... Args>
    void set_kernel_params(Args&&... args) {
      derived()->set_kernel_params(std::forward<Args>(args)...);
    }

private:
    Derived* derived() { return static_cast<Derived*>(this); }

protected:
    // Memory Pools for allocated tensors
    std::unordered_map<uint64_t, void *> allocated_tensors_;

    // Next ID for allocated tensors
    std::atomic<uint64_t> next_id_;

    // PE information
    int mype_;
    int npes_;
    int mype_node_;
    int npes_node_;
};
