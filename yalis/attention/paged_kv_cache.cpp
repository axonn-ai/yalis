#include <deque>
#include <vector>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>

// KVCacheManager manages the assignment of global pages (blocks) for each
// sequence in the batch. It maintains a block table (a tensor of shape
// [batch_size, max_num_blocks_per_seq]) and a tensor tokens_assigned_ to track
// how many tokens have been pushed into the KV cache for each sequence. A FIFO
// queue of free pages is maintained via a std::deque.
class KVCacheManager {
public:
  // Constructor.
  //   batch_size: number of sequences in the batch.
  //   max_num_blocks_per_seq: maximum pages allowed per sequence.
  //   num_blocks: total pages available in the global KV cache.
  //   page_block_size: number of tokens (positions) that fit in one page.
  KVCacheManager(int64_t batch_size, int64_t max_num_blocks_per_seq,
                 int64_t num_blocks, int64_t page_block_size,
                 bool verbose = false)
      : batch_size_(batch_size),
        max_num_blocks_per_seq_(max_num_blocks_per_seq),
        num_blocks_(num_blocks), page_block_size_(page_block_size),
        verbose_(verbose) {
    // Initialize the block table tensor of shape [batch_size,
    // max_num_blocks_per_seq] with all elements set to -1 (indicating "no page
    // assigned"), as int32.

    block_table_ = torch::full(
        {batch_size_, max_num_blocks_per_seq_}, -1,
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    // Initialize tokens_assigned_ which tracks how many tokens are currently
    // in the KV cache for each sequence. Initially, every sequence has 0
    // tokens.
    // TODO: This should be in pinned memory
    tokens_assigned_ = torch::zeros({batch_size_}, torch::TensorOptions().dtype(
      torch::kInt32).device(torch::kCUDA));

    // Initialize the FIFO queue of free pages with indices from 0 to num_blocks
    // - 1.
    for (int32_t i = 0; i < static_cast<int32_t>(num_blocks_); i++) {
      free_pages_.push_back(i);
    }
  }

  // reset:
  // Resets the block table and tokens_assigned_ and refills the free_pages_
  // FIFO queue.
  void reset() {
    block_table_.fill_(-1);
    tokens_assigned_.fill_(0);
    free_pages_.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(num_blocks_); i++) {
      free_pages_.push_back(i);
    }
  }

  // Accessor for the block table (for debugging or introspection).
  torch::Tensor block_table() const { return block_table_; }

  // Accessor for tokens_assigned (current per-row token counts).
  torch::Tensor tokens_assigned() const { return tokens_assigned_; }

  // allocate_sequence:
  // Reserve pages for a sequence row based on initial_tokens and populate the
  // corresponding row of block_table_. Returns the assigned page ids.
  std::vector<int32_t> allocate_sequence(int64_t row_id, int64_t initial_tokens) {
    if (row_id < 0 || row_id >= batch_size_) {
      throw std::runtime_error("allocate_sequence: row_id out of range");
    }
    if (initial_tokens < 0) {
      throw std::runtime_error("allocate_sequence: initial_tokens must be >= 0");
    }
    // If already has tokens, refuse (caller should use extend).
    int64_t current_tokens = tokens_assigned_[row_id].item<int64_t>();
    if (current_tokens != 0) {
      throw std::runtime_error("allocate_sequence: row already initialized; use extend_sequence");
    }
    int64_t page_count =
        (initial_tokens + page_block_size_ - 1) / page_block_size_;
    if (page_count > max_num_blocks_per_seq_) {
      throw std::runtime_error("allocate_sequence: exceeds max_num_blocks_per_seq");
    }
    if (static_cast<int64_t>(free_pages_.size()) < page_count) {
      throw std::runtime_error("allocate_sequence: insufficient free pages");
    }
    std::vector<int32_t> assigned;
    assigned.reserve(static_cast<size_t>(page_count));
    for (int64_t i = 0; i < page_count; ++i) {
      int32_t p = free_pages_.front();
      free_pages_.pop_front();
      block_table_.index_put_({row_id, i}, p);
      assigned.push_back(p);
    }
    tokens_assigned_.index_put_({row_id}, initial_tokens);
    return assigned;
  }

  // extend_sequence:
  // Add n_new_tokens for a sequence row; assigns additional pages if crossing
  // page boundaries. Returns only the newly assigned page ids (if any).
  std::vector<int32_t> extend_sequence(int64_t row_id, int64_t n_new_tokens) {
    if (row_id < 0 || row_id >= batch_size_) {
      throw std::runtime_error("extend_sequence: row_id out of range");
    }
    if (n_new_tokens <= 0) {
      return {};  // nothing to do
    }
    int64_t current_tokens = tokens_assigned_[row_id].item<int32_t>();
    int64_t new_total_tokens = current_tokens + n_new_tokens;
    int64_t old_pages =
        (current_tokens + page_block_size_ - 1) / page_block_size_;
    int64_t new_pages =
        (new_total_tokens + page_block_size_ - 1) / page_block_size_;
    if (new_pages > max_num_blocks_per_seq_) {
      throw std::runtime_error("extend_sequence: exceeds max_num_blocks_per_seq");
    }
    int64_t delta = new_pages - old_pages;
    std::vector<int32_t> newly_assigned;
    if (delta > 0) {
      if (static_cast<int64_t>(free_pages_.size()) < delta) {
        throw std::runtime_error("extend_sequence: insufficient free pages");
      }
      newly_assigned.reserve(static_cast<size_t>(delta));
      for (int64_t i = old_pages; i < new_pages; ++i) {
        int32_t p = free_pages_.front();
        free_pages_.pop_front();
        block_table_.index_put_({row_id, i}, p);
        newly_assigned.push_back(p);
      }
    }
    tokens_assigned_.index_put_({row_id}, new_total_tokens);
    return newly_assigned;
  }

  // free_sequence:
  // Frees all pages assigned to row_id and clears row state. Returns the freed
  // page ids (in FIFO order of the row, not the global queue).
  std::vector<int32_t> free_sequence(int64_t row_id) {
    if (row_id < 0 || row_id >= batch_size_) {
      throw std::runtime_error("free_sequence: row_id out of range");
    }
    int64_t token_count = tokens_assigned_[row_id].item<int32_t>();
    if (token_count == 0) {
      return {};
    }
    int64_t page_count =
        (token_count + page_block_size_ - 1) / page_block_size_;
    std::vector<int32_t> freed;
    freed.reserve(static_cast<size_t>(page_count));
    auto row = block_table_.select(0, row_id).narrow(0, 0, page_count).to(torch::kCPU);
    auto acc = row.accessor<int32_t, 1>();
    for (int64_t i = 0; i < page_count; ++i) {
      int32_t page = acc[i];
      if (page != -1) {
        free_pages_.push_back(page);
        freed.push_back(page);
      }
    }
    block_table_.select(0, row_id).fill_(-1);
    tokens_assigned_.index_put_({row_id}, 0);
    return freed;
  }

private:
  int64_t batch_size_;
  int64_t max_num_blocks_per_seq_;
  int64_t num_blocks_;
  int64_t page_block_size_;
  bool verbose_;

  // Tensor holding the block table of shape [batch_size,
  // max_num_blocks_per_seq] (using int32 type).
  torch::Tensor block_table_;
  // Tensor holding the number of tokens currently in the KV cache for each
  // sequence (shape [batch_size], int64).
  torch::Tensor tokens_assigned_;

  // FIFO queue of free page indices (as int32).
  std::deque<int32_t> free_pages_;
};

// Expose the KVCacheManager as a custom class via PyBind11.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<KVCacheManager>(m, "KVCacheManager")
      .def(py::init<int64_t, int64_t, int64_t, int64_t>())
      .def("allocate_sequence", &KVCacheManager::allocate_sequence)
      .def("extend_sequence", &KVCacheManager::extend_sequence)
      .def("free_sequence", &KVCacheManager::free_sequence)
      .def("reset", &KVCacheManager::reset)
      .def("block_table", &KVCacheManager::block_table)
      .def("tokens_assigned", &KVCacheManager::tokens_assigned);
}
