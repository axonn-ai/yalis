#include <tuple>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// KVCacheManager manages the assignment of global pages (blocks) for each
// sequence in the batch. It maintains a block table (a tensor of shape
// [batch_size, max_num_blocks_per_seq]) and a tensor tokens_assigned_ to track
// how many tokens have been pushed into the KV cache for each sequence.
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
    tokens_assigned_ = torch::zeros({batch_size_}, torch::TensorOptions()
            .dtype(torch::kInt64).device(torch::kCUDA));

    // Initialize the FIFO queue of free pages with indices from 0 to num_blocks
    // - 1.
    this->free_pages = torch::arange(
      num_blocks_,
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)
    );
    // FIFO counter of free pages
    this->next_free = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)
    );
  }

  // accessors to expose to python for avoiding graph breaks
  torch::Tensor tokens_assigned_tensor() const { return tokens_assigned_; }
  torch::Tensor free_pages_tensor() const { return free_pages; }
  torch::Tensor next_page_tensor() const { return next_free; }
  torch::Tensor block_table() const { return block_table_; }

  

  void force_update_tokens_assigned(const torch::Tensor &new_token_counts) {
    // Ensure new_token_counts tensor has the correct batch size.
    if (new_token_counts.sizes()[0] != batch_size_) {
      throw std::runtime_error("Input tensor size does not match batch_size.");
    }
    // Copy the input tensor into tokens_assigned_.
    tokens_assigned_.copy_(new_token_counts);
  }

  // get_pages_for_sequence:
  // Returns a tensor containing the pages assigned to the sequence with index
  // seq_idx. The valid page count is computed as ceil(tokens_assigned_[seq_idx]
  // / page_block_size_).
  torch::Tensor get_pages_for_sequence(int64_t seq_idx) {
    int64_t token_count = tokens_assigned_[seq_idx].item<int64_t>();
    int64_t page_count =
        (token_count + page_block_size_ - 1) / page_block_size_;
    // First select the row for seq_idx, then narrow that row to the first
    // page_count entries.
    auto pages = block_table_.select(0, seq_idx).narrow(0, 0, page_count);
    return pages;
  }

  // release_sequence_pages:
  // Releases (frees) all pages assigned to the sequence at index seq_idx.
  // The freed pages are pushed back into the free_pages_ FIFO queue.
  // The corresponding row in the block table is reset to -1, and
  // tokens_assigned_ is reset to 0.
  /*
  void release_sequence_pages(int64_t seq_idx) {
    int64_t token_count = tokens_assigned_[seq_idx].item<int64_t>();
    int64_t page_count =
        (token_count + page_block_size_ - 1) / page_block_size_;
    // First select the row corresponding to seq_idx, then narrow to the first
    // page_count elements.
    auto row = block_table_.select(0, seq_idx).narrow(0, 0, page_count);
    auto row_accessor = row.accessor<int32_t, 1>();
    for (int64_t i = 0; i < page_count; i++) {
      if (row_accessor[i] != -1) {
        free_pages_.push_back(row_accessor[i]);
      }
    }
    // Reset the row for the sequence by filling it with -1.
    block_table_.select(0, seq_idx).fill_(-1);
    tokens_assigned_.index_put_({seq_idx}, 0);
  }
  */

  // reset:
  // Resets the block table and tokens_assigned_ and refills the free_pages
  // FIFO queue.
  void reset() {
    c10::cuda::CUDAGuard guard(block_table_.device());
    // same storage, just reinitialize contents
    block_table_.fill_(-1);         // [B, M] int32
    tokens_assigned_.zero_();       // [B]    int64
    next_free.zero_();              // [1]    int64
    // refill free_pages as [0, 1, ..., num_blocks_-1]
    auto opts = free_pages.options().dtype(torch::kInt32);
    free_pages.copy_(torch::arange(num_blocks_, opts));
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
  // std::deque<int32_t> free_pages_;
  torch::Tensor free_pages;
  torch::Tensor next_free;
};

static void force_update_tokens_assigned_impl(
  torch::Tensor tokens_assigned, const torch::Tensor &new_counts
) {
  TORCH_CHECK(tokens_assigned.size(0) == new_counts.size(0), "batch mismatch");
  tokens_assigned.copy_(new_counts);
}

static inline at::Tensor ceil_div_tensor(const at::Tensor& x, int64_t d) {
  return at::floor_divide(x + (d - 1), d);
}

// update_block_table_impl:
//   seq_lengths: A tensor of shape [batch_size] containing the number of new
//   tokens
//                to be pushed into the KV cache for each sequence.
// For each sequence, this method adds the incoming tokens to the current
// token count, computes the new required page count, and if new pages are
// needed, assigns additional pages from the free_pages FIFO queue.
static void update_block_table_impl(
  const at::Tensor &block_table,      // int32/int64, [B, M], contiguous
  const at::Tensor &tokens_assigned,  // int64, [B]
  const at::Tensor &next_page,        // int64, [1] or [B]
  const at::Tensor &free_pages,       // int32, [N_pages]
  const at::Tensor &seq_lengths,      // int32/int64, [B]
  int64_t page_block_size,
  int64_t max_blocks_per_seq
) {
  c10::cuda::CUDAGuard guard(block_table.device());

  TORCH_CHECK(block_table.dim() == 2, "block_table must be [B, M]");
  TORCH_CHECK(tokens_assigned.dim() == 1, "tokens_assigned must be [B]");
  TORCH_CHECK(seq_lengths.dim() == 1, "seq_lengths must be [B]");
  TORCH_CHECK(block_table.is_contiguous(), "block_table must be contiguous");
  TORCH_CHECK(tokens_assigned.scalar_type() == at::kLong, "tokens_assigned int64");
  TORCH_CHECK(block_table.scalar_type() == at::kInt || block_table.scalar_type()==at::kLong, "block_table must be int32 or int64");

  const int64_t B = block_table.size(0);
  const int64_t M = block_table.size(1);
  const int64_t N_pages = free_pages.size(0);

  auto dev = block_table.device();
  auto long_opts = at::TensorOptions().device(dev).dtype(at::kLong);
  auto table_dtype = block_table.scalar_type();

  const int64_t K = next_page.numel();
  TORCH_CHECK(K == 1 && next_page.numel() == 1, "next_page must have numel 1 or B; got ", K, " vs B=", B);
  auto next_per = next_page.view({1}).expand({B});

  // Page math (shape-stable)
  auto inc_tokens = seq_lengths.to(at::kLong);        // [B]
  auto old_tokens = tokens_assigned;                  // [B]
  auto new_tokens = old_tokens + inc_tokens;          // [B]
  auto old_pages  = ceil_div_tensor(old_tokens, page_block_size);  // [B]
  auto new_pages  = ceil_div_tensor(new_tokens, page_block_size);  // [B]
  auto room       = (at::full({B}, (long)M, long_opts) - old_pages).clamp_min(0);
  auto delta      = at::minimum((new_pages - old_pages).clamp_min(0), room);     // [B]

  // Base offsets into the FIFO window
  auto csum       = at::cumsum(delta, 0);             // [B]
  auto start_excl = csum - delta;                     // [B]
  auto base_per   = next_per + start_excl;            // [B]

  // Column mask over [0..M-1]
  auto cols   = at::arange(M, long_opts);               // [M]
  auto cols2d = cols.view({1, M}).expand({B, M});       // [B, M]
  auto old2d  = old_pages.view({B, 1}).expand({B, M});  // [B, M]
  auto del2d  = delta.view({B, 1}).expand({B, M});      // [B, M]
  auto mask   = (cols2d >= old2d) & (cols2d < (old2d + del2d));     // [B, M], bool
  auto mask_i = mask.to(at::kLong);                                 // [B, M]

  // Row-local 0 .. (delta-1) counters
  auto k_in_row = (at::cumsum(mask_i, 1) - 1) * mask_i;              // [B, M]

  // Compute page indices, clamp to stay in-bounds (avoid OOB during capture)
  auto base2d   = base_per.view({B,1}).expand({B,M});                // [B, M]
  auto page_idx = (base2d + k_in_row).clamp(0, (long) N_pages - 1)
                    .reshape({B * M}).to(at::kLong);                 // [B*M]

  // Gather and blend (no variable-length selects)
  auto gathered = free_pages.index_select(0, page_idx).to(table_dtype)
                                .view({B, M});                       // [B, M]
  auto blended  = at::where(mask, gathered, block_table);            // [B, M]
  block_table.copy_(blended);

  // Update counters in-place
  tokens_assigned.copy_(new_tokens);                                  // [B]
  // advance the single global head by total granted pages
  next_page.add_(delta.sum());
}

static void force_update_tokens_assigned_meta(
  torch::Tensor tokens_assigned,
  const torch::Tensor &token_counter
) {}

static void update_block_table_meta(
  const torch::Tensor &block_table,
  const torch::Tensor &tokens_assigned,
  const torch::Tensor &next_page,
  const torch::Tensor &free_pages,
  const torch::Tensor &seq_lengths,
  int64_t /*page_block_size*/,
  int64_t /*max_blocks_per_seq*/
) {
  return;
}

TORCH_LIBRARY(yalis, m) {
  m.def("force_update_tokens_assigned_(Tensor(a!) tokens_assigned, Tensor new_counts) -> ()");
  m.def(
    "update_block_table_(Tensor(a!) block_table, "
    "Tensor(b!) tokens_assigned, "
    "Tensor(c!) next_page, "
    "Tensor free_pages, "
    "Tensor seq_lengths, "
    "int page_block_size, "
    "int max_blocks_per_seq) "
    "-> ()"
  );
}

TORCH_LIBRARY_IMPL(yalis, CUDA /*CompositeImplicitAutograd*/, m) {
  m.impl("force_update_tokens_assigned_", force_update_tokens_assigned_impl);
  m.impl("update_block_table_", update_block_table_impl);
}

TORCH_LIBRARY_IMPL(yalis, Meta, m) {
  m.impl("force_update_tokens_assigned_", force_update_tokens_assigned_meta);
  m.impl("update_block_table_", update_block_table_meta);
}

// Expose the KVCacheManager as a custom class via PyBind11.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<KVCacheManager>(m, "KVCacheManager")
      .def(py::init<int64_t, int64_t, int64_t, int64_t>())
      .def("get_pages_for_sequence", &KVCacheManager::get_pages_for_sequence)
      .def("reset", &KVCacheManager::reset)
      .def("block_table", &KVCacheManager::block_table)
      .def("tokens_assigned_tensor", &KVCacheManager::tokens_assigned_tensor)
      .def("free_pages_tensor", &KVCacheManager::free_pages_tensor)
      .def("next_page_tensor", &KVCacheManager::next_page_tensor);
}
