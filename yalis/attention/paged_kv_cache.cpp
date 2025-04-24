#include <torch/torch.h>
#include <torch/extension.h>
#include <deque>
#include <stdexcept>
#include <iostream>

// KVCacheManager manages the assignment of global pages (blocks) for each sequence
// in the batch. It maintains a block table (a tensor of shape [batch_size, max_num_blocks_per_seq])
// and a tensor tokens_assigned_ to track how many tokens have been pushed into the KV cache
// for each sequence. A FIFO queue of free pages is maintained via a std::deque.
class KVCacheManager {
public:
    // Constructor.
    //   batch_size: number of sequences in the batch.
    //   max_num_blocks_per_seq: maximum pages allowed per sequence.
    //   num_blocks: total pages available in the global KV cache.
    //   page_block_size: number of tokens (positions) that fit in one page.
    KVCacheManager(
        int64_t batch_size, 
        int64_t max_num_blocks_per_seq, 
        int64_t num_blocks, 
        int64_t page_block_size,
        bool verbose = false)
      : batch_size_(batch_size),
        max_num_blocks_per_seq_(max_num_blocks_per_seq),
        num_blocks_(num_blocks),
        page_block_size_(page_block_size),
        verbose_(verbose)
    {
        // Initialize the block table tensor of shape [batch_size, max_num_blocks_per_seq]
        // with all elements set to -1 (indicating "no page assigned"), as int32.
        block_table_ = torch::full({batch_size_, max_num_blocks_per_seq_}, -1, 
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        
        // Initialize tokens_assigned_ which tracks how many tokens are currently
        // in the KV cache for each sequence. Initially, every sequence has 0 tokens.
        tokens_assigned_ = torch::zeros({batch_size_}, torch::kInt64);
        
        // Initialize the FIFO queue of free pages with indices from 0 to num_blocks - 1.
        for (int32_t i = 0; i < static_cast<int32_t>(num_blocks_); i++) {
            free_pages_.push_back(i);
        }
    }

    // update_block_table:
    //   seq_lengths: A tensor of shape [batch_size] containing the number of new tokens 
    //                to be pushed into the KV cache for each sequence.
    // For each sequence, this method adds the incoming tokens to the current token count,
    // computes the new required page count, and if new pages are needed, assigns additional 
    // pages from the free_pages_ FIFO queue.
    //
    // Returns:
    //   The updated block table tensor (shape: [batch_size, max_num_blocks_per_seq]).
    torch::Tensor update_block_table(const torch::Tensor &seq_lengths) {
        // Ensure seq_lengths tensor has the correct batch size.
        if (seq_lengths.sizes()[0] != batch_size_) {
            throw std::runtime_error("seq_lengths tensor size does not match batch_size.");
        }

        for (int64_t seq = 0; seq < batch_size_; seq++) {
            // 'incoming_tokens' is the number of new tokens to add.
            int64_t incoming_tokens = seq_lengths[seq].item<int64_t>();
            // Retrieve the current token count for this sequence.
            int64_t current_tokens = tokens_assigned_[seq].item<int64_t>();
            // Compute the new total tokens in the KV cache for this sequence.
            int64_t new_total_tokens = current_tokens + incoming_tokens;
            // Compute required pages for the new total token count.
            int64_t new_page_count = (new_total_tokens + page_block_size_ - 1) / page_block_size_;
            // Compute the old page count based on current_tokens.
            int64_t old_page_count = (current_tokens + page_block_size_ - 1) / page_block_size_;

            // Only assign additional pages if new_page_count is greater than old_page_count.
            if (new_page_count > old_page_count) {
                if (new_page_count > max_num_blocks_per_seq_) {
                    throw std::runtime_error("Exceeded maximum number of blocks per sequence.");
                }
                // For each additional required page, assign a free page.
                for (int64_t block_idx = old_page_count; block_idx < new_page_count; block_idx++) {
                    if (free_pages_.empty()) {
                        throw std::runtime_error("No free pages available in the global KV cache.");
                    }
                    int32_t free_page = free_pages_.front();
                    free_pages_.pop_front();
                    if (verbose_) {
                        std::cout << "Seq " << seq << ": assigning free page " 
                                  << free_page << " at block index " 
                                  << block_idx << std::endl;
                    }
                    // Update the block_table_ for sequence 'seq' at column 'block_idx'.
                    block_table_.index_put_({seq, block_idx}, free_page);
                }
            }
            // Update tokens_assigned_ for this sequence to new_total_tokens.
            tokens_assigned_.index_put_({seq}, new_total_tokens);
        }
        return block_table_;
    }

    void force_update_tokens_assigned(const torch::Tensor &new_token_counts) {
        // Ensure new_token_counts tensor has the correct batch size.
        if (new_token_counts.sizes()[0] != batch_size_) {
            throw std::runtime_error("Input tensor size does not match batch_size.");
        }
        // Copy the input tensor into tokens_assigned_.
        tokens_assigned_.copy_(new_token_counts);
    }

    // get_pages_for_sequence:
    // Returns a tensor containing the pages assigned to the sequence with index seq_idx.
    // The valid page count is computed as ceil(tokens_assigned_[seq_idx] / page_block_size_).
    torch::Tensor get_pages_for_sequence(int64_t seq_idx) {
        int64_t token_count = tokens_assigned_[seq_idx].item<int64_t>();
        int64_t page_count = (token_count + page_block_size_ - 1) / page_block_size_;
        // First select the row for seq_idx, then narrow that row to the first page_count entries.
        auto pages = block_table_.select(0, seq_idx).narrow(0, 0, page_count);
        return pages;
    }
    
    // release_sequence_pages:
    // Releases (frees) all pages assigned to the sequence at index seq_idx.
    // The freed pages are pushed back into the free_pages_ FIFO queue.
    // The corresponding row in the block table is reset to -1, and tokens_assigned_ is reset to 0.
    void release_sequence_pages(int64_t seq_idx) {
        int64_t token_count = tokens_assigned_[seq_idx].item<int64_t>();
        int64_t page_count = (token_count + page_block_size_ - 1) / page_block_size_;
        // First select the row corresponding to seq_idx, then narrow to the first page_count elements.
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
    
    // reset:
    // Resets the block table and tokens_assigned_ and refills the free_pages_ FIFO queue.
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

private:
    int64_t batch_size_;
    int64_t max_num_blocks_per_seq_;
    int64_t num_blocks_;
    int64_t page_block_size_;
    bool verbose_;

    // Tensor holding the block table of shape [batch_size, max_num_blocks_per_seq] (using int32 type).
    torch::Tensor block_table_;
    // Tensor holding the number of tokens currently in the KV cache for each sequence (shape [batch_size], int64).
    torch::Tensor tokens_assigned_;

    // FIFO queue of free page indices (as int32).
    std::deque<int32_t> free_pages_;
};

// Expose the KVCacheManager as a custom class via PyBind11.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<KVCacheManager>(m, "KVCacheManager")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("update_block_table", &KVCacheManager::update_block_table)
        .def("force_update_tokens_assigned", &KVCacheManager::force_update_tokens_assigned)
        .def("get_pages_for_sequence", &KVCacheManager::get_pages_for_sequence)
        .def("release_sequence_pages", &KVCacheManager::release_sequence_pages)
        .def("reset", &KVCacheManager::reset)
        .def("block_table", &KVCacheManager::block_table);
}
