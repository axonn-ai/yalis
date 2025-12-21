#include "ops.h"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Register main vllm_ops library
TORCH_LIBRARY(vllm_ops, m) {
  m.def("silu_and_mul(Tensor! out, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
  
  // MoE Ops
  m.def(
    "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
    "token_expert_indices, Tensor gating_output) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);
  
  // MOE Align block size
  m.def(
    "moe_align_block_size(Tensor topk_ids, int num_experts, "
    "int block_size, Tensor! sorted_token_ids, "
    "Tensor! experts_ids, "
    "Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // no Python-level functions; importing is enough to trigger registration
}

