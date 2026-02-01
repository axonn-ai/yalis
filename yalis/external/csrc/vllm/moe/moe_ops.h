/*  SPDX-License-Identifier: Apache-2.0
    SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    This file is adapted from
       https://github.com/vllm-project/vllm/blob/main/csrc/moe/moe_ops.h
*/
#pragma once

#include <torch/all.h>


void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
  int64_t block_size, torch::Tensor sorted_token_ids,
  torch::Tensor experts_ids,
  torch::Tensor num_tokens_post_pad);