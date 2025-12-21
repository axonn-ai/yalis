/*  SPDX-License-Identifier: Apache-2.0
    SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    This file is adapted from
       https://github.com/vllm-project/vllm/blob/main/csrc/ops.h
*/
#pragma once

#include <torch/all.h>
#include <moe/moe_ops.h>


void silu_and_mul(torch::Tensor& out, torch::Tensor& input);