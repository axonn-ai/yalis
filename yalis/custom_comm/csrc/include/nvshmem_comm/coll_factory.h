// Copyright 2025 Parallel Software and Systems Group, University of Maryland.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include "nvshmem_comm/coll.h"
#include "nvshmem_comm/ll8_coll.cuh"
#include "nvshmem_comm/simple_coll.cuh"

class CollFactory {
  public:
    static IColl* create_coll(Protocol protocol) {
        switch (protocol) {
            case Protocol::LL8:
                return new RecursiveLL8Coll();
            case Protocol::SIMPLE:
                return new RecursiveSimpleColl();
            default:
                throw std::runtime_error("Unsupported protocol type");
        }
    }
};