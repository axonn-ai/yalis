# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch
import torch.nn.functional as F

from torch.autograd import Function

import math

from axonn import axonn as ax
from axonn.intra_layer.communication import (
    Drop,
    Gather,
)
from yalis.external.nccl_comm import CommHandler
from yalis.tensor_parallel.all_reduce_op import tp_all_reduce

from typing import Optional, Sequence


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, in_features_group
):
    params = Drop.apply(params, in_features_group).contiguous()
    return params


@torch.no_grad()
def initialize_params(
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty((out_features, in_features), device=init_device)
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group
    )
    return params



class TPRMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(
        self, 
        size: int, 
        dim: int = -1, 
        eps: float = 1e-6, 
        add_unit_offset: bool = False, 
        transpose=False,
        tensor_parallel_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        if tensor_parallel_dims is not None and torch.distributed.get_rank() == 0:
            print(
                "Manually setting TP dims for RMS Norm layer with shape",
                f" - {(size,)} | tp-dims = {tensor_parallel_dims}",
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        if transpose:
            self.inner_group, self.outer_group = self.outer_group, self.inner_group

        self.inner_nccl_comm_idx = CommHandler.create_communicator_from_process_group(self.inner_group)
        
        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        
        assert size % self.inner_group_size == 0
        local_size = size // self.inner_group_size
        self.weight = torch.nn.Parameter(torch.ones(local_size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset
        self.local_size = local_size 
        self.global_size = size

        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict
        
    def all_reduce(self, x):
        tp_all_reduce(x, self.inner_nccl_comm_idx)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        if self.dim == -1:
            norm_x = self.all_reduce(norm_x) / self.inner_group_size 
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def _is_full_weight(self, weight):
        return (
            weight.ndim == 1
            and weight.size(0) == self.global_size
        )

    def _is_sharded_weight(self, weight):
        return (
            weight.ndim == 1
            and weight.size(0) == self.local_size
        )

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = (
            state_dict[prefix + "weight"] if prefix + "weight" in state_dict else None
        )

        if weight is not None:
            is_full_weight = self._is_full_weight(weight)
            is_sharded_weight = self._is_sharded_weight(weight)

            assert (
                is_full_weight or is_sharded_weight
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight:
                weight = extract_local_params_from_full_params(
                    weight, self.inner_group
                )

            state_dict[prefix + "weight"] = weight


        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)





    
