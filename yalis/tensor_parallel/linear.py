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

from typing import Optional, Sequence
import gc

from yalis.communication.communication import yalis_drop, yalis_all_gather, compute_offset, can_divide


# Wrapper for custom_fwd to handle different versions of PyTorch
def version_aware_custom_fwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_fwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_fwd(*args, **kwargs)


# Wrapper for custom_bwd to handle different versions of PyTorch
def version_aware_custom_bwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_bwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_bwd(*args, **kwargs)


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, out_features_group, in_features_group, asym_split=None, transpose=False
):
    #params = Drop.apply(params, in_features_group)
    params = yalis_drop(params, -1, in_features_group, asym_split)
    params = yalis_drop(torch.t(params).contiguous(), -1, out_features_group, asym_split)
    #params = Drop.apply(torch.t(params).contiguous(), out_features_group)
    params = torch.t(params).contiguous()
    return params


@torch.no_grad()
def initialize_params(
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    init_method,
    init_device="cuda",
    asym_split=None,
    transpose=False
):
    params = torch.empty((out_features, in_features), device=init_device)
    init_method(params)

    # params = extract_local_params_from_full_params(
    #     params, out_features_group, in_features_group, asym_split=asym_split, transpose=transpose
    # )
    rank = dist.get_rank()
    world_size = dist.get_world_size() # Assuming default group for rank/world_size determination

    if transpose:
        # print_rank0(f"Initializing RowParallel weight: ({out_features}, {in_features}) with asym_split={asym_split}")
        shard_group = in_features_group
        shard_dim_size = in_features
        shard_torch_dim = 1
        if asym_split:
            group_rank = dist.get_rank(shard_group)
            #check, num, den = can_divide(shard_dim_size, asym_split[group_rank])
            check = can_divide(shard_dim_size, asym_split)
            if not check:
                raise ValueError(f"Cannot divide in_features {shard_dim_size} for rank {group_rank} with split {asym_split[group_rank]}")
            #local_shard_size = (shard_dim_size * num) // den
            local_shard_size = round(shard_dim_size * asym_split[group_rank])
            # print(f"[Rank {rank}] RowParallel: Sharding dim {shard_torch_dim} ({shard_dim_size}) using group {shard_group}. My rank {group_rank}, target size {local_shard_size}, split {asym_split[group_rank]}")
        else:
            pass
        params = yalis_drop(params, shard_torch_dim, process_group=shard_group, asymmetric=asym_split)

    else:
        # print_rank0(f"Initializing ColumnParallel weight: ({out_features}, {in_features}) with asym_split={asym_split}")
        shard_group = out_features_group
        shard_dim_size = out_features
        shard_torch_dim = 0
        if asym_split:
            group_rank = dist.get_rank(shard_group)
            #check, num, den = can_divide(shard_dim_size, asym_split[group_rank])
            check = can_divide(shard_dim_size, asym_split)
            if not check:
                raise ValueError(f"Cannot divide out_features {shard_dim_size} for rank {group_rank} with split {asym_split[group_rank]}")
            #local_shard_size = (shard_dim_size * num) // den
            local_shard_size = round(shard_dim_size * asym_split[group_rank])
            # print(f"[Rank {rank}] ColParallel: Sharding dim {shard_torch_dim} ({shard_dim_size}) using group {shard_group}. My rank {group_rank}, target size {local_shard_size}, split {asym_split[group_rank]}")
        else:
            pass
        # Drop rows
        params = yalis_drop(params, shard_torch_dim, process_group=shard_group, asymmetric=asym_split)

    # This line is important within this function as placing outside leads to pytorch not deleting the reserved memory
    torch.cuda.empty_cache()

    # This will lead to immediate memory garbage collection but can be slow - Probably not needed
    gc.collect()
    return params


@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class TPLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *args,
        transpose=False,
        bias=True,
        skip_bias_add=False,
        init_method=None,
        expert_mode=True,
        tensor_parallel_dims: Optional[Sequence[int]] = None,
        init_device="cuda",
        asym_split = None,
        do_print = False,
        **kwargs,
    ):
        super(TPLinear, self).__init__()
        self.asym_split = asym_split
        assert expert_mode, "Only expert mode allowed in inference"

        self.init_device = init_device
        # weights are shaped [out_features, in_features]
        # in_features are distributed across self.inner_group (X tensor parallel group)
        # out_features are distributed across self.inner_group (Y tensor parallel group)
        # if transpose is true then X and Y are swapped
        self.transpose = transpose
        if tensor_parallel_dims is not None and torch.distributed.get_rank() == 0:
            print(
                "Manually setting TP dims for a layer with shape",
                f" - {(in_features, out_features)} | tp-dims = {tensor_parallel_dims}",
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        if transpose:
            self.inner_group, self.outer_group = self.outer_group, self.inner_group
        # else:
        #     print("NOT TRANSPOSE, inner_size: ", dist.get_world_size(self.inner_group), " outer_size: ", dist.get_world_size(self.outer_group))

        # depth_group is the Z tensor parallel group (akin to FSDP)
        # self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        assert self.depth_group_size == 1

        # these are the in and out features of the full global weight matrix
        self.in_features = in_features
        self.out_features = out_features

        # expert mode = True -> user needs to parallelize non-linear layers manually
        # expert mode = False -> non-linear layers are parallelized using
        #                        data parallelism
        #                        automatically by AxoNN. This does involve some
        #                        extra communication
        #                        at the beginning and end of each linear layer.
        self.expert_mode = expert_mode

        # init_method -> function to initialize the weight matrix
        if init_method is None:
            init_method = default_init_method

        # in_features should be divisible by inner_group_size
        #assert in_features % self.inner_group_size == 0
        # in_features should be divisible by inner_group_size
        #assert out_features % self.outer_group_size == 0
        # local_in_features - this is the number of in_features on each GPU
        # self.local_in_features = divide(in_features, self.inner_group_size)
        # local_out_features - this is the number of out_features on each GPU
        # self.local_out_features = divide(out_features, self.outer_group_size)
        
        rank = dist.get_rank()
        if self.transpose: # Row Parallel
            shard_group = self.inner_group
            full_dim_size = self.in_features
            if asym_split:
                group_rank = dist.get_rank(shard_group)
                #check, num, den = can_divide(full_dim_size, asym_split[group_rank])
                check = can_divide(full_dim_size, asym_split)
                if not check: raise ValueError("Asymmetric split error for in_features")
                #self.local_in_features = (full_dim_size * num) // den
                self.local_in_features = round(full_dim_size * asym_split[group_rank])
            else:
                if full_dim_size % self.inner_group_size != 0: raise ValueError("in_features not divisible by inner_group_size")
                self.local_in_features = full_dim_size // self.inner_group_size
            self.local_out_features = self.out_features # Not sharded
            # print(f"[Rank {rank}] TPLinear RowParallel Init: local_in={self.local_in_features}, local_out={self.local_out_features}")
        else: # Column Parallel
            shard_group = self.outer_group
            full_dim_size = self.out_features
            if asym_split:
                group_rank = dist.get_rank(shard_group)
                #check, num, den = can_divide(full_dim_size, asym_split[group_rank])
                check = can_divide(full_dim_size, asym_split)
                if not check: raise ValueError("Asymmetric split error for out_features")
                #self.local_out_features = (full_dim_size * num) // den
                self.local_out_features = round(full_dim_size * asym_split[group_rank])
            else:
                if full_dim_size % self.outer_group_size != 0: raise ValueError("out_features not divisible by outer_group_size")
                self.local_out_features = full_dim_size // self.outer_group_size
            self.local_in_features = self.in_features # Not sharded
            # print(f"[Rank {rank}] TPLinear ColParallel Init: local_in={self.local_in_features}, local_out={self.local_out_features}")


        # initialize the weight matrix and grab the local slice for each GPU
        initial_params = initialize_params(
            out_features,
            in_features,
            self.outer_group,
            self.inner_group,
            init_method,
            init_device=self.init_device,
            asym_split=self.asym_split,
            transpose=self.transpose
        )
        # register the weight matrix as a trainable parameter.
        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)
        #if do_print:
        #    print(self.weight.shape)

        # Note: weight shape is (local_out_features, local_in_features) after sharding
        if self.weight.size(0) != self.local_out_features or self.weight.size(1) != self.local_in_features:
             print(f"[Rank {rank}] WARNING: Initialized weight shape {self.weight.shape} mismatch! Expected ({self.local_out_features}, {self.local_in_features}). Transpose={self.transpose}, AsymSplit={self.asym_split}")
             # Override calculated features with actual shape from initialized weight
             self.local_out_features = self.weight.size(0)
             self.local_in_features  = self.weight.size(1)

        # extra book-keeping for the weight tensor.
        # this is needed by AxoNN layer in the sync_gradients and
        # gradient clipping functions.
        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_depth_parallel_gradient_sync", False)
        setattr(
            self.weight,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    self.local_out_features,
                    device=self.init_device,
                )
            )
            setattr(self.bias, "is_tensor_parallel", True)
            setattr(self.bias, "needs_depth_parallel_gradient_sync", True)
            if not transpose:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.outer_intra_layer_parallel_group,
                )
            else:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.inner_intra_layer_parallel_group,
                )
        else:
            self.bias = None

        self.skip_bias_add = skip_bias_add
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

    def all_reduce(self, x, do_print=False):
        #if do_print:
        #    print("WORLD SIZE INNER GROUP: ", dist.get_world_size(group=self.inner_group))
        dist.all_reduce(x, group=self.inner_group)
        return x

    def matmul(self, w, x):
        return F.linear(x, w)

    def forward(
        self,
        x,
        do_print=False
    ):
        #if self.asym_split != None:
        #    x = yalis_drop(x, -1, self.inner_group, self.asym_split) 
        # print(f"[Rank {dist.get_rank()}] Forward Matmul: Input Shape {x.shape}, Weight Shape {self.weight.shape}, Transpose {self.transpose}")
        #if do_print:
        #    print(x.shape)
        x = self.matmul(self.weight, x)
        # print(f"[Rank {dist.get_rank()}] After Matmul Shape: {x.shape}")
        x = self.all_reduce(x, do_print=do_print)
        # print(f"[Rank {dist.get_rank()}] After AllReduce Shape: {x.shape}")

        if self.bias is None:
            return x
        else:
            bias = self.bias
            if self.skip_bias_add:
                return x, bias
            else:
                return x + bias

    def _is_full_weight_matrix(self, weight):
        return (
            weight.ndim == 2
            and weight.size(0) == self.out_features
            and weight.size(1) == self.in_features
        )

    def _is_sharded_weight_matrix(self, weight):
        return (
            weight.ndim == 2
            and weight.size(0) == self.local_out_features
            and weight.size(1) == self.local_in_features
        )

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # If the parameters were initialized on meta-device, we need to materialize them here
        if self.init_device == "meta":
            self.to_empty(device="cuda")
            self.init_device = "cuda"

        weight = (
            state_dict[prefix + "weight"] if prefix + "weight" in state_dict else None
        )

        if weight is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(weight)
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight_matrix:
                out_features_group, in_features_group = (
                    self.outer_group,
                    self.inner_group,
                )
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group,
                    asym_split=self.asym_split
                )

            state_dict[prefix + "weight"] = weight

        if self.bias is not None:
            bias = (
                state_dict[prefix + "bias"] if prefix + "bias" in state_dict else None
            )
            if bias is not None:
                if bias.size(0) == self.out_features:
                    bias = Drop.apply(bias, self.outer_group)
                    state_dict[prefix + "bias"] = bias
                else:
                    assert (
                        bias.size(0) == self.local_out_features
                    ), "This is neither a full checkpoint nor a sharded checkpoint"

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)

    
