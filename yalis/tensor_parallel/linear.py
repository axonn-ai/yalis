# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.  # noqa
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch
import torch.nn.functional as F

import math

from axonn import axonn as ax
from axonn.intra_layer.communication import Drop
from yalis.external.nccl_comm import CommHandler
from yalis.tensor_parallel.all_reduce_op import tp_all_reduce
from yalis.utils import is_process_group_within_node, print_rank0
from yalis.tensor_parallel.nvshmem_comm import NVSHMEMCommHandler
from yalis_nvshmem_collectives import nvshmem_comm_cuda

from typing import Optional, Sequence
import gc

try:
    import torch.distributed._symmetric_memory as symm_mem
    from yalis.tensor_parallel.all_reduce_op import (
        matmul_with_two_shot_allreduce,
        matmul_with_one_shot_allreduce,
    )

    HAS_TORCH_SYMMETRIC = True
except ImportError:
    HAS_TORCH_SYMMETRIC = False


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
    params, out_features_group, in_features_group
):
    # Drop the in_features dimension (last dimension)
    params = Drop.apply(params, in_features_group, -1)

    # Drop the out_features dimension (second last dimension)
    params = Drop.apply(params, out_features_group, -2)

    params = params.contiguous()
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

    # This line is important within this function:
    # placing outside leads to pytorch not deleting the reserved memory
    torch.cuda.empty_cache()

    # This leads to immediate memory garbage collection but can be slow
    # Probably not needed
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
        **kwargs,
    ):
        super(TPLinear, self).__init__()
        assert expert_mode, "Only expert mode allowed in inference"

        self.init_device = init_device
        # weights are shaped [out_features, in_features]
        # in_features are distributed across self.inner_group (X TP group)
        # out_features are distributed across self.inner_group (Y TP group)
        # if transpose is true then X and Y are swapped
        if (
            tensor_parallel_dims is not None
            and torch.distributed.get_rank() == 0
        ):
            print(
                "Manually setting TP dims for a layer with shape",
                f" - {(in_features, out_features)} | tp-dims = {tensor_parallel_dims}",  # noqa: E501
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        if transpose:
            self.inner_group, self.outer_group = (
                self.outer_group,
                self.inner_group,
            )

        self.inner_nccl_comm_idx = (
            CommHandler.create_communicator_from_process_group(
                self.inner_group
            )
        )

        self.inner_nvshmem_comm_idx = (
            NVSHMEMCommHandler.create_communicator_from_process_group(
                self.inner_group
            )
        )

        # We do not need NCCL communicators for the outer and depth group
        # as no collective is performed on them during the forward pass

        # depth_group is the Z tensor parallel group (akin to FSDP)
        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        assert self.depth_group_size == 1

        # these are the in and out features of the full global weight matrix
        self.in_features = in_features
        self.out_features = out_features

        # expert mode = True -> user parallelizes non-linear layers manually
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
        assert in_features % self.inner_group_size == 0

        # local_in_features - this is the number of in_features on each GPU
        self.local_in_features = divide(in_features, self.inner_group_size)

        # local_out_features - this is the number of out_features on each GPU
        if out_features % self.outer_group_size == 0:
            self.local_out_features = divide(
                out_features, self.outer_group_size
            )
        else:
            self.local_out_features = math.ceil(
                out_features / self.outer_group_size
            )
        # initialize the weight matrix and grab the local slice for each GPU
        initial_params = initialize_params(
            out_features,
            in_features,
            self.outer_group,
            self.inner_group,
            init_method,
            init_device=self.init_device,
        )
        # register the weight matrix as a trainable parameter.
        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)

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
        self.symmetric_memory_tensor = None

    def all_reduce(self, x):
        tp_all_reduce(x, self.inner_nccl_comm_idx)
        #dist.all_reduce(x, op=torch.distributed.ReduceOp.SUM, group=self.inner_group)
        return x

    def matmul(self, w, x):
        return F.linear(x, w)

    def set_symmetric_memory_tensor(
        self,
        max_batch_size,
        max_seq_length,
        dtype,
        device,
        symmetric_memory_pool,
        algorithm,
    ):
        """
        This function is used to set the symmmetric memory
        output tensor for the layer. We check if the
        pool already has a tensor for the current cache key.
        If it does, we use that tensor.
        If it does not, we create a new tensor and add it to the pool.
        """
        cache_key = (
            max_batch_size * max_seq_length * self.local_out_features,
            dtype,
            device,
            self.inner_group.group_name,
        )

        if (
            #not is_process_group_within_node(self.inner_group)
            self.inner_group_size <= 1
            or not HAS_TORCH_SYMMETRIC
        ):
            self.symmetric_memory_tensor = None
            return

        # group_name = torch.distributed.group.WORLD.group_name
        # symm_mem.enable_symm_mem_for_group(group_name)
        # self.inner_group = torch.distributed.group.WORLD

        if cache_key not in symmetric_memory_pool:
            # Create a new tensor and add it to the pool
            nelem = (
                max_batch_size * self.local_out_features
            )  # * max_seq_length -> not needed as we only use it for decode

            nvshmem_comm = NVSHMEMCommHandler.get_communicator_from_idx(self.inner_nvshmem_comm_idx)
            msg, msg_id = nvshmem_comm.core.allocate_tensor(nelem, dtype, device, nvshmem_comm_cuda.Protocol.LL8)
            #msg = symm_mem.empty(
            #    nelem,
            #    dtype=dtype,
            #    device=device,
            #)
            #symm_mem.rendezvous(msg, group=self.inner_group)
            symmetric_memory_pool[cache_key] = (msg, msg_id)
            memory_size = nelem * dtype.itemsize
            print_rank0(
                f"Created symmetric memory tensor for {cache_key} - {memory_size / 1024 / 1024} MB"  # noqa: E501
            )
            num_blocks = max(1, memory_size // 8192)
            num_blocks = min(num_blocks, 16)
            nvshmem_comm.core.set_kernel_params(nvshmem_comm_cuda.Protocol.LL8, num_blocks, 512, 8192)
        self.symmetric_memory_tensor = symmetric_memory_pool[cache_key][0]
        self.symmetric_memory_tensor_id = symmetric_memory_pool[cache_key][1]
        if algorithm == "two-shot":
            self.symmetric_allreduce_matmul_fn = matmul_with_two_shot_allreduce
        elif algorithm == "one-shot":
            self.symmetric_allreduce_matmul_fn = matmul_with_one_shot_allreduce
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")

    def forward(
        self,
        x,
        symmetric_memory_pool=None,
    ):
        if self.symmetric_memory_tensor is not None and x.shape[1] == 1:
            offset = x.shape[0] * x.shape[1] * self.local_out_features
            x = self.symmetric_allreduce_matmul_fn(
                self.symmetric_memory_tensor[:offset],
                self.symmetric_memory_tensor_id,
                x,
                self.weight,
                self.inner_group.group_name,
                self.inner_nccl_comm_idx,
            ).view(*x.shape[:-1], self.local_out_features)
        else:
            x = self.matmul(self.weight, x)
            x = self.all_reduce(x)

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
    def _modified_load_from_state_dict(
        self, state_dict, prefix, *args, **kwargs
    ):
        # If the parameters were initialized on meta-device,
        # we need to materialize them here
        if self.init_device == "meta":
            self.to_empty(device="cuda")
            self.init_device = "cuda"

        weight = (
            state_dict[prefix + "weight"]
            if prefix + "weight" in state_dict
            else None
        )

        if weight is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(weight)
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            # TODO: This can be further optimized potentially
            if is_full_weight_matrix and getattr(
                self, "duplicating_kv", False
            ):
                rank = dist.get_rank(self.outer_group)
                hs = self.head_size
                q_per_rank = self.q_per_rank
                q_per_kv = self.total_n_head // self.total_n_query_groups
                blk = q_per_kv + 2
                q_rows = []
                groups = set()
                for h in range(rank * q_per_rank, (rank + 1) * q_per_rank):
                    g = h // q_per_kv
                    groups.add(g)
                    local_q = h % q_per_kv
                    row0 = (g * blk + local_q) * hs
                    q_rows.extend(range(row0, row0 + hs))
                kv_rows = []
                for g in groups:
                    base = g * blk * hs + q_per_kv * hs
                    kv_rows.extend(range(base, base + hs))
                    kv_rows.extend(range(base + hs, base + 2 * hs))
                local_rows = q_rows + kv_rows
                weight = weight.contiguous()
                weight = weight[local_rows, :].contiguous()
                state_dict[prefix + "weight"] = weight
                if self.weight.shape != weight.shape:
                    self.weight = torch.nn.Parameter(
                        torch.empty_like(weight, device="cuda"),
                        requires_grad=True,
                    )
                self.local_out_features = weight.size(0)
            elif is_full_weight_matrix:
                out_features_group, in_features_group = (
                    self.outer_group,
                    self.inner_group,
                )
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group
                )
                state_dict[prefix + "weight"] = weight
            else:
                state_dict[prefix + "weight"] = weight

        if self.bias is not None:
            if getattr(self, "duplicating_kv", False):
                raise NotImplementedError(
                    "There is currently no support for scaling the R dimension"
                    " > #kv heads when using a model with bias"
                )
            bias = (
                state_dict[prefix + "bias"]
                if prefix + "bias" in state_dict
                else None
            )
            if bias is not None:
                if bias.size(0) == self.out_features:
                    bias = Drop.apply(bias, self.outer_group)
                    state_dict[prefix + "bias"] = bias
                else:
                    assert (
                        bias.size(0) == self.local_out_features
                    ), "This is neither a full nor a sharded checkpoint"

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
