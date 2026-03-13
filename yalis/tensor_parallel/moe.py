# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.  # noqa
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch

import math

from axonn import axonn as ax
from axonn.intra_layer.communication import Drop
from yalis.external.nccl_comm import CommHandler
from yalis.external.fused_moe import (
    fused_moe,
    get_moe_configs,
    get_config_dtype_str,
)


from typing import Optional, Sequence


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
    n_experts,
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty(
        (n_experts, out_features, in_features), device=init_device
    )
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group
    )

    # This line is important within this function:
    # placing outside leads to pytorch not deleting the reserved memory
    torch.cuda.empty_cache()

    return params


@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class TPMoE(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        n_experts,
        n_expert_per_token,
        *args,
        init_method=None,
        tensor_parallel_dims: Optional[Sequence[int]] = None,
        init_device="cuda",
        bias=False,
        skip_bias_add=True,
        dtype=None,
        activation="silu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        **kwargs,
    ):
        super(TPMoE, self).__init__()

        self.init_device = init_device
        # moe has 2 weight matrices:
        # 1. gate_up_proj: [n_experts, 2 * intermediate_size, hidden_size]
        # 2. proj: [n_experts, hidden_size, intermediate_size]
        # The first matrix is sharded as:
        #   2 * intermediate_size -> inner_group,
        #   hidden_size -> outer_group
        # The second matmul can be considered as a TPLinear layer
        # with transpose=True. So, the second matrix is sharded as:
        #   hidden_size -> inner_group,
        #   intermediate_size -> outer_group
        if (
            tensor_parallel_dims is not None
            and torch.distributed.get_rank() == 0
        ):
            print(
                "Manually setting TP dims for a layer with shape -",
                f" gate_up_proj: {(n_experts, 2 * intermediate_size, hidden_size)} |"  # noqa: E501
                f" proj: {(n_experts, hidden_size, intermediate_size)} |"
                f" tp-dims = {tensor_parallel_dims}",
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        self.outer_nccl_comm_idx = (
            CommHandler.create_communicator_from_process_group(
                self.outer_group
            )
        )

        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        assert self.depth_group_size == 1

        # For now we only support inner group size of 1
        # TODO (Prajwal): Support 2D TP for MoE
        # Additionally, we can use the depth group for EP
        assert self.inner_group_size == 1, "Only row TP is supported for MoE"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        if init_method is None:
            init_method = default_init_method

        # hidden_size should be divisible by inner_group_size
        assert hidden_size % self.inner_group_size == 0
        # intermediate_size should be divisible by outer_group_size
        assert intermediate_size % self.outer_group_size == 0

        # Number of features on each GPU
        self.local_hidden_size = divide(hidden_size, self.inner_group_size)
        self.local_intermediate_size = divide(
            intermediate_size, self.outer_group_size
        )

        self.n_experts = n_experts
        self.n_expert_per_token = n_expert_per_token

        # initialize the weight matrix and grab the local slice for each GPU
        w1 = initialize_params(
            n_experts,
            2 * intermediate_size,
            hidden_size,
            self.outer_group,
            self.inner_group,
            init_method,
            init_device=self.init_device,
        )

        w2 = initialize_params(
            n_experts,
            hidden_size,
            intermediate_size,
            self.inner_group,
            self.outer_group,
            init_method,
            init_device=self.init_device,
        )

        # register the weight matrix as a trainable parameter.
        self.gate_up_proj = torch.nn.Parameter(w1, requires_grad=True)
        self.proj = torch.nn.Parameter(w2, requires_grad=True)

        # extra book-keeping for the weight tensor.
        # this is needed by AxoNN layer in the sync_gradients and
        # gradient clipping functions.
        setattr(self.gate_up_proj, "is_tensor_parallel", True)
        setattr(self.proj, "is_tensor_parallel", True)
        setattr(self.gate_up_proj, "needs_depth_parallel_gradient_sync", False)
        setattr(self.proj, "needs_depth_parallel_gradient_sync", False)
        setattr(
            self.gate_up_proj,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )
        setattr(
            self.proj,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )

        # Store activation parameters
        self.activation = activation
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit

        # Initialize bias parameters if needed
        if bias:
            # gate_up_proj bias: [n_experts, 2 * local_intermediate_size]
            gate_up_bias = torch.zeros(
                n_experts,
                2 * self.local_intermediate_size,
                device=self.init_device,
                dtype=dtype,
            )
            self.gate_up_bias = torch.nn.Parameter(
                gate_up_bias, requires_grad=True
            )

            # Projection bias is divided by TP world_size at load time
            # (see _modified_load_from_state_dict) so each rank adds
            # its share inside fused_experts. The all_reduce then
            # recovers the correct full bias.
            proj_bias = torch.zeros(
                n_experts,
                hidden_size,
                device=self.init_device,
                dtype=dtype,
            )
            self.proj_bias = torch.nn.Parameter(proj_bias, requires_grad=True)

            # Set tensor parallel attributes for biases
            setattr(self.gate_up_bias, "is_tensor_parallel", True)
            setattr(self.proj_bias, "is_tensor_parallel", False)
            setattr(
                self.gate_up_bias, "needs_depth_parallel_gradient_sync", False
            )
            setattr(
                self.proj_bias, "needs_depth_parallel_gradient_sync", False
            )
            setattr(
                self.gate_up_bias,
                "process_group_for_norm_reduction",
                ax.comm_handle.intra_layer_group,
            )
        else:
            self.gate_up_bias = None
            self.proj_bias = None

        self.bias = bias

        self.skip_bias_add = skip_bias_add
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

        # Load MoE kernel configs once during init (not in forward pass)
        # This avoids file I/O and CUDA calls during torch.compile/CUDA graph
        device_name = torch.cuda.get_device_name()
        dtype_str = get_config_dtype_str(dtype=dtype)
        self._moe_configs = get_moe_configs(
            E=n_experts,
            N=2 * self.local_intermediate_size,  # N for w1 (gate_up_proj)
            dtype=dtype_str,
            _device_name=device_name,
        )

    def all_reduce(self, x):
        dist.all_reduce(
            x, op=torch.distributed.ReduceOp.SUM, group=self.outer_group
        )
        return x

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
        This function is used to set the symmetric memory
        output tensor for the layer. We check if the
        pool already has a tensor for the current cache key.
        If it does, we use that tensor.
        If it does not, we create a new tensor and add it to the pool.
        """
        # TODO (Prajwal): Implement this
        pass

    def forward(
        self,
        x,
        router,
    ):
        y, topk_weights, topk_ids = fused_moe(
            x,
            self.gate_up_proj,
            self.proj,
            router,
            self.n_expert_per_token,
            True,
            inplace=False,
            moe_configs=self._moe_configs,
            activation=self.activation,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_limit=self.swiglu_limit,
            gate_up_bias=self.gate_up_bias,
            proj_bias=self.proj_bias,
            weight_proj_bias=True,
        )
        y = y.to(x.dtype)

        y = self.all_reduce(y)

        return y

    def _is_full_weight_matrix(self, weight, in_features, out_features):
        return (
            weight.ndim == 3
            and weight.size(1) == out_features
            and weight.size(2) == in_features
        )

    def _is_sharded_weight_matrix(
        self, weight, local_in_features, local_out_features
    ):
        return (
            weight.ndim == 3
            and weight.size(1) == local_out_features
            and weight.size(2) == local_in_features
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

        w1 = (
            state_dict[prefix + "gate_up_proj"]
            if prefix + "gate_up_proj" in state_dict
            else None
        )

        w2 = (
            state_dict[prefix + "proj"]
            if prefix + "proj" in state_dict
            else None
        )

        def modify_state_dict(
            weight,
            prefix,
            is_full_weight_matrix,
            is_sharded_weight_matrix,
            in_features_group,
            out_features_group,
        ):
            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight_matrix:
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group
                )
                state_dict[prefix] = weight
            else:
                state_dict[prefix] = weight

        if w1 is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(
                w1, self.hidden_size, 2 * self.intermediate_size
            )
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(
                w1, self.local_hidden_size, 2 * self.local_intermediate_size
            )
            modify_state_dict(
                w1,
                prefix + "gate_up_proj",
                is_full_weight_matrix,
                is_sharded_weight_matrix,
                self.inner_group,
                self.outer_group,
            )
        if w2 is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(
                w2, self.intermediate_size, self.hidden_size
            )
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(
                w2, self.local_intermediate_size, self.local_hidden_size
            )
            modify_state_dict(
                w2,
                prefix + "proj",
                is_full_weight_matrix,
                is_sharded_weight_matrix,
                self.outer_group,
                self.inner_group,
            )

        # Handle bias parameters if they exist
        if self.bias:
            # Handle gate_up_bias (named mlp1_bias in GPT-OSS)
            gate_up_bias_key = prefix + "gate_up_bias"
            mlp1_bias_key = prefix + "mlp1_bias"

            if gate_up_bias_key in state_dict:
                bias = state_dict[gate_up_bias_key]
            elif mlp1_bias_key in state_dict:
                bias = state_dict[mlp1_bias_key]
                # Move to expected key name
                state_dict[gate_up_bias_key] = bias
                if mlp1_bias_key != gate_up_bias_key:
                    del state_dict[mlp1_bias_key]
            else:
                bias = None

            if bias is not None:
                # bias shape: [n_experts, 2 * intermediate_size]
                # Shard the features dim (dim -1) across outer_group
                # to get [n_experts, 2 * local_intermediate_size]
                if bias.size(-1) == 2 * self.intermediate_size:
                    bias = Drop.apply(bias, self.outer_group, -1)
                    state_dict[gate_up_bias_key] = bias

            # Handle proj_bias (named mlp2_bias in GPT-OSS)
            proj_bias_key = prefix + "proj_bias"
            mlp2_bias_key = prefix + "mlp2_bias"

            if proj_bias_key in state_dict:
                bias = state_dict[proj_bias_key]
            elif mlp2_bias_key in state_dict:
                bias = state_dict[mlp2_bias_key]
                # Move to expected key name
                state_dict[proj_bias_key] = bias
                if mlp2_bias_key != proj_bias_key:
                    del state_dict[mlp2_bias_key]
            else:
                bias = None

            if bias is not None:
                # Divide by TP world_size so each rank adds its
                # share inside fused_experts; all_reduce recovers
                # the correct full bias.
                state_dict[proj_bias_key] = bias / self.outer_group_size

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
