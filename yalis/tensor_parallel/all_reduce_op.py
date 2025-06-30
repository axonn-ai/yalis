import torch
from yalis.external.nccl_comm import CommHandler
from typing import Optional, Sequence
from axonn import axonn as ax

try:
    import torch.distributed._symmetric_memory as symm_mem

    HAS_TORCH_SYMMETRIC = True
except ImportError:
    HAS_TORCH_SYMMETRIC = False

symmetric_mem_cache = {}

def get_symmetric_memory_tensor(tensor_numel, tensor_dtype, tensor_device, tp_group, tag=None):
    """
    Gets or creates a symmetric memory tensor with specified properties.

    Reuses cached tensors when available to avoid redundant creation and rendezvous operations.

    Note: This function always returns a 1D tensor.

    Parameters
    ----------
    tensor_numel : int
        Number of elements in the tensor.
    tensor_dtype : torch.dtype
        Data type of the tensor.
    tensor_device : torch.device
        Device on which to allocate the tensor.
    tp_group : dist_group_type
        Process group for rendezvous operation.
    tag : Any, optional
        Optional identifier to further distinguish tensors.

    Returns
    -------
    torch.Tensor
        A symmetric memory tensor with the specified properties.
    """
    # Create a cache key based on tensor properties and group
    cache_key = (tensor_numel, tensor_dtype, tensor_device, tp_group.group_name, tag)

    # Check if we already have a symmetric memory tensor for this configuration
    if cache_key not in symmetric_mem_cache:
        # Create a new symmetric memory tensor if not in cache
        msg = symm_mem.empty(
            tensor_numel,
            dtype=tensor_dtype,
            device=tensor_device,
        )
        # Perform the rendezvous once for this tensor
        symm_mem.rendezvous(msg, group=tp_group)
        # Store in cache
        symmetric_mem_cache[cache_key] = msg
    else:
        # Reuse the existing symmetric memory tensor
        msg = symmetric_mem_cache[cache_key]

    return msg


def symmetric_all_reduce(
    inp: torch.Tensor,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    all_reduce_type: str = "multimem_all_reduce",
):
    """
    Performs an all-reduce operation across multiple processes using symmetric memory.
    If the input tensor is already in the symmetric memory cache we can avoid copy
    overheads by just directly using the input tensor for all reduce.  Externally
    created symmetric memory tensors not in the cache currently will not be able to
    avoid the extra copies.

    Parameters
    ----------
    inp : torch.Tensor
        The input tensor to be reduced. The operation is performed in-place.

    tp_group : Optional[dist_group_type], default=None
        The process group over which to perform the all-reduce operation.
        If None, the default process group is used.

    async_op : bool, default=False
        Whether to perform the operation asynchronously.
        Note: Currently only synchronous operations are supported for symmetric memory variants.

    all_reduce_type : str, default="multimem_all_reduce"
        The type of all-reduce implementation to use. Options include:
        - "nccl": Standard PyTorch distributed all-reduce
        - "multimem_all_reduce": multimem symmetric all-reduce
        - "two_shot": Two-shot symmetric all-reduce
        - "one_shot": One-shot symmetric all-reduce

    Returns
    -------
    Tuple[torch.Tensor, Optional[torch.distributed.Work]]
        - The first element is the input tensor with the all-reduce result.
        - The second element is the async work handle if async_op=True,
          otherwise None.
    """
    assert HAS_TORCH_SYMMETRIC, "Could not import symetric memory from torch"

    if torch.distributed.get_world_size(tp_group) == 1:
        return inp, None

    if all_reduce_type == "nccl":
        # Standard all-reduce implementation
        torch.distributed.all_reduce(inp, group=tp_group)
        return inp

    all_reduce_impl = None
    if all_reduce_type == "multimem_all_reduce":
        all_reduce_impl = torch.ops.symm_mem.multimem_all_reduce_
    elif all_reduce_type == "two_shot":
        all_reduce_impl = torch.ops.symm_mem.two_shot_all_reduce_
    elif all_reduce_type == "one_shot":
        all_reduce_impl = torch.ops.symm_mem.one_shot_all_reduce
    else:
        raise TypeError(f"All reduce type {all_reduce_type} is not supported.")

    group_name = tp_group.group_name
    tensor_shape = inp.shape
    tensor_numel = inp.numel()
    tensor_dtype = inp.dtype
    tensor_device = inp.device

    input_id = id(inp)
    is_cached = any(id(cached_tensor) == input_id for cached_tensor in symmetric_mem_cache.values())
    # Check if the input tensor is already in the symmetric memory cache. If it is we can avoid copy overheads.
    if is_cached:
        all_reduce_impl(
            inp,
            "sum",
            group_name,
        )
    else:
        # Get symmetric memory tensor. Build or retrieve from cache.
        msg = get_symmetric_memory_tensor(tensor_numel, tensor_dtype, tensor_device, tp_group)

        msg.copy_(inp.reshape(-1))

        all_reduce_impl(
            msg,
            "sum",
            group_name,
        )

        # Copy the result back to the input tensor
        inp.copy_(msg.reshape(tensor_shape))

    return inp

@torch.library.custom_op("yalis::tp_all_reduce", mutates_args=("x",))
def tp_all_reduce(x: torch.Tensor, inner_nccl_comm_idx: int) -> None:
    inner_nccl_comm = CommHandler.get_communicator_from_idx(
        inner_nccl_comm_idx
    )
    inner_nccl_comm.all_reduce(x)


@tp_all_reduce.register_fake
def _(x, inner_nccl_comm_idx):
    pass


@torch.library.custom_op("yalis::symmetric_all_reduce", mutates_args=("x",))
def symmetric_all_reduce_op(
    x: torch.Tensor,
    tp_dims: Optional[Sequence[int]] = None,
    transpose: bool = False,
    all_reduce_type: str = "multimem_all_reduce",
) -> torch.Tensor:
    inner_group, outer_group, depth_group = ax.comm_handle.get_intra_layer_groups(tp_dims)
    if transpose:
        inner_group, outer_group = outer_group, inner_group

    return symmetric_all_reduce(x, inner_group, all_reduce_type)


@symmetric_all_reduce_op.register_fake
def _(x, tp_dims, transpose, all_reduce_type):
    return torch.empty_like(x)
