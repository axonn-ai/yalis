import torch
from yalis.external.nccl_comm import CommHandler
from yalis.tensor_parallel.nvshmem_comm import NVSHMEMCommHandler

@torch.library.custom_op("yalis::tp_all_reduce", mutates_args=("x",))
def tp_all_reduce(x: torch.Tensor, inner_nccl_comm_idx: int) -> None:
    inner_nccl_comm = CommHandler.get_communicator_from_idx(
        inner_nccl_comm_idx
    )
    inner_nccl_comm.all_reduce(x)


@tp_all_reduce.register_fake
def _(x, inner_nccl_comm_idx):
    pass


@torch.library.custom_op(
    "yalis::matmul_with_two_shot_allreduce", mutates_args=["out"]
)
def matmul_with_two_shot_allreduce(
    out: torch.Tensor,
    out_id: int,
    x: torch.Tensor,
    w: torch.Tensor,
    inner_group_name: str,
) -> torch.Tensor:
    torch.mm(x.view(-1, x.shape[-1]), w.t(), out=out.view(-1, w.shape[0]))
    torch.ops.symm_mem.two_shot_all_reduce_(out, "sum", inner_group_name)
    return out.clone()


@matmul_with_two_shot_allreduce.register_fake
def _(out, out_id, x, w, inner_group_name):
    return torch.empty_like(out)


@torch.library.custom_op(
    "yalis::matmul_with_one_shot_allreduce", mutates_args=["out"]
)
def matmul_with_one_shot_allreduce(
    out: torch.Tensor,
    out_id: int,
    x: torch.Tensor,
    w: torch.Tensor,
    inner_group_name: str,
) -> torch.Tensor:
    torch.mm(x.view(-1, x.shape[-1]), w.t(), out=out.view(-1, w.shape[0]))
    return torch.ops.symm_mem.one_shot_all_reduce(out, "sum", inner_group_name)

@matmul_with_one_shot_allreduce.register_fake
def _(out, out_id, x, w, inner_group_name): 
    return torch.empty_like(out)



@torch.library.custom_op(
    "yalis::matmul_with_nvshmem_all_reduce", mutates_args=["x"]
)
def matmul_with_nvshmem_all_reduce(
    out: torch.Tensor,
    out_id: int,
    x: torch.Tensor,
    w: torch.Tensor,
    inner_group_idx: int,
) -> torch.Tensor:
    torch.mm(x.view(-1, x.shape[-1]), w.t(), out=out.view(-1, w.shape[0]))
    nvshmem_comm = NVSHMEMCommHandler.get_communicator_from_idx(inner_group_idx)
    nvshmem_comm.core.allreduce_preallocated(out, out_id, torch.cuda.current_stream().cuda_stream, "recursive")
    return out.clone()

@matmul_with_nvshmem_all_reduce.register_fake
def _(out, out_id, x, w, inner_group_idx):
    return torch.empty_like(out)
