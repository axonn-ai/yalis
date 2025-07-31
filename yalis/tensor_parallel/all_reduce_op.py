import torch
from yalis.external.nccl_comm import CommHandler


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
    x: torch.Tensor,
    w: torch.Tensor,
    inner_group_name: str,
) -> torch.Tensor:
    torch.mm(x.view(-1, x.shape[-1]), w.t(), out=out.view(-1, w.shape[0]))
    torch.ops.symm_mem.two_shot_all_reduce_(out, "sum", inner_group_name)
    return out.clone()


@matmul_with_two_shot_allreduce.register_fake
def _(out, x, w, inner_group_name):
    return torch.empty_like(out)


@torch.library.custom_op(
    "yalis::matmul_with_one_shot_allreduce", mutates_args=["out"]
)
def matmul_with_one_shot_allreduce(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    inner_group_name: str,
) -> torch.Tensor:
    torch.mm(x.view(-1, x.shape[-1]), w.t(), out=out.view(-1, w.shape[0]))
    return torch.ops.symm_mem.one_shot_all_reduce(out, "sum", inner_group_name)


@matmul_with_one_shot_allreduce.register_fake
def _(out, x, w, inner_group_name):
    return torch.empty_like(out)