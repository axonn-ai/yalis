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