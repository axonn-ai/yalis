# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# This file is inspired by
#       https://github.com/vllm-project/vllm/blob/main/vllm/distributed/device_communicators/pynccl.py

import ctypes
import torch
from yalis.utils import get_platform

# NCCL constants
NCCL_UNIQUE_ID_BYTES = 128


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * NCCL_UNIQUE_ID_BYTES)]


ncclComm_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: torch.distributed.ReduceOp) -> int:
        if op == torch.distributed.ReduceOp.SUM:
            return cls.ncclSum
        if op == torch.distributed.ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == torch.distributed.ReduceOp.MAX:
            return cls.ncclMax
        if op == torch.distributed.ReduceOp.MIN:
            return cls.ncclMin
        if op == torch.distributed.ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


# This class is used to handle the NCCL communicator creation and destruction
# and keeps a cache of communicators for each process group
class CommHandler:
    # Map from process group to a unique id (different from the NCCL unique id)
    process_group_to_idx = {}
    # Map from index to NCCL communicator
    idx_to_comm = {}
    # Number of communicators created
    num_comms = 0
    # NCCL library
    comm_lib = None

    @staticmethod
    def load_commlib():
        platform = get_platform()
        if platform == "cuda":
            CommHandler.comm_lib = ctypes.cdll.LoadLibrary("libnccl.so.2")
            CommHandler.num_comms = 0
        elif platform == "rocm":
            CommHandler.comm_lib = ctypes.cdll.LoadLibrary("librccl.so")
            CommHandler.num_comms = 0
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    @staticmethod
    def create_communicator_from_process_group(
        process_group: torch.distributed.ProcessGroup,
    ) -> int:
        """
        Create a communicator from a process group and return the index
        of the communicator. If the communicator already exists, return
        the index of the existing communicator.
        Input:
            process_group: torch.distributed.ProcessGroup
        Output:
            idx: int
            The index of the communicator.
        """
        if CommHandler.comm_lib is None:
            CommHandler.load_commlib()

        if process_group not in CommHandler.process_group_to_idx:
            idx = CommHandler.num_comms
            CommHandler.num_comms += 1

            CommHandler.process_group_to_idx[process_group] = idx
            CommHandler.idx_to_comm[idx] = NCCLCommunicator(process_group)

        idx = CommHandler.process_group_to_idx[process_group]
        return idx

    @staticmethod
    def get_communicator_from_idx(idx: int):
        """
        Get a communicator from an index.
        Input:
            idx: int - The index of the communicator created by
                       CommHandler.create_communicator_from_process_group.
        Output:
            comm: NCCLCommunicator - The communicator.
        """
        if CommHandler.comm_lib is None:
            raise RuntimeError(
                "NCCL lib not loaded. Call CommHandler.create_communicator_from_process_group with a process group first."  # noqa: E501
            )
        if idx not in CommHandler.idx_to_comm:
            raise RuntimeError(
                f"NCCL communicator with index {idx} not found. Call CommHandler.create_communicator_from_process_group with a process group first."  # noqa: E501
            )
        return CommHandler.idx_to_comm[idx]

    @staticmethod
    def get_comm_lib():
        if CommHandler.comm_lib is None:
            raise RuntimeError(
                "NCCL lib not loaded. Call CommHandler.load_commlib() first."
            )
        return CommHandler.comm_lib


# Communicator Class for NCCL and RCCL
class NCCLCommunicator:
    def __init__(self, process_group: torch.distributed.ProcessGroup):
        rank = torch.distributed.get_rank(process_group)
        nranks = torch.distributed.get_world_size(process_group)
        device = torch.cuda.current_device()

        if rank == 0:
            unique_id = self.get_unique_id()
        else:
            unique_id = ncclUniqueId()

        tensor = torch.ByteTensor(list(unique_id.internal)).to(device)
        ranks = torch.distributed.get_process_group_ranks(process_group)
        torch.distributed.broadcast(tensor, src=ranks[0], group=process_group)
        byte_list = tensor.cpu().tolist()
        for i, byte in enumerate(byte_list):
            unique_id.internal[i] = byte

        self.initialize_comm(unique_id, nranks, rank, device)

    def initialize_comm(
        self, unique_id: ncclUniqueId, nranks: int, rank: int, device: int
    ):
        self.comm = ncclComm_t()
        print(
            f"NCCLCommunicator: Device: {device}, Rank: {rank}, Nranks: {nranks}"  # noqa: E501
        )
        ret = CommHandler.get_comm_lib().ncclCommInitRank(
            ctypes.byref(self.comm),
            ctypes.c_int(nranks),
            unique_id,
            ctypes.c_int(rank),
        )
        self.check_nccl_error(ret, "ncclCommInitRank failed")
        print(f"Created NCCL communicator for rank {rank}")
        self.nranks = nranks
        self.rank = rank

    def get_comm_size(self):
        return self.nranks

    def get_rank(self):
        return self.rank

    def check_nccl_error(self, ret, msg="NCCL error"):
        if ret != 0:
            # Get error string
            CommHandler.get_comm_lib().ncclGetErrorString.restype = (
                ctypes.c_char_p
            )
            err_str = CommHandler.get_comm_lib().ncclGetErrorString(ret)
            raise RuntimeError(
                f"{msg}: {err_str.decode('utf-8')} (code {ret})"
            )

    def get_unique_id(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        ret = CommHandler.get_comm_lib().ncclGetUniqueId(
            ctypes.byref(unique_id)
        )
        self.check_nccl_error(ret, "ncclGetUniqueId failed")
        return unique_id

    def all_reduce(
        self, tensor, stream=None, op=torch.distributed.ReduceOp.SUM
    ):
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA/ROCm device")

        nccl_dtype = ncclDataTypeEnum.from_torch(tensor.dtype)
        nccl_op = ncclRedOpTypeEnum.from_torch(op)
        if stream is None:
            stream = torch.cuda.current_stream(tensor.device)
        ret = CommHandler.get_comm_lib().ncclAllReduce(
            buffer_type(tensor.data_ptr()),
            buffer_type(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel()),
            nccl_dtype,
            nccl_op,
            self.comm,
            ctypes.c_void_p(stream.cuda_stream),
        )
        self.check_nccl_error(ret, "ncclAllReduce failed")

    def destroy(self):
        if self.comm:
            CommHandler.get_comm_lib().ncclCommDestroy(self.comm)
            self.comm = None

    def __del__(self):
        self.destroy()
