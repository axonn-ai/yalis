import torch
import torch.distributed as dist
import nvshmem_comm_cuda

class NVSHMEMCommHandler:

    process_group_to_idx = {}
    idx_to_comm = {}
    num_comms = 0


    @staticmethod
    def create_communicator_from_process_group(
        process_group: torch.distributed.ProcessGroup,
    ) -> int:
        if process_group not in NVSHMEMCommHandler.process_group_to_idx:
            NVSHMEMCommHandler.process_group_to_idx[process_group] = NVSHMEMCommHandler.num_comms
            NVSHMEMCommHandler.idx_to_comm[NVSHMEMCommHandler.num_comms] = NVSHMEMCommunicator(process_group)
            NVSHMEMCommHandler.num_comms += 1
        return NVSHMEMCommHandler.process_group_to_idx[process_group]
    
    @staticmethod
    def get_communicator_from_idx(idx: int):
        return NVSHMEMCommHandler.idx_to_comm[idx]


class NVSHMEMCommunicator:

    def __init__(self, process_group: torch.distributed.ProcessGroup):
        rank = torch.distributed.get_rank(process_group)
        nranks = torch.distributed.get_world_size(process_group)

        if nranks == 1:
            return

        device = torch.cuda.current_device()


        unique_id = nvshmem_comm_cuda.NVSHMEMCommWrapper.get_unique_id_bytes()

        uid_gpu = unique_id.to("cuda")
        ranks = torch.distributed.get_process_group_ranks(process_group)
        torch.distributed.broadcast(uid_gpu, src=ranks[0], group=process_group)
        torch.distributed.barrier(group=process_group)

        unique_id = uid_gpu.to("cpu")

        self.comm_wrapper = nvshmem_comm_cuda.NVSHMEMCommWrapper(rank, nranks, device, unique_id)
        print(f"NVSHMEMCommunicator created for process group {process_group} with rank {rank} and nranks {nranks}")

    @property
    def core(self):
        return self.comm_wrapper







        
    
