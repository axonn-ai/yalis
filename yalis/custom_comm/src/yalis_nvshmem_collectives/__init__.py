from .comm import NVSHMEMCommunicator, NVSHMEMCommHandler

try:
    from . import nvshmem_comm_cuda  # available if NVSHMEM build succeeded
    HAS_NVSHMEM = True
except Exception:
    nvshmem_comm_cuda = None  # type: ignore
    HAS_NVSHMEM = False


