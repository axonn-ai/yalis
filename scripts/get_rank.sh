#!/bin/bash
# select_gpu_device wrapper script
export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/$USER/.cache/torchinductor/torchinductor_${RANK}"
export TRITON_HOME="/dev/shm/$USER/.cache/triton/triton_${RANK}"
export TRITON_CACHE_DIR="/dev/shm/$USER/.cache/triton/triton_${RANK}"

exec nsys profile -o yalistrace -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node $*
#exec $*
