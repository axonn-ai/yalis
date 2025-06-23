#!/bin/bash
# select_gpu_device wrapper script
export RANK=${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR="/dev/shm/${USER}/.cache/torch_inductor/rank_${RANK}"
exec $*
