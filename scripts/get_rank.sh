#!/bin/bash
# select_gpu_device wrapper script
export RANK=${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR="/tmp/prajwal/torchinductor_${RANK}"
export TRITON_HOME="/tmp/prajwal/triton_${RANK}"
export TRITON_CACHE_DIR="/tmp/prajwal/triton_${RANK}"
exec $*
