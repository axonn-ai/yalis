#!/bin/bash
# select_gpu_device wrapper script
export RANK=${SLURM_PROCID}
exec $*
