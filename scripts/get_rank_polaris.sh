#!/bin/bash -l
num_gpus=4
# need to assign GPUs in reverse order due to topology
# See Polaris Device Affinity Information https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
export RANK=${PMI_RANK}
export WORLD_SIZE=${PMI_SIZE}
exec $*
