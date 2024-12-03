#!/bin/bash
# select_gpu_device wrapper script
export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
#export NCCL_DEBUG=INFO
#SCRIPT="nsys profile -s none \
#	--gpu-metrics-device all\
#	-t nvtx,cuda -o ${SCRATCH}/GodonBellData/traces/test_${RANK}.qdrep \
#	--force-overwrite=true  \
#	$* \
#	"
exec $*
