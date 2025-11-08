#!/bin/bash
# select_gpu_device wrapper script
export JOBID=${SLURM_JOB_ID}
export RANK=${SLURM_PROCID}
export WORLD_SIZE=${SLURM_NTASKS}
export LOCAL_RANK=${SLURM_LOCALID}
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor/rank_${RANK}"

export HF_HOME="${SCRATCH}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

#cmd="nsys profile -o ./yalistrace_${JOBID}_${RANK} -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node $*"
cmd=$*

exec $cmd
