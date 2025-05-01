#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m4641_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu

export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_SUPPORT_ENABLED=0


SCRIPT="examples/infer.py"
export PYTHONPATH="$PYTHONPATH:."
chmod +x scripts/get_rank.sh
run_cmd="mpiexec -n 8 --ppn 4 --depth=1 --cpu-bind depth --env NCCL_CUMEM_ENABLE=0 -env TORCH_NCCL_AVOID_RECORD_STREAMS=1 ./scripts/get_rank_polaris.sh python -u $SCRIPT"


echo $run_cmd
eval $run_cmd
