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

#module load pytorch/2.3.1
module load cudatoolkit/12.4
. $SCRATCH/axonn_venv/bin/activate

NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 ))
## master addr and port
#NNODES=1
#GPUS=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
#export NCCL_NET="AWS Libfabric"
#export FI_CXI_RDZV_THRESHOLD=0
#export FI_CXI_RDZV_GET_MIN=0
#export FI_CXI_OFLOW_BUF_SIZE=1073741824
#export FI_CXI_OFLOW_BUF_COUNT=1
export MPICH_GPU_SUPPORT_ENABLED=0


SCRIPT="examples/infer.py"
export PYTHONPATH="$PYTHONPATH:."
run_cmd="srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./get_rank.sh python -u $SCRIPT"

echo $run_cmd
eval $run_cmd
