#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m2404_g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu

export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
#export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"
export HF_TOKEN=""

module load cudatoolkit/12.4
module load nccl
. $SCRATCH/yalis_venv/bin/activate

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
export MPICH_GPU_SUPPORT_ENABLED=0
export NCCL_CUMEM_ENABLE=0

#SCRIPT="examples/infer.py"
export PYTHONPATH="$PYTHONPATH:."
chmod +x scripts/get_rank.sh

TP_R=$1
TP_C=$2
TOKENS_TO_GENERATE=${4:-512}

run -N2 -n8 --ntasks-per-node=4 \
    ./scripts/get_rank.sh python3 -u examples/benchmark_yalis.py \
    --use_wandb \
    --tokens_to_gen $TOKENS_TO_GENERATE \
    --tp $TP_R $TP_C 1 2>&1 | \
    tee "YALIS_TOKENS_${TOKENS_TO_GENERATE}_TPR_${TPR}_TPC_${TPC}.log" 

#./scripts/get_rank.sh torchrun --nproc_per_node=4 --nnodes=2 examples/benchmark_yalis.py --tp 8 1 1 2>&1 | tee DEBUG_BM_YALIS.log 

#run_cmd="NCCL_CUMEM_ENABLE=0 TORCH_NCCL_AVOID_RECORD_STREAMS=1 srun -C gpu -N $NNODES -n $GPUS -c 32 --cpu-bind=cores --gpus-per-node=4 ./scripts/get_rank.sh python -u $SCRIPT"
#echo $run_cmd
#eval $run_cmd
