# for use w/ interactive jobs
export HF_HOME="${SCRATCH}/.cache/huggingface"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"
module load cuda/12.6.3
#module load nccl
. $SCRATCH/yalis_venv/bin/activate
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 2 )) # 
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=1,0 #3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export MPICH_GPU_SUPPORT_ENABLED=0
export RANK=${SLURM_PROCID}
