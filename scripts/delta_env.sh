# for use w/ interactive jobs
export HF_HOME="${SCRATCH}/.cache/huggingface"
#export TORCHINDUCTOR_CACHE_DIR="/dev/shm/$USER/.cache/torch_inductor"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"
module load cuda/12.6.3
#module load cuda/11.8.0 #cuda/12.6.3
module load libfabric/1.15.2.0
#module load nccl/2.19.3-1.awsplugin
#module swap cuda/11.8.0 cuda/12.6.3
#module load nccl
. $SCRATCH/yalis_venv/bin/activate
NNODES=$SLURM_JOB_NUM_NODES
GPUS=$(( NNODES * 4 )) # 
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}
export LD_LIBRARY_PATH=$HOME/ofi-plugin-226/lib:$LD_LIBRARY_PATH 
export LD_PRELOAD=$HOME/ofi-plugin-226/lib/libnccl-net-ofi.so
#export CUDA_DEVICE_MAX_CONNECTIONS=1 # TODO: maybe revert
#export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=2 #PHB # TODO: maybe revert
export CUDA_VISIBLE_DEVICES=3,2,1,0 #3,2,1,0
export NCCL_CROSS_NIC=1
export NCCL_NET=ofi # TODO: add back in
export FI_PROVIDER=cxi # TODO: add back in
#export NCCL_SOCKET_IFNAME=hsn # TODO: maybe revert
export MPICH_GPU_SUPPORT_ENABLED=0
export RANK=${SLURM_PROCID}
