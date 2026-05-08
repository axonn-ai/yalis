#!/bin/bash
NNODES=1
GPUS_DEFAULT=1
GPUS=${GPUS:-$GPUS_DEFAULT}


export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}

## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsi
export MPICH_GPU_SUPPORT_ENABLED=0
export CUDA_VISIBLE_DEVICES=0

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_HOME="$SCRATCH/hf_cache"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache"
export YALIS_CACHE="/scratch/11195/akshitab/yalis/yalis/external"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"

SCRIPT_SDPA="-c tests/basic_correctness/sdpa.ini tests/basic_correctness/test_yalis_vs_hf.py"
SCRIPT_FLASH="-c tests/basic_correctness/flash.ini tests/basic_correctness/test_yalis_vs_hf.py"

export PYTHONPATH="$PYTHONPATH:."

chmod +x tests/get_rank_tests.sh

# SDPA Tests
# SDPA Tests (line 33)
sdpa_cmd="NCCL_CUMEM_ENABLE=0 srun -N $NNODES -n $GPUS -c 16 --cpu-bind=cores ./tests/get_rank_tests.sh pytest $SCRIPT_SDPA"
echo $sdpa_cmd
eval $sdpa_cmd

# Flash Tests (line 38)
flash_cmd="NCCL_CUMEM_ENABLE=0 srun -N $NNODES -n $GPUS -c 16 --cpu-bind=cores ./tests/get_rank_tests.sh pytest $SCRIPT_FLASH"
echo $flash_cmd
eval $flash_cmd