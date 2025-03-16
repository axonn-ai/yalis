#!/bin/bash -l
#PBS -l select=4:system=polaris
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A DemocAI

. ~/.zshrc
module use /soft/modulefiles
module load conda/2024-04-29-aws-nccl
module load cudatoolkit-standalone/12.6.0.lua
. activate /lus/eagle/projects/DemocAI/prajwal/conda_envs/yalisenv-torch26
#source /lus/eagle/projects/DemocAI/prajwal/venvs/yalis-env26/bin/activate

export YALIS_DIR="${SCRATCH}/yalis/"
cd ${YALIS_DIR}

export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

#export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_CROSS_NIC=1
#export NCCL_COLLNET_ENABLE=1
#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.6.0/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_DEFAULT_CQ_SIZE=131072

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export CUDA_VISIBLE_DEVICES=3,2,1,0


#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_DEFAULT_CQ_SIZE=131072
#export FI_CXI_DEFAULT_TX_SIZE=131072
#export FI_CXI_RDZV_PROTO=alt_read
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_REQ_BUF_SIZE=16MB

#watch -n 1 nvidia-smi | tee -l nvsmi.out &
#
export NCCL_CUMEM_ENABLE=0 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

SCRIPT="examples/infer_speculative.py"
export PYTHONPATH="$PYTHONPATH:."
echo $PATH
chmod +x scripts/get_rank_polaris.sh
run_cmd="mpiexec -n 8 --ppn 4 --depth=4 --cpu-bind depth -env NCCL_CUMEM_ENABLE=0 -env TORCH_NCCL_AVOID_RECORD_STREAMS=1 ./scripts/get_rank_polaris.sh /lus/eagle/projects/DemocAI/prajwal/conda_envs/yalisenv-torch26/bin/python -u $SCRIPT | tee yalis.out 2>&1"

echo $run_cmd
eval $run_cmd
