#!/bin/bash
#SBATCH --gpus-per-node=4
#SBATCH -A m4641_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
export SCRATCH=$HOME/scratch
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"
export TRITON_CACHE_DIR="${SCRATCH}/.cache/.triton"

module load cuda/12.3.0/gcc/11.3.0/x86_64
echo "Copying python environment to fast node local storage"
start=`date +%s`
mkdir -p /tmp/yalis_venv
tar -xzf ${SCRATCH}/miniconda_yalis.tar.gz -C /tmp/yalis_venv
end=`date +%s`
runtime=$((end-start))
echo "Copy completed. Time taken = ${runtime} s"

unset PYTHONHOME
unset CONDA_PREFIX
unset LD_LIBRARY_PATH

source /tmp/yalis_venv/bin/activate
which python
pip show torch
which torchrun

NNODES=$SLURM_JOB_NUM_NODES


GPUS=$(( NNODES * 2))
## master addr and port
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=${GPUS}
echo $WORLD_SIZE
## nccl env vars to speedup stuff
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NET_GDR_LEVEL=PHB
export CUDA_VISIBLE_DEVICES=1,0
export NCCL_CROSS_NIC=1
export NCCL_CROSS_NIC=1

module load nccl/gcc/11.3.0/
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=0


ARG_FILE="sample_args_file.json"

SCRIPT="examples/infer.py"
export VIRTUAL_ENV="/tmp/yalis_venv"
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PATH="/tmp/yalis_venv/bin:$PATH"
export YALIS_DISABLE_TORCH_COMPILE=1
export TORCH_LOGS="graph_breaks"
export TORCHRUN_PYTHON_EXECUTABLE="/tmp/yalis_venv/bin/python"

export HUGGINGFACE_TOKEN="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"
export HF_TOKEN="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"
run_cmd="/tmp/yalis_venv/bin/python -m torch.distributed.run --nproc_per_node=2 examples/infer.py"
echo "Displaying GPUS"
echo "Displaying GPUS"
echo $GPUS

echo $run_cmd
eval $run_cmd












