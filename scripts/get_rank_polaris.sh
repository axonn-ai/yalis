#!/bin/bash -l
num_gpus=4
# need to assign GPUs in reverse order due to topology
# See Polaris Device Affinity Information https://www.alcf.anl.gov/support/user-guides/polaris/hardware-overview/machine-overview/index.html
gpu=$((${num_gpus} - 1 - ${PMI_LOCAL_RANK} % ${num_gpus}))
export CUDA_VISIBLE_DEVICES=$gpu
export RANK=${PMI_RANK}
export WORLD_SIZE=${PMI_SIZE}
export TORCHINDUCTOR_CACHE_DIR="/local/scratch/.cache/torch_inductor_${RANK}"

export NCCL_CUMEM_ENABLE=0 
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
#export NCCL_DEBUG=INFO
#export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_CROSS_NIC=1
#export NCCL_COLLNET_ENABLE=1
#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
#export FI_CXI_DISABLE_HOST_REGISTER=1
#export FI_MR_CACHE_MONITOR=userfaultfd
#export FI_CXI_DEFAULT_CQ_SIZE=131072

#!/bin/bash
# USE AWS 1.6.0 plugin
#export AWS_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && cd ../aws/v1.6.0 && pwd )

#export NCCL_NET="AWS Libfabric"
#export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
#export NCCL_CROSS_NIC=1
#export FI_CXI_RX_MATCH_MODE=software
#export FI_CXI_RDZV_PROTO=alt_read
#export FI_CXI_REQ_BUF_SIZE=8388608
#export FI_CXI_DEFAULT_TX_SIZE=1028576
#export FI_CXI_RDZV_THRESHOLD=16384
##export NCCL_DEBUG=TRACE
##export NCCL_NET_GDR_LEVEL=PHB
#export NCCL_COLLNET_ENABLE=1
#export NCCL_SOCKET_IFNAME=hsn

#echo $LD_LIBRARY_PATH
export TORCH_LOGS="recompiles,dynamic"

exec $*
