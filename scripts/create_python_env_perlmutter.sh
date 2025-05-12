#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=$SCRATCH
ENV_NAME="yalis_venv_py311"  # Updated to reflect Python 3.11 environment

YALIS_DIR="/pscratch/sd/m/mahua04/yalis"

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"

# Use a specific, modern Python version
module unload python
module load cray-python/3.11.7

python -m venv $WRKSPC/$ENV_NAME 

echo -e "${RED}Installing Dependencies:${GREEN}"
module load cudatoolkit/12.4
source $WRKSPC/$ENV_NAME/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyTorch 2.6.0
pip install torch==2.6.0 torchvision torchaudio 

# Step 2 - install axonn from source
git clone git@github.com:axonn-ai/axonn.git
cd axonn
pip install -e .

# Step 3 - install other packages
pip install litgpt --no-deps
pip install lightning
pip install transformers
pip install datasets
pip install wheel

# Flash-Attn requires proper C/C++ compilers
CC=cc CXX=CC pip install flash-attn --no-build-isolation
pip install axonn

# Back to yalis directory
cd ${YALIS_DIR}
CC=cc CXX=CC pip install -e .

# Confirm installation
python -c "import torch; print(torch.__version__)"

echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}module load cray-python/3.11.7${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
