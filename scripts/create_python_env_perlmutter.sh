#!/bin/bash
#
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

WRKSPC=$SCRATCH
# everything will be installed in $WRKSPC

ENV_NAME="yalis_venv"
# this is the name of your python venv, change if needed

YALIS_DIR=$(pwd)   # Save the current directory before changing it

cd $WRKSPC
echo -e "${RED}Creating Python Environment in $WRKSPC:${GREEN}"
module load python

python -m venv $WRKSPC/$ENV_NAME 
module unload python

echo -e "${RED}Installing Dependencies:${GREEN}"
module load cudatoolkit/12.4
source $WRKSPC/$ENV_NAME/bin/activate
pip3 install torch torchvision torchaudio


#Step 2 - install axonn from source
git clone git@github.com:axonn-ai/axonn.git
cd axonn
pip install -e .

#Step 3 - install other packages
pip install litgpt --no-deps
pip install lightning
pip install transformers
pip install datasets
pip install flash-attn --no-build-isolation
pip install axonn

cd ${YALIS_DIR}
CC=cc CXX=CC pip install -e .

python -c "import torch; print(torch.__version__)"
echo -e "${RED}Your Python Environment is ready. To activate it run the following commands in the SAME order:${NC}"
echo -e "${GREEN}source $WRKSPC/$ENV_NAME/bin/activate${NC}"
echo ""
echo -e "${NC}"
