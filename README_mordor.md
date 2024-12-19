# yalis
YALIS stands for Yet Another LLM Inference System. It is what it is. 
On Mordor, please clone this repo in `/scratch0/$(whoami)`

## Installing Dependencies
Go to `scripts` and run:
```bash
bash create_python_env_mordor.sh
```
This should create a python environment for you with all dependencies. 

## Important Environment Variables for YALIS
Before running anything with yalis, please ensure that the following environment variables are set.

```bash
export SCRATCH=/scratch0/$(whoami)
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"

```

All env variables prefixed by HF are used by huggingface to store it's model checkpoints. `YALIS_CACHE` is where 
YALIS stores it's checkpoints. If you do not set `YALIS_CACHE`, models will be downloaded to your home directory.
This can be undesirable if you have limited storage in your home directory, which is often the case on HPC clusters.

## Downloading Model Checkpoints
First, ensure the environment variables discussed in the previous section are set appropriately. 
Then, go to `yalis/external`. Run the following to get a list of supported models:

```bash
python download.py list
```

Now say you want to download Meta Llama-3 8B Instruct. Run:

```bash
export HF_TOKEN=".." # for models that require authorization. See huggingface docs for more info.
python download.py meta-llama/Meta-Llama-3-8B-Instruct
```

## Running 
Let's say we want to run Llama-3 8B Instruct on mordor. Run the following command.

```bash
bash scripts/run_mordor.sh
```

This will basically run `examples/infer.py`.
