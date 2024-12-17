# yalis
YALIS stands for Yet Another LLM Inference System. It is what it is. 

(On Perlmutter, please clone this repo in `${SCRATCH}`)

## Installing Dependencies
If you are on Perlmutter, very cool. Go to `scripts` and do:
```bash
bash create_python_env_perlmutter.sh
```
This should create a python environment for you with all dependencies. 

## Important Environment Variables for YALIS
Before running anything with yalis, please ensure that the following environment variables are set.
```bash 
export HF_HOME=... # some location with large disk space.
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCHINDUCTOR_CACHE_DIR=... # some location with large disk space.
export YALIS_CACHE=... # some location with large disk space.

```
All env variables prefixed by HF are used by huggingface to store it's model checkpoints. `YALIS_CACHE` is where 
YALIS stores it's checkpoints. If you do not set YALIS_CACHE, then models will be downloaded to your home directory, which
can we undesirable.

If you are on Perlmutter, just use
```bash
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH}/.cache/torch_inductor"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"

```

## Downloading Model Checkpoints
First, ensure the environment variables discussed in the previous section are set appropriately. 
Then, go to `yalis/external`. Run the following to get a list of supported models:

```bash
python download.py list

```

Now say you want to download Llama 2 7B chat. Run:

```bash
python download.py meta-llama/Llama-2-7b-chat-hf
```