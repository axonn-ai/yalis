# YALIS

**YALIS** stands for **Yet Another LLM Inference System**.  
It is what it is.

YALIS is a modular, high-performance, research-friendly inference system built to plug into LLM training and deployment pipelines. It supports attention backend switching, paged KV caching, intra-head parallelism, and more — with clean APIs and fast execution.

---

## 🚀 Features

- 🔁 **Pluggable attention backends** (`flash`, `sdpa`, `flex`) via a unified API
- 🧠 **Paged KV caching** support for efficient generation
- ⚙️ **Intra-head parallelism** and tensor parallelism
- 📦 Clean Python/C++ extension integration
- 🔍 TorchDynamo/`torch.compile` friendly design

---

## 🛠️ Installing Dependencies


Before installing YALIS, ensure you have **PyTorch (>=2.6)** installed.  
Please refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for installation instructions tailored to your environment.

Once PyTorch is installed, run the following commands in your Python environment to install other dependencies:

```bash
pip install litgpt --no-deps
pip install lightning
pip install transformers
pip install datasets
pip install flash-attn --no-build-isolation
pip install axonn
```

## 🛠️ Building YALIS

To build YALIS (including its C++ extensions) in editable mode, run the following command in your terminal:

```bash
pip install -e .
```

On some systems, however, you might need to set the compilers explicitly via the CC and CXX environment variables. For example, on Cray systems like Perlmutter and Frontier, the default compilers may not be gcc; in those cases, run:

```bash
CC=cc CXX=CXX pip install -e .
```

## 📁 Important Environment Variables for YALIS

Before running anything with YALIS, please ensure the following environment variables are set.

> **Note:**  
> Your `SCRATCH` directory should point to a filesystem with **plenty of space** and **fast I/O**, as model checkpoints can be extremely large.  
> For example, **LLaMA 3 70B requires ~140GB** just for weights. If you're working on an HPC cluster, make sure you're using a burst buffer or high-throughput scratch space — **not your home directory**.

```bash
export HF_HOME="${SCRATCH}/.cache/huggingface"
export HF_TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export YALIS_CACHE="${SCRATCH}/yalis/yalis/external"
```


## 💾 Downloading Model Checkpoints

First, ensure the environment variables discussed in the [previous section](#important-environment-variables-for-yalis) are set appropriately — especially `HF_HOME` and `YALIS_CACHE`.

Then, navigate to the `yalis/external/` directory:

```bash
cd yalis/external
```

To see a list of supported model
```bash
python download.py list
```

### 📥 Downloading a Specific Model
For example, to download Meta Llama-3 8B Instruct:
```bash
export HF_TOKEN="..."  # Required for gated models (e.g. Meta models). See Hugging Face docs.
python download.py meta-llama/Meta-Llama-3-8B-Instruct
```

> 🔑 **Note:** Some models like LLaMA require a Hugging Face token. You can generate one at https://huggingface.co/settings/tokens


## Running 
Let's say we want to run Llama-3 8B Instruct on a single node of Perlmutter. First request an interactive session - 

```bash
salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus 4 --account=m4641_g
```

Once a node has been granted to you, run the following command
```bash
bash scripts/run_pm.sh
```

On other clusters, modify the workflow and scripts accordingly.
