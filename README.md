# YALIS

**YALIS** stands for **Yet Another LLM Inference System**.  
It is what it is.

YALIS is a modular, high-performance, research-friendly inference system built to plug into LLM training and deployment pipelines. It supports attention backend switching, paged KV caching, intra-head parallelism, and more — with clean APIs and fast execution.

---

## 🚀 Features

- 🔁 **Pluggable attention backends** (`flash`, `sdpa`, `flex`) via a unified API
- 🧠 **Paged KV caching** support for efficient generation
- ⚙️  **2D tensor parallelism** for large scale multi-node inference
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
> For example, **LLaMA 3 70B requires ~140GB** just for weights. If you're working on an HPC cluster, make sure you're using a high-throughput scratch space — **not your home directory**.

```bash
SCRATCH="... some high performance file system"
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

## CPU Offloading

CPU offloading allows you to run models that don't fit entirely in GPU memory by keeping model weights on CPU and streaming layers to GPU on demand. YALIS uses async CUDA streams to overlap computation with data transfer, so the next layer is prefetched while the current layer executes.

### Enabling CPU Offloading

Set `use_cpu_offloading=True` in your `InferenceConfig`:

```python
from yalis import ModelConfig, InferenceConfig, LLMEngine

model_config = ModelConfig(model_name="Qwen/Qwen3-30B-A3B-Instruct-2507", precision="bf16")

inference_config = InferenceConfig(
    max_batch_size=1,
    attention_backend="flash",
    # CPU offloading options
    use_cpu_offloading=True,
    cpu_offload_num_prefetch_layers=1,  # Number of layers to prefetch ahead
    cpu_offload_pin_memory=True,        # Pin CPU memory for faster transfers
    cpu_offload_use_preallocated_buffers=True,  # Zero-allocation GPU buffers
    cpu_offload_components=["mlp"],     # Which components to offload (see below)
    cpu_offload_mode="all",             # Prefetch mode (see below)
)

engine = LLMEngine(model_config=model_config, inference_config=inference_config)
```

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_cpu_offloading` | `False` | Enable CPU offloading |
| `cpu_offload_mode` | `"all"` | Prefetch mode: `"all"` (full layers), `"rows"` (sparse MoE rows), `"inline"` (on-demand after routing) |
| `cpu_offload_num_prefetch_layers` | `1` | Number of layers to prefetch ahead of execution |
| `cpu_offload_pin_memory` | `True` | Pin CPU memory for faster CPU-to-GPU transfers |
| `cpu_offload_use_preallocated_buffers` | `False` | Use fixed GPU buffers with `.copy_()` instead of `.to()` |
| `cpu_offload_components` | `["mlp", "attn", "norm"]` | Components to offload. Options: `"mlp"`, `"attn"`, `"norm"`. Non-listed components stay on GPU permanently |

### Selective Component Offloading

You can choose which layer components are offloaded to CPU:

- `["mlp", "attn", "norm"]` — Full layer offload (default). Minimum GPU memory usage.
- `["mlp"]` — Only MLP weights offloaded. Attention and norms stay on GPU. Good for MoE models where expert weights dominate memory.
- `["attn"]` — Only attention weights offloaded.

### Prefetch Modes

- **`"all"`** — Prefetches the entire next layer(s) while the current layer runs. Simple and effective for dense models.
- **`"rows"`** — Prefetches only selected rows (experts) of the next MoE layer. Reduces transfer volume for sparse models.
- **`"inline"`** — Computes routing first, then fetches only the needed expert rows before execution. Highest precision but no overlap with prior layer.

### Example

See `examples/infer_cpu_offload.py` for a complete working example.

## Default-Vector MoE Prefetch

For MoE models, YALIS supports a default-vector-based routing prefetch scheme. Instead of waiting for the current layer's MoE output to route the next layer, it uses precomputed default vectors to estimate the next layer's expert routing in advance.

### Enabling Prefetch

```python
inference_config = InferenceConfig(
    # ...
    use_prefetched=True,
    prefetch_default_vect_path="./defaultvect/dv_buff_qwen_instruct/",
)
```

The `prefetch_default_vect_path` directory should contain one file per MoE layer: `buff_0.pt`, `buff_1.pt`, etc. Each file stores a `{"default_vect": tensor}` dict with shape `(n_expert, n_embd)`.

### Combining with CPU Offloading

Prefetch and CPU offloading can be used together. When both are active, the prefetched expert IDs are forwarded to the offload manager so it can selectively stream only the needed expert rows for the next layer:

```python
inference_config = InferenceConfig(
    # ...
    use_cpu_offloading=True,
    cpu_offload_mode="rows",
    cpu_offload_components=["mlp"],
    use_prefetched=True,
    prefetch_default_vect_path="./defaultvect/dv_buff_qwen_instruct/",
)
```

See `examples/infer.py` for a complete MoE prefetch example.
