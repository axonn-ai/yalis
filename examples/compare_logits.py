#!/usr/bin/env python3
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis.model import get_model
from yalis.constants import EnginePhase

model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

# Setup
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "How to bake a cake?"}],
    add_generation_prompt=True,
    tokenize=False,
)
inputs = tokenizer(prompt, return_tensors="pt")

# HuggingFace forward pass
print("Loading HuggingFace model...")
hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True
)
hf_model.eval()
with torch.no_grad():
    hf_inputs = {k: v.to("cuda") for k, v in inputs.items()}
    hf_outputs = hf_model(**hf_inputs, output_hidden_states=False)
    hf_logits = hf_outputs.logits[0, -1, :]  # last token logits
    hf_top_tokens = torch.topk(hf_logits, 5)

print("HF top 5 next tokens:", hf_top_tokens.indices.cpu().tolist())
print("HF top 5 logits:", hf_top_tokens.values.cpu().tolist())

# Initialize distributed for YALIS (torchrun sets RANK, WORLD_SIZE, etc.)
print("\nInitializing distributed...")
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

# YALIS forward pass
print("Loading YALIS model...")
yalis_model = get_model(
    model_id,
    model_dtype=torch.bfloat16,
    attention_backend="sdpa",
    use_paged_kv_caching=False,
    prestore_kv_cache=True,
)
yalis_model.eval()

# Set up KV cache for single sequence
yalis_model.set_kv_cache(max_batch_size=1, max_seq_length=inputs.input_ids.shape[1])

with torch.no_grad():
    token_ids = inputs.input_ids.to("cuda")
    # GPT.forward requires: input_ids, phase, actual_sequence_lengths (optional)
    yalis_outputs = yalis_model(token_ids, phase=EnginePhase.PREFILL)
    yalis_logits = yalis_outputs["logits"][0, -1, :]  # last token logits
    yalis_top_tokens = torch.topk(yalis_logits, 5)

print("YALIS top 5 next tokens:", yalis_top_tokens.indices.cpu().tolist())
print("YALIS top 5 logits:", yalis_top_tokens.values.cpu().tolist())

# Compare
print("\nComparison:")
print(f"HF top token: {hf_top_tokens.indices[0].item()}")
print(f"YALIS top token: {yalis_top_tokens.indices[0].item()}")
print(f"Match: {hf_top_tokens.indices[0].item() == yalis_top_tokens.indices[0].item()}")

# Check if logits are completely different
logit_diff = (hf_logits - yalis_logits).abs().max().item()
print(f"\nMax logit difference: {logit_diff}")

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()

