#!/usr/bin/env python3
"""
Compare HuggingFace vs YALIS logits for GPT-OSS-20B.
Runs models sequentially to avoid OOM, with proper memory cleanup between runs.
"""
import gc
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis.model import get_model
from yalis.constants import EnginePhase
from yalis.attention.backends import AttentionBackend

model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

# Prepare prompt (shared for both models)
print("Preparing prompt...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

# Test with a simple prompt WITHOUT chat template first to isolate the issue
use_simple_prompt = True
if use_simple_prompt:
    prompt = "The capital of France is"
    print("Using simple prompt (no chat template)")
else:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "How to bake a cake?"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    print("Using Harmony chat template")

inputs = tokenizer(prompt, return_tensors="pt")
print(f"Prompt tokens shape: {inputs.input_ids.shape}")
print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")
print(f"First 10 token IDs: {inputs.input_ids[0, :10].tolist()}")
print(f"Last 10 token IDs: {inputs.input_ids[0, -10:].tolist()}")
print()

# ============================================================================
# PHASE 1: HuggingFace forward pass
# ============================================================================
print("=" * 80)
print("PHASE 1: Running HuggingFace model")
print("=" * 80)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", dtype=torch.bfloat16, trust_remote_code=True
)
hf_model.eval()

with torch.no_grad():
    hf_inputs = {k: v.to("cuda") for k, v in inputs.items()}
    hf_outputs = hf_model(**hf_inputs, output_hidden_states=False)
    # Extract and move to CPU immediately to save GPU memory
    hf_logits = hf_outputs.logits[0, -1, :].cpu().clone()
    del hf_outputs  # Free output tensor

hf_top_tokens = torch.topk(hf_logits, 10)
print(f"HF top 10 tokens: {hf_top_tokens.indices.tolist()}")
print(f"HF top 10 logits: {[f'{v:.4f}' for v in hf_top_tokens.values.tolist()]}")
print(f"HF decoded tokens: {[repr(tokenizer.decode([t])) for t in hf_top_tokens.indices[:5].tolist()]}")

# Critical: Free HuggingFace model memory before loading YALIS
print("\nCleaning up HuggingFace model from GPU...")
del hf_model
del hf_inputs
torch.cuda.empty_cache()
gc.collect()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB allocated")

# ============================================================================
# PHASE 2: YALIS forward pass
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 2: Running YALIS model")
print("=" * 80)

# Initialize distributed for YALIS (required even for single GPU)
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

yalis_model = get_model(
    model_id,
    model_dtype=torch.bfloat16,
    attention_backend=AttentionBackend.SDPA,
    use_paged_kv_caching=False,
    prestore_kv_cache=True,
    disable_tp=True,
)
yalis_model = yalis_model.to("cuda")
yalis_model.eval()

print(f"YALIS config: backend={yalis_model.config.attention_backend}, vocab_size={yalis_model.config.vocab_size}")

# Diagnostic: Check if embeddings are properly loaded
wte_weight = yalis_model.transformer.wte.weight
print(f"Embedding shape: {wte_weight.shape}")
print(f"Embedding stats: mean={wte_weight.mean().item():.6f}, std={wte_weight.std().item():.6f}")
print(f"Embedding for token 200005: mean={wte_weight[200005].mean().item():.6f}, std={wte_weight[200005].std().item():.6f}")
print(f"Embedding for token 6: mean={wte_weight[6].mean().item():.6f}, std={wte_weight[6].std().item():.6f}")

# Allocate KV cache for this sequence length
yalis_model.set_kv_cache(max_batch_size=1, max_seq_length=inputs.input_ids.shape[1])

with torch.no_grad():
    token_ids = inputs.input_ids.to("cuda")
    yalis_outputs = yalis_model(token_ids, phase=EnginePhase.PREFILL)
    # Extract and move to CPU
    # IMPORTANT: YALIS uses padded_vocab_size, but we need to slice to actual vocab_size
    yalis_logits_full = yalis_outputs["logits"][0, -1, :].cpu().clone()
    actual_vocab_size = yalis_model.config.vocab_size
    yalis_logits = yalis_logits_full[:actual_vocab_size]  # Slice to match HF vocab size
    del yalis_outputs
    
    print(f"YALIS vocab info: padded={yalis_logits_full.shape[0]}, actual={actual_vocab_size}")

yalis_top_tokens = torch.topk(yalis_logits, 10)
print(f"YALIS top 10 tokens: {yalis_top_tokens.indices.tolist()}")
print(f"YALIS top 10 logits: {[f'{v:.4f}' for v in yalis_top_tokens.values.tolist()]}")
print(f"YALIS decoded tokens: {[repr(tokenizer.decode([t])) for t in yalis_top_tokens.indices[:5].tolist()]}")

# ============================================================================
# PHASE 3: Comparison
# ============================================================================
print("\n" + "=" * 80)
print("PHASE 3: Comparison Results")
print("=" * 80)

# Check top token match
top_match = hf_top_tokens.indices[0].item() == yalis_top_tokens.indices[0].item()
print(f"Top token match: {'✓ YES' if top_match else '✗ NO'}")
print(f"  HF top token:    {hf_top_tokens.indices[0].item()} (logit: {hf_top_tokens.values[0].item():.4f})")
print(f"  YALIS top token: {yalis_top_tokens.indices[0].item()} (logit: {yalis_top_tokens.values[0].item():.4f})")

# Check top-5 overlap
top5_hf = set(hf_top_tokens.indices[:5].tolist())
top5_yalis = set(yalis_top_tokens.indices[:5].tolist())
overlap = len(top5_hf & top5_yalis)
print(f"\nTop-5 overlap: {overlap}/5 tokens")
print(f"  HF top 5:    {list(top5_hf)}")
print(f"  YALIS top 5: {list(top5_yalis)}")

# Full logit comparison
if hf_logits.shape != yalis_logits.shape:
    print(f"\n⚠️  ERROR: Shape mismatch! HF: {hf_logits.shape}, YALIS: {yalis_logits.shape}")
else:
    # Compute differences
    logit_diff = (hf_logits - yalis_logits).abs()
    max_diff = logit_diff.max().item()
    mean_diff = logit_diff.mean().item()
    
    print(f"\nLogit statistics:")
    print(f"  Max difference:  {max_diff:.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Vocab size:      {hf_logits.shape[0]}")
    
    # Diagnosis
    if max_diff > 100:
        print("\n❌ VERDICT: LARGE DIVERGENCE - Models produce very different outputs")
        print("   This suggests fundamental computation differences (attention/MLP/norms)")
    elif max_diff > 10:
        print("\n⚠️  VERDICT: MODERATE DIVERGENCE - Some numerical differences")
        print("   Could be due to precision, operation ordering, or minor implementation differences")
    elif max_diff > 1:
        print("\n⚡ VERDICT: MINOR DIVERGENCE - Small numerical differences")
        print("   Likely due to floating point precision or operation fusion differences")
    else:
        print("\n✓ VERDICT: MODELS MATCH - Outputs are nearly identical")

# Cleanup
print("\n" + "=" * 80)
del yalis_model
torch.cuda.empty_cache()
if dist.is_initialized():
    dist.destroy_process_group()
print("Cleanup complete.")

