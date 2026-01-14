#!/usr/bin/env python3
"""
Compare HuggingFace vs YALIS logits for GPT-OSS-20B using OpenAI Harmony format.
Ensures identical tokenization and channel conditioning for both models.
"""
import gc
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis.model import get_model
from yalis.constants import EnginePhase
from yalis.attention.backends import AttentionBackend

# Local path to GPT-OSS-20B checkpoint
model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

def format_harmony_prompt(prompt_text, tokenizer):
    """
    Renders the Harmony Schema tokens required for GPT-OSS models.
    Structure: <|start|>system...<|end|><|start|>user...<|end|><|start|>assistant<|message|>
    """
    # Standard GPT-OSS system message from OpenAI docs
    system_content = (
        "You are ChatGPT, a large language model trained by OpenAI. "
        "Knowledge cutoff: 2024-06 Current date: 2026-01-13 "
        "Reasoning: high # Valid channels: analysis, commentary, final."
    )
    
    # Constructing the raw string for the Harmony envelope
    # GPT-OSS uses the o200k_harmony tokenizer where these are special tokens
    harmony_string = (
        f"<|start|>system<|message|>{system_content}<|end|>"
        f"<|start|>user<|message|>{prompt_text}<|end|>"
        f"<|start|>assistant<|message|>"
    )
    
    return tokenizer(harmony_string, return_tensors="pt")

# Initialize Tokenizer (o200k_harmony)
print("Loading o200k_harmony tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

prompts_to_test = [
    "The capital of France is",
    "How to bake a cake?",
    "Explain quantum entanglement like I'm five.",
]

for prompt_idx, raw_prompt in enumerate(prompts_to_test):
    print(f"\n\n{'='*80}")
    print(f"TESTING HARMONY PROMPT {prompt_idx}: {repr(raw_prompt)}")
    print(f"{'='*80}")

    # Format input using Harmony tokens
    inputs = format_harmony_prompt(raw_prompt, tokenizer)
    print(f"Prompt tokens shape: {inputs.input_ids.shape}")
    print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")

    # ============================================================================
    # PHASE 1: HuggingFace forward pass
    # ============================================================================
    print("\nPHASE 1: Running HuggingFace model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cuda", 
        dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    hf_model.eval()

    with torch.no_grad():
        hf_inputs = {k: v.to("cuda") for k, v in inputs.items()}
        # Compare raw embeddings
        hf_embed_layer = hf_model.model.embed_tokens
        hf_embeddings_raw = hf_embed_layer(hf_inputs["input_ids"])[0, -1, :].cpu()
        
        hf_outputs = hf_model(**hf_inputs)
        hf_logits = hf_outputs.logits[0, -1, :].cpu().clone()
        del hf_outputs

    hf_top_tokens = torch.topk(hf_logits, 5)
    print(f"HF top 5 tokens: {hf_top_tokens.indices.tolist()}")
    print(f"HF decoded: {[tokenizer.decode([t]) for t in hf_top_tokens.indices.tolist()]}")

    # Cleanup HF
    del hf_model
    del hf_inputs
    torch.cuda.empty_cache()
    gc.collect()

    # ============================================================================
    # PHASE 2: YALIS forward pass
    # ============================================================================
    print("\nPHASE 2: Running YALIS model...")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    yalis_model = get_model(
        model_id,
        model_dtype=torch.bfloat16,
        attention_backend=AttentionBackend.SDPA,
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
        disable_tp=True,
    ).to("cuda")
    yalis_model.eval()

    # Allocate KV cache for sequence length
    yalis_model.set_kv_cache(max_batch_size=1, max_seq_length=inputs.input_ids.shape[1])

    with torch.no_grad():
        token_ids = inputs.input_ids.to("cuda")
        yalis_embeddings = yalis_model.transformer.wte(token_ids)[0, -1, :].cpu()
        
        yalis_outputs = yalis_model(token_ids, phase=EnginePhase.PREFILL)
        yalis_logits_full = yalis_outputs["logits"][0, -1, :].cpu().clone()
        
        # Slice to actual vocab size (Harmony o200k has specific padding)
        actual_vocab_size = yalis_model.config.vocab_size
        yalis_logits = yalis_logits_full[:actual_vocab_size]
        del yalis_outputs

    yalis_top_tokens = torch.topk(yalis_logits, 5)
    print(f"YALIS top 5 tokens: {yalis_top_tokens.indices.tolist()}")
    print(f"YALIS decoded: {[tokenizer.decode([t]) for t in yalis_top_tokens.indices.tolist()]}")

    # ============================================================================
    # PHASE 3: Comparison
    # ============================================================================
    print("\nPHASE 3: Comparison Results")
    print("-" * 40)
    
    emb_diff = (hf_embeddings_raw - yalis_embeddings).abs().max().item()
    print(f"Embedding Max Diff: {emb_diff:.6f}")
    
    logit_diff = (hf_logits - yalis_logits).abs()
    max_diff = logit_diff.max().item()
    print(f"Logit Max Diff:     {max_diff:.4f}")

    # Cleanup YALIS for next prompt
    del yalis_model
    torch.cuda.empty_cache()

if dist.is_initialized():
    dist.destroy_process_group()

print("\nAll Harmony-formatted tests complete.")