#!/usr/bin/env python3
"""
Comprehensive diagnostic script to compare HF vs YALIS GPT-OSS-20B implementations.
Checks: tokenization, model configs, architecture, attention patterns, and logits.
"""
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis.model import get_model
from yalis.initialize import init_distributed
from yalis.constants import EnginePhase
from yalis.attention.backends import AttentionBackend
import gc

model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

# Initialize
print("="*80)
print("DIAGNOSTIC: HF vs YALIS Comparison")
print("="*80)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)
test_prompt = "The capital of France is"
inputs = tokenizer(test_prompt, return_tensors="pt")

print(f"\n1. TOKENIZATION CHECK")
print(f"   Prompt: {repr(test_prompt)}")
print(f"   Token IDs: {inputs.input_ids.tolist()}")
print(f"   Decoded back: {repr(tokenizer.decode(inputs.input_ids[0]))}")

# Load HF model
print(f"\n2. HUGGINGFACE MODEL")
torch.cuda.empty_cache()
gc.collect()

hf_model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="cuda", 
    dtype=torch.bfloat16, 
    trust_remote_code=True
)
hf_model.eval()

print(f"   Config:")
print(f"     vocab_size: {hf_model.config.vocab_size}")
print(f"     n_layer: {hf_model.config.n_layer}")
print(f"     n_embd: {hf_model.config.n_embd}")
print(f"     n_head: {hf_model.config.n_head}")
print(f"     n_query_groups: {getattr(hf_model.config, 'n_query_groups', 'N/A')}")
print(f"     sliding_window_mode: {getattr(hf_model.config, 'sliding_window_mode', 'N/A')}")
print(f"     sliding_window_size: {getattr(hf_model.config, 'sliding_window_size', 'N/A')}")
print(f"     sliding_window_indices: {getattr(hf_model.config, 'sliding_window_indices', 'N/A')}")

# Check if HF model has sinks
print(f"   Architecture:")
first_block = hf_model.transformer.h[0]
print(f"     First block type: {type(first_block)}")
print(f"     Has sinks: {hasattr(first_block, 'sinks')}")
if hasattr(first_block, 'sinks'):
    print(f"     Sinks shape: {first_block.sinks.shape}")
    print(f"     Sinks mean: {first_block.sinks.mean().item():.6f}")
print(f"     Attention type: {type(first_block.attn)}")

# HF PREFILL
print(f"\n3. HF PREFILL FORWARD PASS")
with torch.no_grad():
    hf_out = hf_model(input_ids=inputs.input_ids.to("cuda"), use_cache=True)
    hf_logits = hf_out.logits[0, -1, :hf_model.config.vocab_size].cpu()
    hf_top5_tokens = hf_logits.topk(5).indices.tolist()
    hf_top5_values = hf_logits.topk(5).values.tolist()
    
print(f"   Logits shape: {hf_logits.shape}")
print(f"   Logits stats: mean={hf_logits.mean().item():.4f}, std={hf_logits.std().item():.4f}")
print(f"   Top-5 tokens: {hf_top5_tokens}")
print(f"   Top-5 values: {[f'{v:.4f}' for v in hf_top5_values]}")
print(f"   Top-5 decoded: {[tokenizer.decode([t]) for t in hf_top5_tokens]}")
print(f"   Argmax token: {hf_logits.argmax().item()} -> '{tokenizer.decode([hf_logits.argmax().item()])}'")

del hf_model
torch.cuda.empty_cache()
gc.collect()

# Load YALIS model
print(f"\n4. YALIS MODEL")
if not dist.is_initialized():
    init_distributed()

torch.cuda.empty_cache()
gc.collect()

yalis_model = get_model(
    model_id,
    model_dtype=torch.bfloat16,
    attention_backend=AttentionBackend.SDPA,
    use_paged_kv_caching=False,
    prestore_kv_cache=True,
    disable_tp=True,
).to("cuda")
yalis_model.eval()

print(f"   Config:")
print(f"     vocab_size: {yalis_model.config.vocab_size}")
print(f"     n_layer: {yalis_model.config.n_layer}")
print(f"     n_embd: {yalis_model.config.n_embd}")
print(f"     n_head: {yalis_model.config.n_head}")
print(f"     n_query_groups: {getattr(yalis_model.config, 'n_query_groups', 'N/A')}")
print(f"     sliding_window_mode: {getattr(yalis_model.config, 'sliding_window_mode', 'N/A')}")
print(f"     sliding_window_size: {getattr(yalis_model.config, 'sliding_window_size', 'N/A')}")
print(f"     sliding_window_indices: {getattr(yalis_model.config, 'sliding_window_indices', 'N/A')}")

print(f"   Architecture:")
first_block = yalis_model.transformer.h[0]
print(f"     First block type: {type(first_block)}")
print(f"     Has sinks: {hasattr(first_block, 'sinks')}")
if hasattr(first_block, 'sinks'):
    print(f"     Sinks shape: {first_block.sinks.shape}")
    print(f"     Sinks mean: {first_block.sinks.mean().item():.6f}")
print(f"     Attention type: {type(first_block.attn)}")
print(f"     Apply sliding window: {getattr(first_block.attn, 'apply_sliding_window_attention', 'N/A')}")

# Allocate KV cache
total_seq_len = inputs.input_ids.shape[1] + 20
yalis_model.set_kv_cache(max_batch_size=1, max_seq_length=total_seq_len, device=torch.device("cuda"))

print(f"\n5. YALIS PREFILL FORWARD PASS")
with torch.no_grad():
    token_ids = inputs.input_ids.to("cuda")
    print(f"   token_counter before: {yalis_model.token_counter[0].item()}")
    yalis_out = yalis_model(token_ids, phase=EnginePhase.PREFILL)
    print(f"   token_counter after: {yalis_model.token_counter[0].item()}")
    
    yalis_logits = yalis_out["logits"][0, -1, :yalis_model.config.vocab_size].cpu()
    yalis_top5_tokens = yalis_logits.topk(5).indices.tolist()
    yalis_top5_values = yalis_logits.topk(5).values.tolist()
    
print(f"   Logits shape: {yalis_logits.shape}")
print(f"   Logits stats: mean={yalis_logits.mean().item():.4f}, std={yalis_logits.std().item():.4f}")
print(f"   Top-5 tokens: {yalis_top5_tokens}")
print(f"   Top-5 values: {[f'{v:.4f}' for v in yalis_top5_values]}")
print(f"   Top-5 decoded: {[tokenizer.decode([t]) for t in yalis_top5_tokens]}")
print(f"   Argmax token: {yalis_logits.argmax().item()} -> '{tokenizer.decode([yalis_logits.argmax().item()])}'")

# Check KV cache
layer0_k_cache = yalis_model.transformer.h[0].attn.kv_cache.k
prompt_len = inputs.input_ids.shape[1]
print(f"   KV cache check:")
print(f"     K cache shape: {layer0_k_cache.shape}")
print(f"     K cache filled positions mean: {layer0_k_cache[0, :, :prompt_len, :].abs().mean().item():.4f}")
print(f"     K cache unfilled positions mean: {layer0_k_cache[0, :, prompt_len:, :].abs().mean().item():.6f}")

print(f"\n6. LOGITS COMPARISON (PREFILL)")
l2_dist = torch.norm(hf_logits - yalis_logits).item()
cos_sim = torch.nn.functional.cosine_similarity(hf_logits.unsqueeze(0), yalis_logits.unsqueeze(0)).item()
top1_match = (hf_logits.argmax().item() == yalis_logits.argmax().item())
top5_overlap = len(set(hf_top5_tokens) & set(yalis_top5_tokens))

print(f"   L2 distance: {l2_dist:.4f}")
print(f"   Cosine similarity: {cos_sim:.6f}")
print(f"   Top-1 match: {top1_match}")
print(f"   Top-5 overlap: {top5_overlap}/5")

if l2_dist > 10.0 or not top1_match:
    print(f"\n   ⚠️  SIGNIFICANT DIVERGENCE DETECTED!")
    print(f"   This suggests architectural or implementation differences.")
    print(f"   Possible causes:")
    print(f"     - Different attention implementations (sinks handling)")
    print(f"     - Sliding window applied differently")
    print(f"     - RoPE configuration mismatch")
    print(f"     - Weights loaded incorrectly")

# Test DECODE_SINGLE
print(f"\n7. YALIS DECODE_SINGLE TEST")
first_token = yalis_logits.argmax().item()
with torch.no_grad():
    next_input = torch.tensor([[first_token]], dtype=torch.long, device="cuda")
    print(f"   Input token: {first_token} -> '{tokenizer.decode([first_token])}'")
    print(f"   token_counter before: {yalis_model.token_counter[0].item()}")
    
    decode_out = yalis_model(next_input, phase=EnginePhase.DECODE_SINGLE)
    print(f"   token_counter after: {yalis_model.token_counter[0].item()}")
    
    decode_logits = decode_out["logits"][0, -1, :yalis_model.config.vocab_size].cpu()
    decode_top5 = decode_logits.topk(5).indices.tolist()
    
print(f"   Output logits stats: mean={decode_logits.mean().item():.4f}, std={decode_logits.std().item():.4f}")
print(f"   Top-5 tokens: {decode_top5}")
print(f"   Top-5 decoded: {[tokenizer.decode([t]) for t in decode_top5]}")
print(f"   Argmax: {decode_logits.argmax().item()} -> '{tokenizer.decode([decode_logits.argmax().item()])}'")

# Check if cache was updated
current_pos = yalis_model.token_counter[0].item()
k_at_new_pos = layer0_k_cache[0, :, current_pos-1, :].abs().mean().item()
print(f"   K cache at position {current_pos-1}: mean={k_at_new_pos:.4f}")

del yalis_model
torch.cuda.empty_cache()

print(f"\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

if dist.is_initialized():
    dist.destroy_process_group()
