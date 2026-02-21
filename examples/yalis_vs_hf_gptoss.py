#!/usr/bin/env python3
"""
Compare HuggingFace vs YALIS logits for GPT-OSS-20B.
Tests generation quality and consistency across both implementations.
"""

import gc
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from yalis.model import get_model
from yalis.initialize import init_distributed
from yalis.constants import EnginePhase
from yalis.attention.backends import AttentionBackend


def sample_token(logits, temperature=0.0, top_p=0.9):
    """
    Sample a token from logits using temperature and top-p.

    If `temperature` is 0 or None, uses greedy sampling.
    """
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits).item())
    # Apply temperature scaling
    if temperature != 1.0:
        logits = logits / float(temperature)
    # Top-p (nucleus) sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs_sorted = torch.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs_sorted, dim=-1)
    sorted_indices_to_remove = cumsum > float(top_p)
    # Keep at least the top token
    sorted_indices_to_remove[..., 0] = False
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    probs = torch.softmax(sorted_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    next_token = sorted_indices[choice]
    return int(next_token.item())


# Local path to GPT-OSS-20B checkpoint
model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

# Initialize Tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, trust_remote_code=True, local_files_only=True
)

prompts_to_test = [
    "The capital of France is",
    "How to bake a cake?",
    "Explain quantum entanglement like I'm five.",
]

for prompt_idx, raw_prompt in enumerate(prompts_to_test):
    print(f"\n\n{'='*80}")
    print(f"TESTING PROMPT {prompt_idx}: {repr(raw_prompt)}")
    print(f"{'='*80}")

    # Tokenize raw prompt
    inputs = tokenizer(raw_prompt, return_tensors="pt")
    print(f"Prompt tokens shape: {inputs.input_ids.shape}")
    print(f"Prompt length: {inputs.input_ids.shape[1]} tokens")
    # Shared sampling temperature for HF and YALIS
    sample_temperature = 0.0

    # ============================================================================
    # PHASE 1: 20-Token Generation (HuggingFace)
    # ============================================================================
    print("\nPHASE 1: 20-Token Generation (HuggingFace)")
    print("-" * 40)

    # Only load HF model on rank 0 to avoid OOM when running on multiple GPUs
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    hf_generated_text = None

    if local_rank == 0:
        # Aggressive cleanup before loading HF
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        hf_model.eval()
        # Use HF model's vocab size when needed
        hf_vocab_size = getattr(hf_model.config, "vocab_size", None)

        hf_generated_ids = inputs.input_ids.clone()
        with torch.no_grad():
            # HF PREFILL (compute once, return past_key_values)
            hf_prefill = hf_model(
                input_ids=inputs.input_ids.to("cuda"), use_cache=True
            )
            hf_prefill_logits = hf_prefill.logits[0, -1, :hf_vocab_size].cpu()
            next_token = sample_token(
                hf_prefill_logits, temperature=sample_temperature, top_p=0.9
            )
            hf_generated_ids = torch.cat(
                [hf_generated_ids, torch.tensor([[next_token]])], dim=1
            )
            past = hf_prefill.past_key_values
            torch.cuda.synchronize()

            # HF incremental decoding using cached past_key_values to
            # match YALIS DECODE_SINGLE
            for gen_step in range(19):
                next_input = torch.tensor([[next_token]], device="cuda")
                hf_out = hf_model(
                    input_ids=next_input, past_key_values=past, use_cache=True
                )
                hf_logits = hf_out.logits[0, -1, :hf_vocab_size].cpu()
                next_token = sample_token(
                    hf_logits, temperature=sample_temperature, top_p=0.9
                )
                hf_generated_ids = torch.cat(
                    [hf_generated_ids, torch.tensor([[next_token]])], dim=1
                )
                past = hf_out.past_key_values
                if gen_step % 5 == 4:
                    torch.cuda.synchronize()

        hf_generated_text = tokenizer.decode(
            hf_generated_ids[0], skip_special_tokens=False
        )
        print(f"HF Generated (20 tokens): {hf_generated_text}")

        del hf_model
        torch.cuda.empty_cache()
        gc.collect()
    else:
        print("(Rank 1: Skipping HF inference, only running YALIS)")

    # ============================================================================
    # PHASE 2: 20-Token Generation (YALIS)
    # ============================================================================
    print("\nPHASE 2: 20-Token Generation (YALIS)")
    print("-" * 40)

    # Ensure distributed and Axonn are initialized so ax.comm_handle is set.
    if not dist.is_initialized():
        init_distributed()

    # Aggressive cleanup before reloading YALIS
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    # Reinitialize YALIS model for generation
    yalis_model_gen = get_model(
        model_id,
        model_dtype=torch.bfloat16,
        attention_backend=AttentionBackend.SDPA,
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
    ).to("cuda")
    yalis_model_gen.eval()
    # Use YALIS model's vocab size for slicing logits
    actual_vocab_size = getattr(yalis_model_gen.config, "vocab_size", None)

    # Check if sinks are loaded (not zeros)
    first_block_sinks = yalis_model_gen.transformer.h[0].sinks
    sinks_mean = first_block_sinks.mean().item()
    sinks_std = first_block_sinks.std().item()
    print(
        "[DEBUG] Layer 0 sinks: shape=%s, mean=%.6f, std=%.6f"
        % (first_block_sinks.shape, sinks_mean, sinks_std)
    )

    # Check sliding window configuration per layer
    print(
        "[DEBUG] Config sliding_window_indices:",
        yalis_model_gen.config.sliding_window_indices,
    )
    print(
        "[DEBUG] Config sliding_window_size:",
        yalis_model_gen.config.sliding_window_size,
    )
    print("[DEBUG] Sliding window pattern (first 8 layers):")
    for i in range(min(8, len(yalis_model_gen.transformer.h))):
        block = yalis_model_gen.transformer.h[i]
        sw_size = (
            yalis_model_gen.config.sliding_window_size
            if block.attn.apply_sliding_window_attention
            else 0
        )
        print(
            (
                "  Layer %d: sliding_window=%d, sinks_mean=%.3f, apply_sw=%s"
                % (
                    i,
                    sw_size,
                    block.sinks.mean().item(),
                    block.attn.apply_sliding_window_attention,
                )
            )
        )

    # Allocate KV cache (with buffer for 20 new tokens)
    total_seq_len = inputs.input_ids.shape[1] + 20
    yalis_model_gen.set_kv_cache(
        max_batch_size=1,
        max_seq_length=total_seq_len,
        device=torch.device("cuda"),
    )

    # DIAGNOSTIC: Check token_counter after set_kv_cache
    print(
        "[DEBUG] token_counter after set_kv_cache:",
        yalis_model_gen.token_counter[:1],
    )

    yalis_generated_ids = inputs.input_ids.clone()
    generated_tokens = []  # Accumulate tokens instead of repeated torch.cat

    with torch.no_grad():
        # PREFILL initial prompt
        token_ids = inputs.input_ids.to("cuda")
        print(
            "[DEBUG] Before PREFILL: token_counter=",
            yalis_model_gen.token_counter[:1],
        )
        prefill_out = yalis_model_gen(token_ids, phase=EnginePhase.PREFILL)
        print(
            "[DEBUG] After PREFILL: token_counter=",
            yalis_model_gen.token_counter[:1],
        )

        # Check KV cache after prefill (layer 0)
        layer0_k_cache = yalis_model_gen.transformer.h[0].attn.kv_cache.k
        prompt_len = inputs.input_ids.shape[1]
        k_mean = layer0_k_cache[0, :, :prompt_len, :].abs().mean().item()
        print(
            "[DEBUG] Layer 0 K cache after PREFILL: shape=",
            layer0_k_cache.shape,
            ", prompt_len=",
            prompt_len,
            ", mean=%.4f" % k_mean,
        )

        # Sample first token from PREFILL output
        first_logits = prefill_out["logits"][0, -1, :actual_vocab_size].cpu()
        next_token = int(first_logits.argmax().item())
        print(
            "[DEBUG] First sampled token:",
            next_token,
            "->",
            tokenizer.decode([next_token]),
        )
        fl_mean = first_logits.mean().item()
        fl_std = first_logits.std().item()
        fl_max = first_logits.max().item()
        print(
            "[DEBUG] PREFILL logits stats: mean=%.4f, std=%.4f, max=%.4f"
            % (fl_mean, fl_std, fl_max)
        )
        top5 = first_logits.topk(5).indices.tolist()
        print("[DEBUG] PREFILL top-5 tokens:", top5)
        decoded_top5 = [tokenizer.decode([t]) for t in top5]
        print("[DEBUG] PREFILL top-5 decoded:", decoded_top5)
        generated_tokens.append(next_token)
        torch.cuda.synchronize()

        # DECODE_SINGLE loop for remaining 19 tokens
        for gen_step in range(19):
            # Create input token on same device as prompt to
            # avoid device mismatch
            current_token = torch.tensor(
                [[next_token]], dtype=torch.long, device=token_ids.device
            )
            print(
                "[DEBUG] Step",
                gen_step,
                "token_counter=",
                yalis_model_gen.token_counter[:1],
                "input_token=",
                next_token,
            )
            yalis_out_gen = yalis_model_gen(
                current_token, phase=EnginePhase.DECODE_SINGLE
            )
            yalis_logits_gen = yalis_out_gen["logits"][
                0, -1, :actual_vocab_size
            ].cpu()
            next_token = sample_token(
                yalis_logits_gen, temperature=sample_temperature, top_p=0.9
            )
            print(
                "[DEBUG] Step",
                gen_step,
                "sampled token=",
                next_token,
                "->",
                tokenizer.decode([next_token]),
            )
            if gen_step < 3:  # Detailed stats for first few steps
                lg_mean = yalis_logits_gen.mean().item()
                lg_std = yalis_logits_gen.std().item()
                lg_max = yalis_logits_gen.max().item()
                print(
                    "[DEBUG]   Logits stats: mean=%.4f, std=%.4f, max=%.4f"
                    % (lg_mean, lg_std, lg_max)
                )
                top5_gen = yalis_logits_gen.topk(5).indices.tolist()
                print("[DEBUG]   Top-5 tokens:", top5_gen)
                decoded_top5_gen = [tokenizer.decode([t]) for t in top5_gen]
                print("[DEBUG]   Top-5 decoded:", decoded_top5_gen)

            # After step 2, check if cache was updated
            if gen_step == 2:
                current_pos = yalis_model_gen.token_counter[0].item()
                k_at_pos = (
                    layer0_k_cache[0, :, current_pos - 1, :]
                    .abs()
                    .mean()
                    .item()
                )
                print(
                    "[DEBUG] After step 2: K cache at position",
                    current_pos - 1,
                    ", mean=%.4f" % k_at_pos,
                )

            generated_tokens.append(next_token)
            if gen_step % 5 == 4:
                torch.cuda.synchronize()

    # Concatenate all generated tokens once
    yalis_generated_ids = torch.cat(
        [yalis_generated_ids, torch.tensor([generated_tokens])], dim=1
    )
    yalis_generated_text = tokenizer.decode(
        yalis_generated_ids[0], skip_special_tokens=False
    )
    print(f"YALIS Generated (20 tokens): {yalis_generated_text}")

    del yalis_model_gen
    torch.cuda.empty_cache()
    gc.collect()

if dist.is_initialized():
    dist.destroy_process_group()
