import pytest
import torch
import torch.distributed as dist
import random
import numpy as np
import warnings
import gc
import logging
import os
from utils import alpaca_prompt
from yalis import ModelConfig, InferenceConfig, LLMEngine, SpeculativeLLMEngine

IGNORE_EOS = True

# Get logger instance
logger = logging.getLogger(__name__)

# Local rank for per-rank logging
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

# Test parameters
BATCH_SIZES = [2]
PROMPT_LENGTHS = [128]
TOKEN_GENERATION_LENGTHS = [4]
GAMMA_VALUES = [2, 5]

def _get_standard_output(engine, prompts, num_tokens):
    """Get output from standard LLMEngine."""
    output_tokens, _ = engine.generate(
        prompts,
        tokens_to_generate=num_tokens,
        report_throughput=False,
        ignore_eos=IGNORE_EOS,  # Ignore EOS for exact comparison
    )
    # Return list of tensors for each batch item
    return [output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))]


def _get_speculative_output(engine, prompts, num_tokens, gamma=5):
    """Get output from SpeculativeLLMEngine."""
    output_tokens, metrics = engine.generate_speculative(
        prompts,
        tokens_to_generate=num_tokens,
        gamma=gamma,
        report_throughput=False,
        ignore_eos=IGNORE_EOS,  # Ignore EOS for exact comparison
    )
    # Return list of tensors for each batch item
    return [output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))]


def _compare_token_outputs(standard_tokens, speculative_tokens, tokenizer):
    """
    Compare token outputs between standard and speculative decoding.
    Returns True if tokens match exactly, False otherwise.
    """
    assert len(standard_tokens) == len(speculative_tokens), (
        f"Batch size mismatch:"
        f"std - {len(standard_tokens)} vs spec - {len(speculative_tokens)}"
    )

    all_match = True
    mismatches = 0

    for i, (std_tokens, spec_tokens) in enumerate(
        zip(standard_tokens, speculative_tokens)
    ):
        assert len(std_tokens) == len(spec_tokens), (
            f"[Batch {i}] Token len mismatch: "
            f"std - {len(std_tokens)} vs spec - {len(spec_tokens)}"
        )
        # Compare tokens exactly
        tokens_match = torch.equal(std_tokens, spec_tokens)
        if not tokens_match:
            all_match = False
            mismatches += 1

            # Find first mismatch position
            diff_positions = (std_tokens != spec_tokens).nonzero().flatten()
            first_diff = (
                diff_positions[0].item() if len(diff_positions) > 0 else -1
            )

            warnings.warn(
                f"Batch {i}: Token mismatch at position {first_diff}\n"
                f"Standard: {std_tokens[max(0, first_diff-2):first_diff+3]}\n"
                f"Spec: {spec_tokens[max(0, first_diff-2):first_diff+3]}\n"
                f"Standard text: {tokenizer.decode(std_tokens[max(0, first_diff-2):first_diff+3], skip_special_tokens=True)}\n"  # noqa: E501
                f"Spec text: {tokenizer.decode(spec_tokens[max(0, first_diff-2):first_diff+3], skip_special_tokens=True)}"  # noqa: E501
            )

    return all_match


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.parametrize("num_tokens", TOKEN_GENERATION_LENGTHS)
@pytest.mark.parametrize("gamma", GAMMA_VALUES)
def test_speculative(
    tokenizer,
    batch_size,
    prompt_length,
    num_tokens,
    alpaca_dataset,
    gamma,
    attn_backend,
    dtype,
    model_id,
    draft_model_id,
):
    """
    Test that speculative decoding produces exactly the same tokens
    as standard decoding with greedy decoding.
    """

    if attn_backend.yalis != "sdpa":
        pytest.skip(
            "Non-SDPA attention backends do not uphold greedy equality"
        )

    # Ensure consistent random sampling across all ranks for TP tests
    # All ranks must generate identical prompts for distributed inference
    random_seed = 42
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate test prompts
    prompts = alpaca_prompt(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )

    # Resolve model paths
    if not os.path.isabs(model_id):
        target_model_path = os.path.abspath(model_id)
    else:
        target_model_path = model_id
    
    if not os.path.isabs(draft_model_id):
        draft_model_path = os.path.abspath(draft_model_id)
    else:
        draft_model_path = draft_model_id
    
    target_model_name = os.path.basename(target_model_path)
    draft_model_name = os.path.basename(draft_model_path)
    
    # Create model configs
    target_model_config = ModelConfig(
        target_model_name, model_path=target_model_path, precision=dtype.yalis
    )
    draft_model_config = ModelConfig(
        draft_model_name, model_path=draft_model_path, precision=dtype.yalis
    )
    inference_config = InferenceConfig(
        max_batch_size=batch_size,
        max_length_of_generated_sequences=2048,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend=attn_backend.yalis,
        use_paged_kv_caching=False,
    )

    # Load and run standard inference
    logger.info(
        f"[rank {LOCAL_RANK}] Loading standard LLMEngine for inference..."
    )
    standard_engine = LLMEngine(
        model_config=target_model_config, inference_config=inference_config
    )
    
    # Synchronize after engine initialization
    if dist.is_initialized():
        dist.barrier()
    
    logger.info(
        f"[rank {LOCAL_RANK}] Running standard inference with {num_tokens} tokens..."
    )
    output_tokens, _ = standard_engine.generate(
        prompts,
        tokens_to_generate=num_tokens,
        report_throughput=False,
        ignore_eos=IGNORE_EOS,
    )
    standard_tokens = [
        output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))
    ]
    
    # Synchronize after inference before cleanup
    if dist.is_initialized():
        dist.barrier()
    
    # Clean up standard engine to free GPU memory
    del standard_engine
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"[rank {LOCAL_RANK}] Cleaned up standard engine")
    
    # Synchronize after cleanup before loading next engine
    if dist.is_initialized():
        dist.barrier()

    # Load and run speculative inference
    logger.info(
        f"[rank {LOCAL_RANK}] Loading SpeculativeLLMEngine for inference..."
    )
    speculative_engine = SpeculativeLLMEngine(
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        inference_config=inference_config,
    )
    
    # Synchronize after engine initialization
    if dist.is_initialized():
        dist.barrier()
    
    logger.info(
        f"[rank {LOCAL_RANK}] Running speculative inference with {num_tokens} tokens and gamma={gamma}..."
    )
    # Tokenize prompts to match the type signature of speculative generate
    tokenized_prompts = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"]
    output_tokens, metrics = speculative_engine.generate_speculative(
        tokenized_prompts,
        tokens_to_generate=num_tokens,
        gamma=gamma,
        report_throughput=False,
        ignore_eos=IGNORE_EOS,
    )
    speculative_tokens = [
        output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))
    ]
    
    # Synchronize after inference before cleanup
    if dist.is_initialized():
        dist.barrier()
    
    # Clean up speculative engine
    del speculative_engine
    torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"[rank {LOCAL_RANK}] Cleaned up speculative engine")

    # Synchronize after cleanup
    if dist.is_initialized():
        dist.barrier()

    # Compare outputs - they should match exactly
    logger.info(f"[rank {LOCAL_RANK}] Comparing token outputs...")
    tokens_match = _compare_token_outputs(
        standard_tokens, speculative_tokens, tokenizer
    )

    assert tokens_match, (
        f"Speculative decoding tokens do not match standard decoding tokens "
        f"for batch_size={batch_size}, prompt_length={prompt_length}, "
        f"num_tokens={num_tokens}, gamma={gamma}"
    )
