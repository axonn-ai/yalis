import pytest
import torch
import random
import numpy as np
import warnings
from utils import alpaca_prompt

IGNORE_EOS = True

# Test parameters
BATCH_SIZES = [1, 4, 8]
PROMPT_LENGTHS = [128, 1024]
TOKEN_GENERATION_LENGTHS = [128, 512]
GAMMA_VALUES = [2, 5]


def _get_standard_output(engine, prompts, num_tokens):
    """Get output from standard LLMEngine."""
    # Set deterministic seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

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
    # Set deterministic seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

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
            first_diff = diff_positions[0].item() if len(diff_positions) > 0 else -1

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
    speculative_engine,
    batch_size,
    prompt_length,
    num_tokens,
    alpaca_dataset,
    gamma,
    attn_backend,
):
    """
    Test that speculative decoding produces exactly the same tokens
    as standard decoding with greedy decoding.
    """

    if attn_backend.yalis != "sdpa":
        pytest.skip("Non-SDPA attention backends do not uphold greedy equality")

    # Generate test prompts
    prompts = alpaca_prompt(alpaca_dataset, tokenizer, prompt_length, batch_size)

    # Get outputs from both engines
    standard_tokens = _get_standard_output(speculative_engine, prompts, num_tokens)
    speculative_tokens = _get_speculative_output(
        speculative_engine, prompts, num_tokens, gamma=gamma
    )

    # Compare outputs - they should match exactly
    tokens_match = _compare_token_outputs(
        standard_tokens, speculative_tokens, tokenizer
    )

    assert tokens_match, (
        f"Speculative decoding tokens do not match standard decoding tokens "
        f"for batch_size={batch_size}, prompt_length={prompt_length}, "
        f"num_tokens={num_tokens}, gamma={gamma}"
    )
