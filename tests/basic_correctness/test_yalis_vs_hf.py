import pytest
import torch
import torch.distributed as dist
import warnings
import gc
import logging
from utils import alpaca_prompt
from transformers import StoppingCriteriaList, StoppingCriteria
import random as random_module

# Configure logger
logger = logging.getLogger(__name__)

NUM_LOGPROBS = 5
# BATCH_SIZES = [1, 4, 8]
BATCH_SIZES = [1]
# PROMPT_LENGTHS = [128, 256, 512, 1024]
PROMPT_LENGTHS = [64]

class NeverStop(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # Return a tensor of False for each element in batch
        batch_size = input_ids.shape[0]
        return torch.tensor([False] * batch_size, dtype=torch.bool)


def _generate_and_broadcast_prompts(
    alpaca_dataset, tokenizer, prompt_length, batch_size
):
    """
    Generate prompts on rank 0 and broadcast to all ranks.

    This ensures all ranks in a distributed setting use identical prompts,
    which is critical for tensor-parallel inference where all replicas must
    process the same input tensors.

    Args:
        alpaca_dataset: AlpacaDataset instance
        tokenizer: HF tokenizer
        prompt_length: Target prompt length in tokens
        batch_size: Batch size

    Returns:
        List of prompt strings (identical on all ranks)
    """
    rank = 0
    world_size = 1

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    if rank == 0:
        random_module.seed(42)
        prompts = alpaca_prompt(
            alpaca_dataset, tokenizer, prompt_length, batch_size
        )
    else:
        prompts = None

    # Broadcast prompts from rank 0 to all other ranks
    if world_size > 1:
        prompts_list = [prompts]
        dist.broadcast_object_list(prompts_list, src=0)
        prompts = prompts_list[0]

    return prompts


def _get_logprobs(logits):
    # logits: list of [batch_size, vocab_size] tensors of length num_tokens
    num_tokens = len(logits)
    batch_size = logits[0].shape[0]

    logprob_list: list[torch.Tensor] = []
    topk_list: list[torch.Tensor] = []
    for logit in logits:
        logprobs = torch.log_softmax(logit, dim=-1, dtype=torch.float32)
        logprob_list.append(logprobs)

        topk_indices = torch.argsort(
            logprobs, dim=-1, descending=True, stable=True
        )[:, :NUM_LOGPROBS]
        topk_list.append(topk_indices)

    # We need to convert this to a list of [num_tokens, -1] shaped
    # tensors of length batch_size
    final_logprob_list = []
    final_topk_list = []
    for i in range(batch_size):
        per_prompt_logprobs = []
        per_prompt_topk = []
        for j in range(num_tokens):
            per_prompt_logprobs.append(logprob_list[j][i, :].cpu())
            per_prompt_topk.append(topk_list[j][i, :].cpu())

        per_prompt_logprobs = torch.stack(per_prompt_logprobs)
        per_prompt_topk = torch.stack(per_prompt_topk)

        final_logprob_list.append(per_prompt_logprobs)
        final_topk_list.append(per_prompt_topk)

    return final_logprob_list, final_topk_list


def _get_hf_output(tokenizer, model, prompts, num_tokens):
    # In distributed mode, only rank 0 has the HF model loaded
    if model is None:
        return None, None

    # For device_map="auto", use "cuda:0" for inputs
    # - HF handles cross-GPU movement
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=None,
            stopping_criteria=StoppingCriteriaList([NeverStop()]),
            temperature=0.0,
            top_p=0.0,
            output_logits=True,
            output_hidden_states=False,
            return_dict_in_generate=True,
        )

    # For batch, return list of new tokens for each prompt
    new_tokens = []
    for i in range(len(prompts)):
        input_len = inputs["input_ids"][i].shape[0]
        new_tokens.append(
            output.sequences[i][input_len : input_len + num_tokens].cpu()
        )

    # new_tokens: list of [num_tokens] tensors of length batch_size
    # output.logits: list of [batch_size, vocab_size] tensors of length num_tokens  # noqa: E501
    return new_tokens, output.logits


def _get_yalis_output(engine, prompts, num_tokens):
    output_tokens, _, logits = engine.generate(
        prompts,
        report_throughput=False,
        tokens_to_generate=num_tokens,
        get_logits=True,
    )
    # output_tokens: (batch, num_tokens)
    return [
        output_tokens[i][:num_tokens].cpu() for i in range(len(prompts))
    ], logits


# This test does not mean a failure
def _compare_tokens_and_text(tokenizer, tokens1, tokens2):
    token_mismatches = 0
    text_mismatches = 0
    assert len(tokens1) == len(
        tokens2
    ), f"Batch size mismatch: {len(tokens1)} vs {len(tokens2)}"
    for t1, t2 in zip(tokens1, tokens2):
        assert len(t1) == len(
            t2
        ), f"Token length mismatch: {len(t1)} vs {len(t2)}"
        num_matches = sum(a == b for a, b in zip(t1, t2))
        if num_matches < len(t1) - 1:
            warnings.warn(f"Token mismatch: {t1} vs {t2}")
            token_mismatches += 1
        text1 = tokenizer.decode(t1, skip_special_tokens=True)
        text2 = tokenizer.decode(t2, skip_special_tokens=True)
        if text1.strip()[:10] != text2.strip()[:10]:
            warnings.warn(f"Text mismatch: {text1} vs {text2}")
            text_mismatches += 1

    return token_mismatches == 0 and text_mismatches == 0


def _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens):
    hf_logprobs, hf_topk = _get_logprobs(hf_logits)
    yalis_logprobs, yalis_topk = _get_logprobs(yalis_logits)

    assert len(hf_logprobs) == len(
        yalis_logprobs
    ), f"Batch size mismatch: {len(hf_logprobs)} vs {len(yalis_logprobs)}"
    assert len(hf_topk) == len(
        hf_tokens
    ), f"HF tokens and topk length mismatch: {len(hf_topk)} vs {len(hf_tokens)}"  # noqa: E501
    assert len(yalis_topk) == len(
        yalis_tokens
    ), f"Yalis tokens and topk length mismatch: {len(yalis_topk)} vs {len(yalis_tokens)}"  # noqa: E501
    assert len(hf_logprobs) == len(
        hf_tokens
    ), f"HF logprobs and tokens length mismatch: {len(hf_logprobs)} vs {len(hf_tokens)}"  # noqa: E501

    for (
        hf_token,
        yalis_token,
        hf_logprob,
        yalis_logprob,
        hf_topk,
        yalis_topk,
    ) in zip(
        hf_tokens,
        yalis_tokens,
        hf_logprobs,
        yalis_logprobs,
        hf_topk,
        yalis_topk,
    ):
        assert (
            hf_token.shape == yalis_token.shape
        ), f"Token shape mismatch: {hf_token.shape} vs {yalis_token.shape}"

        for i in range(len(hf_token)):
            hf_token_i = hf_token[i]
            yalis_token_i = yalis_token[i]

            token_mismatch = hf_token_i != yalis_token_i

            if token_mismatch:
                warnings.warn(
                    f"Token mismatch {i}: HF Token: {hf_token_i} vs Yalis Token: {yalis_token_i}"  # noqa: E501
                )
                # Check if the tokens are in the top NUM_LOGPROBS of each other
                hf_token_i_in_topk = hf_token_i in yalis_topk[i]
                yalis_token_i_in_topk = yalis_token_i in hf_topk[i]

                assert (
                    hf_token_i_in_topk
                ), f"HF token {hf_token_i} not in Yalis topk {yalis_topk[i]}, {i}: HF topk {len(hf_topk)} - {hf_logprob[i]}, {yalis_logprob[i]}"  # noqa: E501
                assert (
                    yalis_token_i_in_topk
                ), f"Yalis token {yalis_token_i} not in HF topk {hf_topk[i]}, {i}: Yalis topk {len(yalis_topk)} - {yalis_logprob[i]}, {hf_logprob[i]}"  # noqa: E501

                # Now the tokens will diverge, so need to break
                break


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.filterwarnings("ignore:.*do_sample.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*co_lnotab.*:DeprecationWarning")
def test_01_prefill(
    tokenizer,
    yalis_engine,
    hf_model,
    batch_size,
    prompt_length,
    dtype,
    alpaca_dataset,
):
    # Generate prompts on rank 0 and broadcast to all ranks to ensure
    # identical inputs for tensor-parallel inference
    prompts = _generate_and_broadcast_prompts(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )
    
    # Synchronize all ranks BEFORE HF inference to prevent timing mismatch:
    # HF only runs on rank 0 (asymmetric operation), but all ranks must
    # coordinate before this. Rank 1 finishes quickly (returns None) and must
    # not reach TP collectives before rank 0 completes HF inference.
    if dist.is_initialized():
        rank = dist.get_rank()
        logger.info(f"[Rank {rank}] Synchronizing ranks before HF inference")
        dist.barrier()
        logger.info(f"[Rank {rank}] Synchronization complete, starting HF inference")
    
    hf_tokens, hf_logits = _get_hf_output(
        tokenizer,
        hf_model,
        prompts,
        num_tokens=1,
    )
    logger.info("HF inference complete, cleaning up HF model")
    # Garbage collect HF model before YALIS inference
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("HF model garbage collected")
    
    # Synchronize all ranks before YALIS inference to ensure both are ready
    # for collective operations
    if dist.is_initialized():
        rank = dist.get_rank()
        logger.info(f"[Rank {rank}] Synchronizing ranks before YALIS inference")
        dist.barrier()
        logger.info(f"[Rank {rank}] Synchronization complete, starting YALIS inference")
    
    yalis_tokens, yalis_logits = _get_yalis_output(
        yalis_engine, prompts, num_tokens=1
    )
    logger.info("YALIS inference complete, cleaning up engine")
    # Garbage collect YALIS before comparison
    del yalis_engine
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("YALIS engine garbage collected")
    
    # Only compare on rank 0 where HF model is loaded
    if hf_tokens is not None:
        _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.filterwarnings("ignore:.*do_sample.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*co_lnotab.*:DeprecationWarning")
def test_02_decode(
    tokenizer,
    yalis_engine,
    hf_model,
    batch_size,
    prompt_length,
    dtype,
    alpaca_dataset,
):
    # Generate prompts on rank 0 and broadcast to all ranks to ensure
    # identical inputs for tensor-parallel inference
    prompts = _generate_and_broadcast_prompts(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )
    
    # Synchronize all ranks BEFORE HF inference to prevent timing mismatch:
    # HF only runs on rank 0 (asymmetric operation), but all ranks must
    # coordinate before this. Rank 1 finishes quickly (returns None) and must
    # not reach TP collectives before rank 0 completes HF inference.
    if dist.is_initialized():
        rank = dist.get_rank()
        logger.info(f"[Rank {rank}] Synchronizing ranks before HF inference")
        dist.barrier()
        logger.info(f"[Rank {rank}] Synchronization complete, starting HF inference")
    
    hf_tokens, hf_logits = _get_hf_output(
        tokenizer, hf_model, prompts, num_tokens=32
    )
    logger.info("HF inference complete, cleaning up HF model")
    # Garbage collect HF model before YALIS inference
    del hf_model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("HF model garbage collected")
    
    # Synchronize all ranks before YALIS inference to ensure both are ready
    # for collective operations
    if dist.is_initialized():
        rank = dist.get_rank()
        logger.info(f"[Rank {rank}] Synchronizing ranks before YALIS inference")
        dist.barrier()
        logger.info(f"[Rank {rank}] Synchronization complete, starting YALIS inference")
    
    yalis_tokens, yalis_logits = _get_yalis_output(
        yalis_engine, prompts, num_tokens=32
    )
    logger.info("YALIS inference complete, cleaning up engine")
    # Garbage collect YALIS before comparison
    del yalis_engine
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("YALIS engine garbage collected")
    
    # Only compare on rank 0 where HF model is loaded
    if hf_tokens is not None:
        _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)


# TODO: Add perplexity test
