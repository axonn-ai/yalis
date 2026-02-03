import pytest
import torch
import os
import warnings
import gc
import logging
from utils import alpaca_prompt
import torch.distributed as dist
from transformers import StoppingCriteriaList, StoppingCriteria

# Get logger instance (logging configured in conftest.py)
logger = logging.getLogger(__name__)

NUM_LOGPROBS = 5
BATCH_SIZES = [1, 2]
PROMPT_LENGTHS = [128]

# Local rank for per-rank logging
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def log_gpu_memory(phase: str):
    """Log current GPU memory usage.
    
    Args:
        phase: Descriptive name of the current phase (e.g., "After HF inference")
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        logger.info(
            f"[rank {LOCAL_RANK}] [GPU Memory] {phase:.<50} "
            f"Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | "
            f"Peak: {max_allocated:.2f}GB"
        )

class NeverStop(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return False

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
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
        model.device
    )
    with torch.no_grad(), torch.autocast(
        "cuda", dtype=torch.float16, cache_enabled=False
    ):
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
            output_hidden_states=True,
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
    hf_model_loader,
    yalis_engine_loader,
    batch_size,
    prompt_length,
    dtype,
    alpaca_dataset,
):
    logger.info(
        f"[rank {LOCAL_RANK}] Starting test_01_prefill with batch_size={batch_size}, "
        f"prompt_length={prompt_length}"
    )
    log_gpu_memory("Test start")

    prompts = alpaca_prompt(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )

    # Load and run HF inference
    logger.info(f"[rank {LOCAL_RANK}] Loading HF model...")
    hf_model = hf_model_loader()
    log_gpu_memory("After HF load")

    logger.info(f"[rank {LOCAL_RANK}] Starting HF model inference...")
    if hf_model is not None:
        hf_tokens, hf_logits = _get_hf_output(
            tokenizer, hf_model, prompts, num_tokens=1
        )
        log_gpu_memory("After HF inference")
        
        # Move HF logits to CPU immediately and cleanup GPU memory
        logger.info(f"[rank {LOCAL_RANK}] Moving HF logits to CPU and clearing GPU memory...")
        hf_logits = [logit.cpu() for logit in hf_logits]
        del hf_model
    else:
        # Rank 1: Dummy outputs for compatibility
        hf_tokens = [torch.zeros(1, dtype=torch.long) for _ in prompts]
        hf_logits = [torch.zeros(1, 50257) for _ in range(1)]  # num_tokens=1
        logger.info("(Rank 1: Using dummy HF outputs)")
    
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu_memory("After HF cleanup")

    # Load and run YALIS inference
    logger.info(f"[rank {LOCAL_RANK}] Loading YALIS engine...")
    yalis_engine = yalis_engine_loader()
    log_gpu_memory("After YALIS load")
    
    # Synchronize all ranks before starting inference to avoid collective op mismatches
    if dist.is_initialized():
        dist.barrier()
        if LOCAL_RANK == 0:
            logger.info("All ranks synchronized, starting YALIS inference...")

    logger.info(f"[rank {LOCAL_RANK}] Starting YALIS engine inference...")
    yalis_tokens, yalis_logits = _get_yalis_output(
        yalis_engine, prompts, num_tokens=1
    )
    log_gpu_memory("After YALIS inference")


    logger.info(f"[rank {LOCAL_RANK}] Comparing logprobs...")
    _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)
    log_gpu_memory("After comparison")
    logger.info(f"[rank {LOCAL_RANK}] test_01_prefill completed successfully")


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("prompt_length", PROMPT_LENGTHS)
@pytest.mark.filterwarnings("ignore:.*do_sample.*:UserWarning")
@pytest.mark.filterwarnings("ignore:.*co_lnotab.*:DeprecationWarning")
def test_02_decode(
    tokenizer,
    hf_model_loader,
    yalis_engine_loader,
    batch_size,
    prompt_length,
    dtype,
    alpaca_dataset,
):
    logger.info(
        f"[rank {LOCAL_RANK}] Starting test_02_decode with batch_size={batch_size}, "
        f"prompt_length={prompt_length}"
    )
    log_gpu_memory("Test start")

    prompts = alpaca_prompt(
        alpaca_dataset, tokenizer, prompt_length, batch_size
    )

    # Load and run HF inference
    logger.info(f"[rank {LOCAL_RANK}] Loading HF model...")
    hf_model = hf_model_loader()
    log_gpu_memory("After HF load")

    logger.info(f"[rank {LOCAL_RANK}] Starting HF model inference...")
    if hf_model is not None:
        hf_tokens, hf_logits = _get_hf_output(
            tokenizer, hf_model, prompts, num_tokens=32
        )
        log_gpu_memory("After HF inference")
        
        # Move HF logits to CPU immediately and cleanup GPU memory
        logger.info(f"[rank {LOCAL_RANK}] Moving HF logits to CPU and clearing GPU memory...")
        hf_logits = [logit.cpu() for logit in hf_logits]
        del hf_model
    else:
        # Rank 1: Dummy outputs for compatibility
        hf_tokens = [torch.zeros(1, dtype=torch.long) for _ in prompts]
        hf_logits = [torch.zeros(1, 50257) for _ in range(32)]  # num_tokens=32
        logger.info("(Rank 1: Using dummy HF outputs)")
    
    torch.cuda.empty_cache()
    gc.collect()
    log_gpu_memory("After HF cleanup")
    
    # Load and run YALIS inference
    logger.info(f"[rank {LOCAL_RANK}] Loading YALIS engine...")
    yalis_engine = yalis_engine_loader()
    log_gpu_memory("After YALIS load")
    
    # Synchronize all ranks before starting inference to avoid collective op mismatches
    if dist.is_initialized():
        dist.barrier()
        if LOCAL_RANK == 0:
            logger.info("All ranks synchronized, starting YALIS inference...")

    logger.info(f"[rank {LOCAL_RANK}] Starting YALIS engine inference...")
    yalis_tokens, yalis_logits = _get_yalis_output(
        yalis_engine, prompts, num_tokens=32
    )
    log_gpu_memory("After YALIS inference")


    logger.info(f"[rank {LOCAL_RANK}] Comparing logprobs...")
    _compare_logprobs(hf_logits, hf_tokens, yalis_logits, yalis_tokens)
    log_gpu_memory("After comparison")
    logger.info(f"[rank {LOCAL_RANK}] test_02_decode completed successfully")


# TODO: Add perplexity test