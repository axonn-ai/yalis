from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import numpy as np
from torch.profiler import _KinetoProfile
import os
from vllm.inputs.data import TokensPrompt
import gc
import time
from typing import List
import random

_KinetoProfile._get_distributed_info = lambda self: None


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None
enable_profiling = False

if __name__ == "__main__":
    target_model_id = "meta-llama/Llama-3.1-70B-Instruct"
    draft_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    # Initialize prompt and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    user_prompts = [
        "How to bake a cake?",
        "How to drive a car on a freeway?",
        "What are the best practices for time management?",
        "Explain quantum mechanics in simple terms.",
        "How do I write a great resume for a software engineer role?",
        "What are the steps to start a successful business?",
        "How can I improve my public speaking skills?",
        "What are the benefits of a balanced diet?",
        "How to train a dog to follow basic commands?",
        "What is the process for applying to graduate school in the US?",
        "How do I troubleshoot a slow internet connection?",
        "What is the meaning of life according to philosophy?",
        "How can I learn to play the guitar?",
        "What are the key elements of a good story?",
        "How do I stay motivated while working from home?",
        "What is the easiest way to learn a new language?",
    ]

    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },  # not needed for gemma
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    sampling_params = SamplingParams(
        temperature=0.0, min_tokens=256, max_tokens=256, ignore_eos=True
    )

    # Non-speculative
    # llm = LLM(
    #     model=target_model_id,
    #     tensor_parallel_size=4,
    # )

    # Speculative
    llm = LLM(
        model=target_model_id,
        speculative_model=draft_model_id,
        tensor_parallel_size=4,
        num_speculative_tokens=5,
        max_num_batched_tokens=4096,
        max_model_len=4096,
        use_v2_block_manager=True,
        # disable_log_stats=False,
        skip_tokenizer_init=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    for prompt in input_prompts:
        print(f"prompt = {prompt}")
        tokenized_input = tokenizer(
            [prompt], return_tensors="pt", padding=True
        )["input_ids"]
        ip = [TokensPrompt(prompt_token_ids=ids) for ids in tokenized_input]
        for _ in range(10):
            outputs = llm.generate(ip, sampling_params)
            # dist.barrier()
        llm.llm_engine.model_executor.driver_worker.spec_decode_sampler.reset_metrics()
        print("==========================\n")

    # Print the output
    # for prompt, output in zip(input_prompts, outputs):
    #    print(f"prompt = {prompt}")
    #    print(f"output = {output}")
    #    print("==========================\n\n")
