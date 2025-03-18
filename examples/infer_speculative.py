try:
    from mpi4py import MPI
except ImportError:
    pass
from yalis import (
    ModelConfig,
    InferenceConfig,
    print_rank0,
    LLMEngine,
    SpecDecLLMEngine,
)
from transformers import AutoTokenizer
import torch
from torch.profiler import profile, record_function, ProfilerActivity
import random
import torch.distributed as dist
import numpy as np
from contextlib import nullcontext
from yalis.utils import get_gpu_memory_info, test_allreduce_bandwidth
import argparse

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Enable profiling
    parser.add_argument(
        "--enable_profiling",
        action="store_true",
        help="Enable profiling",
    )

    # Target Model ID
    parser.add_argument(
        "--target_model_id",
        type=str,
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Target model ID",
    )

    # Draft Model ID
    parser.add_argument(
        "--draft_model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Draft model ID",
    )

    # Tokens to generate
    parser.add_argument(
        "--tokens_to_gen",
        type=int,
        default=256,
        help="Number of tokens to generate",
    )

    # Target TP
    parser.add_argument(
        "--target_tp",
        type=int,
        nargs=3,
        default=None,
        help="Target tensor parallelism",
    )

    # Draft TP
    parser.add_argument(
        "--draft_tp",
        type=int,
        nargs=3,
        default=None,
        help="Draft tensor parallelism",
    )

    args = parser.parse_args()

    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    # target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # target_model_id = "meta-llama/Llama-2-70b-chat-hf"
    # target_model_id = "meta-llama/Llama-3.1-405B-Instruct"
    # target_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # draft_model_id = "meta-llama/Llama-2-7b-chat-hf"
    # draft_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # draft_model_id = "lmsys/vicuna-7b-v1.3"
    # draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    target_model_id = args.target_model_id
    draft_model_id = args.draft_model_id

    if args.target_tp is not None:
        target_tp = tuple(args.target_tp)
    else:
        target_tp = None

    if args.draft_tp is not None:
        draft_tp = tuple(args.draft_tp)
    else:
        draft_tp = None

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

    tokens_to_gen = args.tokens_to_gen

    # Configs
    target_model_config = ModelConfig(
        model_name=target_model_id, precision="bf16"
    )
    draft_model_config = ModelConfig(
        model_name=draft_model_id, precision="bf16", tp_dims=draft_tp
    )

    inference_config = InferenceConfig(
        batch_size=1,
        max_length_of_generated_sequences=512,
        top_p=0.0,
        temperature=0.0,
        tp_dims=target_tp,
    )

    if args.enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler_context = nullcontext()

    engine = SpecDecLLMEngine(
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        inference_config=inference_config,
    )

    print_rank0(f"[INFO] Target Model ID: {target_model_id}")
    print_rank0(f"[INFO] Draft Model ID: {draft_model_id}")
    print_rank0(f"[INFO] Target TP: {target_tp}")
    print_rank0(f"[INFO] Draft TP: {draft_tp}")

    with profiler_context as prof:
        for prompt in input_prompts:
            prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
            tputs = []
            acceptance_rates = []
            for i in range(30):
                # print (f"[{dist.get_rank()}] Running Iteration {i}")
                output_tokens, tput, acceptance_rate = engine.generate(
                    prompt_tokens,
                    tokens_to_gen,
                    gamma=5,
                    report_throughput=True,
                )
                tputs.append(tput)
                acceptance_rates.append(acceptance_rate)
                if args.enable_profiling:
                    prof.step()

                # print (f"[{dist.get_rank()}] Executing Barrier {i}")
                dist.barrier()

            # Average of last 5 throughput values
            avg_tput = sum(tputs[-10:]) / 10
            avg_acceptance_rate = sum(acceptance_rates[-10:]) / 10

            if dist.get_rank() == 0:
                print(
                    f"AT:{avg_tput:.2f} tok/s| AAR:{avg_acceptance_rate:.2f}"
                )

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    # detokenized_text = tokenizer.batch_decode(
    #    output_tokens, skip_special_tokens=True
    # )

    # for prompt, output in zip(input_prompts, detokenized_text):
    #    print_rank0(f"prompt = {prompt}")
    #    print_rank0(f"output = {output}")
    #    print_rank0("==========================\n\n")

    if args.enable_profiling:
        print_rank0(
            profiler_context.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10
            )
        )
        # Export json trace
        # prof.export_chrome_trace(f"trace_{dist.get_rank()}.json")
        # prof.export_memory_timeline(f"memory_{dist.get_rank()}.html")
