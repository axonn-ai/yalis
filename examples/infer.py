try:
    from mpi4py import MPI
except ImportError:
    pass

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None

from contextlib import nullcontext

if __name__ == "__main__":
    # Model ID from Hugging Face
    # model_id = "google/gemma-2-27b-it"
    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "meta-llama/Llama-2-7b-chat-hf"

    # Input prompt for the model
    user_prompts = [
        "How to bake a cake?",
        # "How to drive a car on a freeway?",
    ] * 8
    system_prompt = "You are a helpful chatbot. Answer the following question.\n"

    # profile the run or not
    enable_profiling = False

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},  # not needed for gemma
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    # Number of tokens to generate
    tokens_to_gen = 256

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size=len(input_prompts))

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    # print_rank0(torch.cuda.memory_allocated() / 1e9)

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
        )
    else:
        profiler_context = nullcontext()

    with profiler_context as prof:
        for iter in range(4):
            output_tokens = engine.generate(
                input_prompts, report_throughput=True, tokens_to_generate=tokens_to_gen
            )
            if enable_profiling:
                prof.step()
            dist.barrier()

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")
        break

    if enable_profiling:
        print_rank0(
            prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        )
