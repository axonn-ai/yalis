from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
from contextlib import nullcontext

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None


if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "/home/hoffmuki/scratch/yalis/yalis/external/checkpoints/openai/gpt-oss-20b"
    # model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # For GPT-OSS, use config name for model architecture, path for tokenizer/checkpoint
    model_name = "gpt-oss-20b"

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

    # take 16 prompts from this dataset
    user_prompts = user_prompts[:16]
    print(f"Number of prompts = {len(user_prompts)}")

    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    # profile the run or not
    enable_profiling = False

    # Tokenizer for encoding the prompt (load from local checkpoint directory)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

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
        # Print first prompt to verify Harmony format
        if len(input_prompts) == 1:
            print(f"First formatted prompt (first 300 chars):")
            print(repr(formatted_prompt[:300]))
            print(f"Full first prompt contains Harmony tags: {('<|start|>user' in formatted_prompt and '<|end|>' in formatted_prompt)}")


    # Number of tokens to generate
    tokens_to_gen = 512

    # Max batch size
    MAX_BATCH_SIZE = 32

    if len(input_prompts) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(input_prompts)} cannot be greater "
            f"than max batch size {MAX_BATCH_SIZE}"
        )

    # configs
    model_config = ModelConfig(
        model_name=model_name,  # Use config name, not path
        model_path="/home/hoffmuki/scratch/yalis/yalis/external/checkpoints/openai/gpt-oss-20b",
        precision="bf16",
    )
    inference_config = InferenceConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_length_of_generated_sequences=1024,
        top_p=0.80,
        temperature=1.0,
        tp_dims=None,
        attention_backend="sdpa",
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
    )

    engine = LLMEngine(
        model_config=model_config, inference_config=inference_config
    )

    if enable_profiling:
        profiler_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=5, warmup=2, active=1),
        )
    else:
        profiler_context = nullcontext()

    # Tokenize prompts to get input token IDs for concatenation
    prompt_tokens = tokenizer(
        input_prompts, 
        return_tensors="pt", 
        padding=True
    ).input_ids

    with profiler_context as prof:
        for iter in range(10):
            output_tokens, metrics = engine.generate(
                input_prompts,
                report_throughput=True,
                tokens_to_generate=tokens_to_gen,
            )
            if enable_profiling:
                prof.step()
            dist.barrier()

    output_tokens = output_tokens.cpu()
    prompt_tokens = prompt_tokens.cpu()

    # Concatenate prompt tokens with generated tokens for proper decoding
    # This is needed because GPT-OSS uses Harmony format tags that must be 
    # present in the sequence for correct token decoding
    full_sequences = torch.cat([prompt_tokens, output_tokens], dim=1)

    # Decode the full token sequences into text
    detokenized_text = tokenizer.batch_decode(
        full_sequences, skip_special_tokens=False
    )

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        
        # Extract the final answer from Harmony format
        # The model outputs: <|start|>assistant<|channel|>analysis...<|end|><|channel|>final<|message|>ANSWER<|end|>
        # We want to show the full output for debugging
        print_rank0(f"full output = {output}")
        
        # Try to extract just the final answer
        if "<|channel|>final<|message|>" in output:
            final_start = output.find("<|channel|>final<|message|>") + len("<|channel|>final<|message|>")
            final_end = output.find("<|end|>", final_start)
            if final_end > final_start:
                final_answer = output[final_start:final_end].strip()
                print_rank0(f"\nfinal answer = {final_answer}")
        
        print_rank0("==========================\n\n")

    if enable_profiling:
        print_rank0(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10
            )
        )
