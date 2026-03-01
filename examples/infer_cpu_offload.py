"""
Example script demonstrating CPU offloading for large model inference.

CPU offloading allows you to run models that don't fit entirely in GPU memory
by keeping the model weights on CPU and streaming layers to GPU on demand.

Key features:
- Model weights reside on CPU
- One layer (or more) is loaded to GPU at a time
- Next layer is prefetched while current layer executes
- Uses CUDA streams to overlap computation with data transfer

Advanced features:
- Pre-allocated GPU buffers: Uses .copy_() instead of .to() for zero-allocation
  transfers. Enables selective prefetching.
- Selective prefetching: Prefetch only MLP, only attention, or specific rows
  of weight matrices (useful for sparse/expert computation).
"""

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from yalis.config import CPUOffloadConfig
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

NUM_ITERATIONS = 5
PROFILE_START_ITERATION = 3

if __name__ == "__main__":
    # Model ID from Hugging Face
    # CPU offloading is especially useful for larger models like 70B
    # model_id = "meta-llama/Llama-3.2-1B-Instruct"
    model_id = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"

    user_prompts = [
        # " ".join(["How to bake a cake?"] * 160)
        "Explain quantum mechanics in simple terms.",
        # "What are the best practices for time management?",
        # "How do I write a great resume for a software engineer role?",
    ]

    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    input_prompts = []
    for user_prompt in user_prompts:
        prompt = system_prompt + user_prompt
        print(prompt)
        conversation = [
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    # Number of tokens to generate
    tokens_to_gen = 10

    # Max batch size (smaller batch sizes work better with CPU offloading
    # due to memory constraints)
    MAX_BATCH_SIZE = 1

    # Model configuration
    model_config = ModelConfig(model_name=model_id, precision="bf16")

    # Inference configuration with CPU offloading enabled
    inference_config = InferenceConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_length_of_generated_sequences=1024,
        top_p=0.0,
        temperature=0.0,
        tp_dims=None,
        attention_backend="flash",
        # CPU offloading config
        cpu_offload=CPUOffloadConfig(
            modules=["mlp.experts"],
            prefetch_mode="selective",
            num_prefetch_layers=1,
            pin_memory=True,
            use_preallocated_buffers=True,
        ),
        use_paged_kv_caching=False,
        prestore_kv_cache=True,
        default_vector_prefetching=True,
        default_vector_path="./defaultvect/dv_buff_qwen_instruct",
    )
    # Initialize the engine with CPU offloading
    engine = LLMEngine(
        model_config=model_config, inference_config=inference_config
    )

    print_rank0("=" * 60)
    print_rank0("CPU Offloading Inference Example")
    print_rank0("=" * 60)
    print_rank0(f"Model: {model_id}")
    print_rank0(f"CPU Offload: {inference_config.cpu_offload}")
    print_rank0("=" * 60)

    print_rank0("\nRunning inference with CPU offloading...")

    # Run inference
    for iteration in range(NUM_ITERATIONS):
        dist.barrier()
        if iteration == PROFILE_START_ITERATION:
            torch.cuda.profiler.start()

        output_tokens, metrics = engine.generate(
            input_prompts,
            report_throughput=True,
            tokens_to_generate=tokens_to_gen,
            enable_nvtx=True,
        )

        dist.barrier()

        if iteration == NUM_ITERATIONS - 1:
            torch.cuda.profiler.stop()

    output_tokens = output_tokens.cpu()
    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )

    print_rank0("\n" + "=" * 60)
    print_rank0("Results")
    print_rank0("=" * 60)

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0(f"\nPrompt: {prompt}")
        print_rank0(f"Output: {output}")
        print_rank0("-" * 40)

    # Clean up if using CPU offloading
    if hasattr(engine.model, "cleanup"):
        engine.model.cleanup()
