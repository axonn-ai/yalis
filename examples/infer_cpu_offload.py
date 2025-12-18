"""
Example script demonstrating CPU offloading for large model inference.

CPU offloading allows you to run models that don't fit entirely in GPU memory
by keeping the model weights on CPU and streaming layers to GPU on demand.

Key features:
- Model weights reside on CPU
- One layer (or more) is loaded to GPU at a time
- Next layer is prefetched while current layer executes
- Uses CUDA streams to overlap computation with data transfer
"""

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
import torch.distributed as dist

NUM_ITERATIONS = 5
PROFILE_START_ITERATION = 2

if __name__ == "__main__":
    # Model ID from Hugging Face
    # CPU offloading is especially useful for larger models like 70B
    model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    user_prompts = [
        "How to bake a cake?",
        "Explain quantum mechanics in simple terms.",
    ]

    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        input_prompts.append(formatted_prompt)

    # Number of tokens to generate
    tokens_to_gen = 4

    # Max batch size (smaller batch sizes work better with CPU offloading
    # due to memory constraints)
    MAX_BATCH_SIZE = 4

    # Model configuration
    model_config = ModelConfig(model_name=model_id, precision="bf16")

    # Inference configuration with CPU offloading enabled
    inference_config = InferenceConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_length_of_generated_sequences=512,
        top_p=0.80,
        temperature=1.0,
        tp_dims=None,
        attention_backend="flash",
        # CPU offloading options
        use_cpu_offloading=True,  # Enable CPU offloading
        cpu_offload_num_prefetch_layers=1,  # Prefetch 1 layer ahead
        cpu_offload_pin_memory=True,  # Pin CPU memory for faster transfers
        # Note: Paged KV caching is not yet compatible with CPU offloading
        use_paged_kv_caching=True,
        prestore_kv_cache=True,
    )
    # Initialize the engine with CPU offloading
    engine = LLMEngine(
        model_config=model_config, inference_config=inference_config
    )


    print_rank0("=" * 60)
    print_rank0("CPU Offloading Inference Example")
    print_rank0("=" * 60)
    print_rank0(f"Model: {model_id}")
    print_rank0(f"CPU Offloading: Enabled")
    print_rank0(f"Prefetch Layers: {inference_config.cpu_offload_num_prefetch_layers}")
    print_rank0(f"Pin Memory: {inference_config.cpu_offload_pin_memory}")
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
    if hasattr(engine.model, 'cleanup'):
        engine.model.cleanup()

