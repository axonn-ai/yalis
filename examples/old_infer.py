from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer
import torch
from contextlib import nullcontext

# needed to work with pytorch 2.3
from torch.profiler import _KinetoProfile

_KinetoProfile._get_distributed_info = lambda self: None


if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    # Available prompts for testing different batch sizes
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

    # Set the maximum batch size for the model
    MAX_BATCH_SIZE = 8

    # Set batch sizes to test (must all be <= MAX_BATCH_SIZE)
    # BATCH_SIZES_TO_TEST = [1, 2, 4, 8]
    BATCH_SIZES_TO_TEST = [
        1,
        1,
        2,
        2,
    ]  # First batch is initialization, any subsequent batch should be faster.

    for batch_size in BATCH_SIZES_TO_TEST:
        if batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {batch_size} cannot be greater "
                f"than max batch size {MAX_BATCH_SIZE}"
            )

    print(f"Testing dynamic batching with max_batch_size={MAX_BATCH_SIZE}")
    print(f"Will test batch sizes: {BATCH_SIZES_TO_TEST}")

    system_prompt = (
        "You are a helpful chatbot. Answer the following question.\n"
    )

    # profile the run or not
    enable_profiling = False

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Number of tokens to generate
    tokens_to_gen = 512

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_length_of_generated_sequences=1024,
        top_p=0.80,
        temperature=1.0,
        tp_dims=None,
        attention_backend="flash",
        use_paged_kv_caching=False,
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

    # all_results = {}

    # Test each batch size
    for current_batch_size in BATCH_SIZES_TO_TEST:
        print_rank0(f"\n{'='*50}")
        print_rank0(f"TESTING BATCH SIZE: {current_batch_size}")
        print_rank0(f"{'='*50}")

        current_batch_prompts = user_prompts[:current_batch_size]

        # Format prompts for the model
        input_prompts = []
        for user_prompt in current_batch_prompts:
            conversation = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": user_prompt},
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            input_prompts.append(formatted_prompt)

        with profiler_context as prof:
            output_tokens, metrics = engine.generate(
                input_prompts,
                report_throughput=True,
                tokens_to_generate=tokens_to_gen,
            )
            if enable_profiling:
                prof.step()
            # dist.barrier()

        # Store results for this batch size
        # all_results[current_batch_size] = {
        #     'metrics': metrics,
        #     'output_tokens': output_tokens.cpu(),
        #     'prompts': current_batch_prompts
        # }

        # Decode the token IDs into text
        detokenized_text = tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )

        for i, (prompt, output) in enumerate(
            zip(current_batch_prompts, detokenized_text)
        ):
            print_rank0(
                f"\n===== Batch {current_batch_size}, Sample {i+1} ====="
            )
            print_rank0(f"prompt: {prompt}")
            print_rank0(f"output: {output}")

        if enable_profiling:
            print_rank0(
                prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=10
                )
            )
    # Print summary of all results
    # print_rank0(f"\n{'='*60}")
    # print_rank0("DYNAMIC BATCHING TEST SUMMARY")
    # print_rank0(f"{'='*60}")

    # for batch_size, results in all_results.items():
    #     metrics = results['metrics']
    #     print_rank0(f"\nBatch Size {batch_size}:")
    #     print_rank0(f"  Throughput: {metrics['Throughput']:.2f} tokens/sec")
    #     print_rank0(f"  TTFT: {metrics['TTFT']:.4f} ms")
    #     print_rank0(f"  TBT: {metrics['TBT']:.4f} ms")
    #     print_rank0(f"  E2E: {metrics['E2E']:.4f} ms")
