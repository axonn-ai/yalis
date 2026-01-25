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
    model_id = "yalis/external/checkpoints/openai/gpt-oss-20b"

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

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, local_files_only=True)

    # Set a default chat template if not already present
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'user' %}"
            "{{ message['content'] }}\n\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}Assistant:{% endif %}"
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
    model_config = ModelConfig(model_name="gpt-oss-20b", model_path=model_id, precision="bf16")
    inference_config = InferenceConfig(
        max_batch_size=MAX_BATCH_SIZE,
        max_length_of_generated_sequences=1024,
        top_p=0.80,
        temperature=0.0,
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

    # Clamp token IDs to valid vocab range. The model's logits may include
    # padding up to padded_vocab_size, but the tokenizer only recognizes
    # tokens up to vocab_size. Clamp to prevent invalid token decoding.
    actual_vocab_size = tokenizer.vocab_size
    output_tokens = torch.clamp(output_tokens, max=actual_vocab_size - 1)

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(
        output_tokens, skip_special_tokens=True
    )

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")

    if enable_profiling:
        print_rank0(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10
            )
        )
