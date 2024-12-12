try:
    from mpi4py import MPI
except ImportError:
    pass

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer


if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Input prompt for the model
    user_prompts = [
        "How to bake a cake?",
        "How to drive a car on a freeway?",
    ] 
    system_prompt = "You are a helpful chatbot. Answer the following question.\n"

    input_prompts = []
    for user_prompt in user_prompts:
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(conversation, 
                add_generation_prompt=True, 
                tokenize=False)
        input_prompts.append(formatted_prompt)


    # Tokenize the input prompt
    prompt_tokens = tokenizer(input_prompts, return_tensors="pt", padding=True)
    unpadded_prompt_lengths = prompt_tokens.attention_mask.sum(dim=1)
    prompt_tokens = prompt_tokens.input_ids

    # Number of tokens to generate
    tokens_to_gen = 256

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size=len(input_prompts))

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    for _ in range(10):
        output_tokens = engine.generate(prompt_tokens, 
                                        unpadded_prompt_lengths=unpadded_prompt_lengths,
                                        report_throughput=True,
                                        tokens_to_generate=tokens_to_gen)

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    # detokenized_texts = [tokenizer.decode(output_tokens_for_prompt, skip_special_tokens=True) for output_tokens_for_prompt in output_tokens]

    for prompt, output in zip(user_prompts, detokenized_text):
        print_rank0("==========================\n\n")
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")
