try:
    from mpi4py import MPI
except ImportError:
    pass

from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine
from transformers import AutoTokenizer


if __name__ == "__main__":
    # Model ID from Hugging Face
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    # Input prompt for the model
    prompt = [
        "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?",
        "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"
        ]

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token="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")

    # Tokenize the input prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids  # Remove batch dimension

    # Number of tokens to generate
    tokens_to_gen = 256

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size = len(prompt))

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    for _ in range(10):
        output_tokens = engine.generate(prompt_tokens)

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    # detokenized_texts = [tokenizer.decode(output_tokens_for_prompt, skip_special_tokens=True) for output_tokens_for_prompt in output_tokens]

    print_rank0(detokenized_text)


