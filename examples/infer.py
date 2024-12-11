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
    prompt = "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize the input prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.squeeze(0)  # Remove batch dimension

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig()

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    for _ in range(10):
        output_tokens = engine.generate(prompt_tokens, tokens_to_generate=256)

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    print_rank0(detokenized_text)


