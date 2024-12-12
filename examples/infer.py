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
    prompts = [
        "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?",
        ] * 16

    # Tokenizer for encoding the prompt
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Tokenize the input prompt
    prompt_tokens = tokenizer(prompts, return_tensors="pt").input_ids 

    # Number of tokens to generate
    tokens_to_gen = 256

    # configs
    model_config = ModelConfig(model_name=model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size = len(prompts))

    engine = LLMEngine(model_config=model_config, inference_config=inference_config)

    for _ in range(10):
        output_tokens = engine.generate(prompt_tokens, report_throughput=True)

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
    # detokenized_texts = [tokenizer.decode(output_tokens_for_prompt, skip_special_tokens=True) for output_tokens_for_prompt in output_tokens]

    for prompt, output in zip(prompts, detokenized_text):
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")

