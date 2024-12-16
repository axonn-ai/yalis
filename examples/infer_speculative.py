
try:
    from mpi4py import MPI
except ImportError:
    pass
from yalis import ModelConfig, InferenceConfig, print_rank0, LLMEngine, SpecDecLLMEngine
from transformers import AutoTokenizer
import torch
import random
import numpy as np

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    #target_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    target_model_id = "meta-llama/Llama-2-70b-chat-hf"
    #draft_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #draft_model_id = "lmsys/vicuna-7b-v1.3"
    draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # Initialize prompt and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    prompt = [
        "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"
    ]
    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids
    tokens_to_gen = 256

    # Configs
    target_model_config = ModelConfig(model_name=target_model_id, precision="bf16")
    draft_model_config = ModelConfig(model_name=draft_model_id, precision="bf16")
    inference_config = InferenceConfig(batch_size=1)

    engine = SpecDecLLMEngine(
        target_model_config=target_model_config,
        draft_model_config=draft_model_config,
        inference_config=inference_config,
    )

    for _ in range(10):
        output_tokens = engine.generate(prompt_tokens, tokens_to_gen, gamma=3, report_throughput=True)
        torch.cuda.synchronize()

    output_tokens = output_tokens.cpu()

    # Decode the token IDs into text
    detokenized_text = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    for prompt, output in zip(prompt, detokenized_text):
        print_rank0(f"prompt = {prompt}")
        print_rank0(f"output = {output}")
        print_rank0("==========================\n\n")

