import yalis
from yalis.pipelines import SpeculativeDecodingPipeline
from transformers import AutoTokenizer
import torch

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    fabric = yalis.init_everything()
    target_model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    target_model = yalis.model.get_model(
        target_model_id,
        fabric,
        litgpt_checkpoint_directory=f"../yalis/external/checkpoints/{target_model_id}"
    ).cuda()

    draft_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    draft_model = yalis.model.get_model(
        draft_model_id,
        fabric,
        litgpt_checkpoint_directory=f"../yalis/external/checkpoints/{draft_model_id}"
    ).cuda()

    # Initialize prompt and tokenizer
    prompt = "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    tokens_to_gen = 256
    # Print the initial prompt details
    yalis.print_rank0(f"Initial prompt: '{prompt}'")
    #print(f"Tokenized prompt (IDs): {tokens.tolist()}")


    # Initialize pipeline
    pipeline = SpeculativeDecodingPipeline(target_model, draft_model, tokenizer, torch.bfloat16, "cuda")
    pipeline.setup()

    pipeline.run(prompt, tokens_to_gen, 3, profile=True)
