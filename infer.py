try:
    from mpi4py import MPI
except ImportError:
    pass

from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from model import get_model
from transformers import AutoTokenizer
import torch



def init_everything(num_nodes=1, dtype="bf16-mixed"):
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    assert world_size == 1, "does not support tensor parallelism yet"
    assert num_nodes == 1, "does not support multi-node yet"
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
        num_nodes=num_nodes,
        precision=dtype,
    )
    fabric.launch()
    return fabric

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    fabric = init_everything()
    model_id = "meta-llama/Llama-2-7b-hf"
    model = get_model(
        model_id,
        fabric,
        litgpt_checkpoint_directory=f"./external/checkpoints/{model_id}"
    )
    model.max_seq_length = 256  # arbitrary sequence length

    # Initialize prompt and tokenizer
    prompt = "High Performance Computing is"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze().cuda()
    tokens_to_gen = 16
    prefill_token = True
    prompt_size = tokens.size(0)

    # Print the initial prompt details
    print(f"Initial prompt: '{prompt}'")
    print(f"Tokenized prompt (IDs): {tokens.tolist()}")
    print(f"Prompt size: {prompt_size}")

    # Setup input position and model's KV cache for fast generation
    input_pos = torch.arange(0, prompt_size, device="cuda", dtype=torch.int64)
    model.set_kv_cache(batch_size=1, device='cuda', dtype=torch.bfloat16)

    # Generation loop
    output_tokens = []
    print("\nStarting token generation:")
    with torch.no_grad():
        for step in range(tokens_to_gen):
            # Forward pass through the model
            logits = model(tokens.view(1, -1), input_pos)["logits"]
            token_id = torch.distributions.Categorical(logits=logits[0, -1]).sample()
            
            # Append token to output and log details
            output_tokens.append(token_id.item())
            print(f"Step {step + 1}: Generated token ID {token_id.item()}, "
                  f"Text: '{tokenizer.decode([token_id])}'")

            # Update `tokens` and `input_pos` for the next step
            if prefill_token:
                prefill_token = False
                input_pos = torch.tensor([prompt_size], device="cuda", dtype=torch.int64)
            else:
                input_pos.add_(1)
            tokens = token_id

    # Decode and display the final generated output
    generated_text = tokenizer.decode(output_tokens)
    print("\nGenerated text:\n" + "-" * 40)
    print(generated_text)
    print("-" * 40)

