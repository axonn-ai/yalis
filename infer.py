try:
    from mpi4py import MPI
except ImportError:
    pass

from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from model import get_model
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import time

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")

def init_everything(num_nodes=1, dtype="bf16-mixed"):
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    strategy = AxonnStrategy(
            G_intra_r=world_size,
            G_intra_c=1,
            G_intra_d=1,
            overlap_communication=True,
            enable_timers=False,
        )
    fabric = Fabric(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        num_nodes=num_nodes,
        precision=dtype,
        strategy=strategy,
    )
    fabric.launch()
    # this is very important to ensure that the same token is sampled on each TP rank!
    seed_everything(1234)
    return fabric

@torch.no_grad()
def prefill(model, tokens):
    # Forward pass through the model
    input_pos = torch.arange(0, tokens.size(0), device="cuda", dtype=torch.int64)
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    #token_id = torch.distributions.Categorical(logits=logits[0, -1]).sample()
    token_id = torch.argmax(logits[0, -1])
    return token_id
    
@torch.no_grad()
def generate(model, tokens, input_pos):
    # Forward pass through the model
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    #token_id = torch.distributions.Categorical(logits=logits[0, -1]).sample()
    token_id = torch.argmax(logits[0, -1])
    return token_id

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    fabric = init_everything()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = get_model(
        model_id,
        fabric,
        litgpt_checkpoint_directory=f"./external/checkpoints/{model_id}"
    ).cuda()


    # Initialize prompt and tokenizer
    prompt = "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze().cuda()
    tokens_to_gen = 256

    # Print the initial prompt details
    print_rank0(f"Initial prompt: '{prompt}'")
    #print(f"Tokenized prompt (IDs): {tokens.tolist()}")

    # Setup input position and model's KV cache for fast generation
    model.set_kv_cache(batch_size=1, device='cuda', dtype=torch.bfloat16)

    # Generation loop
    #print("\nStarting token generation:")
    for TRIAL in range(10):
        start = time.time()
        tokens = prompt_tokens
        output_tokens = []
        with torch.no_grad():
            for step in range(tokens_to_gen):
                if step == 0: # prefill
                    next_token = prefill(model, tokens)
                    input_pos = torch.tensor([tokens.size(0)], device="cuda", dtype=torch.int64)
                else:
                    next_token = generate(model, tokens, input_pos)
                    input_pos.add_(1)
                # Append token to output and log details
                output_tokens.append(next_token.item())
                tokens = next_token.clone()
        end = time.time()
        time_taken = end - start
        # Decode and display the final generated output
        generated_text = tokenizer.decode(output_tokens)
        if TRIAL == 0:
            print_rank0("\nGenerated text:\n" + "-" * 40)
            print_rank0(generated_text)
            print_rank0("-" * 40)

        tokens_per_second = len(output_tokens) / time_taken
        print_rank0(f"Output {tokens_per_second} tok/s") 
