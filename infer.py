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

def init_everything(dtype="bf16-mixed"):
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    if world_size > 1:
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
            num_nodes=world_size // torch.cuda.device_count(),
            precision=dtype,
            strategy=strategy,
        )
    else:
        fabric = Fabric(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            precision=dtype,
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
    token_id = torch.argmax(logits[0, -1])
    return token_id
    
@torch.no_grad()
def generate(model, tokens, input_pos):
    # Forward pass through the model
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    token_id = torch.argmax(logits[0, -1])
    return token_id

def compiled_fns():
    #torch._dynamo.config.automatic_dynamic_shapes = True
    #torch._inductor.config.triton.unique_kernel_names = True
    #torch._inductor.config.coordinate_descent_tuning = True
    #https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.assert_indirect_indexing = False
    prefill_compiled = torch.compile(prefill, fullgraph=True)
    generate_compiled = torch.compile(generate, fullgraph=True, mode="reduce-overhead")
    return prefill_compiled, generate_compiled

if __name__ == "__main__":
    # Assuming model and fabric setup functions exist as init_everything() and get_model()
    fabric = init_everything()
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = get_model(
        model_id,
        fabric,
        litgpt_checkpoint_directory=f"./external/checkpoints/{model_id}"
    ).cuda()

    prefill_fn, generate_fn = compiled_fns()


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
        output_tokens = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            for step in range(tokens_to_gen):
                if step == 0: # prefill
                    next_token = prefill_fn(model, prompt_tokens)
                    input_pos = torch.tensor([prompt_tokens.size(0)], device="cuda", dtype=torch.int64)
                    tokens = next_token.clone()
                else:
                    with torch.nn.attention.sdpa_kernel(
                                torch.nn.attention.SDPBackend.MATH):
                        next_token = generate_fn(model, tokens, input_pos)
                        input_pos.add_(1)
                        tokens.copy_(next_token)
                # Append token to output and log details
                output_tokens.append(next_token.clone())
        end = time.time()
        time_taken = end - start
        # Decode and display the final generated output
        generated_text = tokenizer.decode([x.item() for x in output_tokens])
        if TRIAL == 0:
            print_rank0("\nGenerated text:\n" + "-" * 40)
            print_rank0(generated_text)
            print_rank0("-" * 40)

        tokens_per_second = len(output_tokens) / time_taken
        print_rank0(f"Output {tokens_per_second} tok/s") 
