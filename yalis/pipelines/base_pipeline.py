import torch
from yalis import print_rank0

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False


@torch.no_grad()
@torch.compile(fullgraph=True)
def prefill(model, tokens):
    # Forward pass through the model
    input_pos = torch.arange(0, tokens.size(0), device="cuda", dtype=torch.int64)
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    token_id = torch.argmax(logits[0, -1])
    return token_id


# todo - verify does not work
# @torch.compile(fullgraph=True)
@torch.no_grad()
def verify(model, tokens, target_pos):
    # Forward pass through the model
    # print (target_pos)
    input_pos = target_pos + torch.arange(
        0, tokens.size(0), device="cuda", dtype=torch.int64
    )
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    # print (logits.size())
    token_id = torch.argmax(logits[0, -1])
    return token_id, logits


@torch.no_grad()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def generate(model, tokens, input_pos, get_probs=False):
    # Forward pass through the model
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    token_id = torch.argmax(logits[0, -1])
    if get_probs:
        return token_id, logits
    else:
        return token_id


# This is the base class defining the interface of all pipelines
class BasePipeline:
    def __init__(self, device):
        self.device = device

    def run(self, *args, **kwargs):
        return NotImplementedError
