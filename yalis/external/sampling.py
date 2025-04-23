import torch
from typing import Optional

def multinomial_num_samples_1(probs: torch.Tensor) -> torch.Tensor:
    if torch._dynamo.is_compiling():
        # Faster alternative to `torch.multinomial(probs, num_samples=1)` that is also CUDAGraph friendly
        distribution = torch.empty_like(probs).exponential_(1)
        return torch.argmax(probs / distribution, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    # Sort logits along the last dimension (V)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    
    # Compute cumulative probabilities
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    
    # Mask tokens with cumulative probabilities greater than top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # Always keep at least one token per batch
    sorted_indices_to_remove[..., 0] = False  # Ensure the most probable token is not removed
    
    # Scatter the removal mask back to the original indices
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
    
    # Mask out the logits to remove
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: float = 1.0
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")
    
    # optionally crop the logits to only the top k options
    if top_k is not None:
        # Step 1: Compute the top-k values and indices along the last dimension
        v, i = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)

        # Step 2: Create a tensor of -inf values with the same shape as logits
        logits_masked = torch.full_like(logits, float("-inf"))

        # Step 3: Scatter the top-k values back into their original positions
        logits_masked.scatter_(-1, i, v)

        # Step 4: Assign the masked logits back to the original logits
        logits = logits_masked
    # optionally scale the logits and sample from a probability distribution
    if temperature > 0.0 or top_p > 0.0:
        if temperature > 0.0:
            logits = logits / temperature
        # optionally crop the logits to smallest set of logits with a cumulative probability above top_p
        if top_p < 1.0:
            logits = sample_top_p(logits, top_p)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return multinomial_num_samples_1(probs)
    return torch.argmax(logits, dim=-1, keepdim=True)

