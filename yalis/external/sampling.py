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
    sorted_indices_to_remove[..., 0] = (
        False  # Ensure the most probable token is not removed
    )

    # Scatter the removal mask back to the original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )

    # Mask out the logits to remove
    logits = logits.masked_fill(indices_to_remove, float("-inf"))
    return logits


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    get_probs: bool = False,
    should_modify_probs: bool = False,
) -> torch.Tensor:
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")

    #print(f"Sampling got logits shape - {logits.shape}")

    if get_probs and not should_modify_probs:
        probs_returned = torch.nn.functional.softmax(logits, dim=-1)

    do_multinomial_sample = False
    if temperature > 0.0:
        logits = logits / temperature
        do_multinomial_sample = True

    if do_multinomial_sample:  # Multi-nomial Sampling
        # optionally crop the logits to only the top k options
        if top_k is not None:
            # Step 1: Compute the top-k values and indices along the last dimension
            v, i = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
            # Step 2: Create a tensor of -inf values with the same shape as logits
            logits.fill_(float("-inf"))
            # Step 3: Scatter the top-k values back into their original positions
            logits.scatter_(-1, i, v)

        # optionally scale the logits and sample from a probability distribution
        if top_p > 0.0 and top_p < 1.0:
            # Crop the logits to smallest set of logits with a cumulative probability above top_p
            logits = sample_top_p(logits, top_p)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        token_ids = multinomial_num_samples_1(probs)
    else:  # Greedy Sampling
        #print(f"I am greedy sampling")

        token_ids = torch.argmax(logits, dim=-1, keepdim=True)
        #print (f"token_ids: {token_ids}")
        #print (f"token_ids: {token_ids.shape}")

        # This is required for SpecDec which expects the probabilities to encode the sampling method
        if get_probs and should_modify_probs:
            probs = torch.zeros_like(logits)
            probs.scatter_(-1, token_ids, 1.0)

    if not get_probs:
        return token_ids
    else:
        if should_modify_probs:
            probs_returned = probs
        return token_ids, probs_returned
