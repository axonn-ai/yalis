import torch
from typing import Union, List, Optional
from .config import ModelConfig, InferenceConfig
from .model import get_model 
from .initialize import init_distributed
from .utils import print_rank0
import logging
import torch.distributed as dist
import torch._dynamo

# These flags are taken from the following URL - 
# https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False

precision_to_dtype = {
    "bf16" : torch.bfloat16,
    "fp16" : torch.float16,
    "fp32" : torch.float32,
} 

@torch._dynamo.disable
@torch.no_grad()
@torch.compile(fullgraph=True)
def prefill(model, tokens):
    """
    Prefill function for generating the first token.

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.

    Returns:
        token_id: The next predicted token.
    """

    input_pos = torch.arange(0, tokens.size(1), device= "cuda", dtype=torch.int64).unsqueeze(0).repeat(tokens.size(0),1).to("cuda")

    logits = model(tokens, input_pos)["logits"]
    token_id = torch.argmax(logits[:, -1, :], dim=1).unsqueeze(1)
    return token_id

@torch._dynamo.disable
@torch.no_grad()
@torch.compile(fullgraph=True, mode="reduce-overhead")
def generate(model, tokens, input_pos, get_probs=False):
    """
    Generate function for producing the next token(s).

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.
        input_pos: Position indices for the tokens.
        get_probs: If True, returns logits as well.

    Returns:
        token_id: The next predicted token.
        logits: (Optional) The raw logits from the model.
    """
    logits = model(tokens, input_pos)["logits"]
    token_id = torch.argmax(logits[:, -1, :], dim=1).unsqueeze(1)
    if get_probs:
        return token_id, logits
    else:
        return token_id


class LLMEngine:
    """
    The core engine for managing and running inference on large language models.
    """
    def __init__(
        self, 
        model_config: ModelConfig, 
        inference_config: InferenceConfig,
        device="cuda"
    ):
        """
        Initialize the LLM Engine with model and inference configurations.
        
        Args:
            model_config (ModelConfig): Configuration for model setup.
            inference_config (InferenceConfig): Configuration for inference behavior.
        """
        self.model_config = model_config
        self.inference_config = inference_config
        self.model = None  # Placeholder for the loaded model
        self.fabric = init_distributed()
        self.device = device
        self.dtype = precision_to_dtype[self.model_config.precision]
        self._initialize_model()

    def _initialize_model(self):
        """
        Internal method to load and set up the model based on ModelConfig.
        """
        print_rank0(f"Initializing model: {self.model_config.model_name}")
        print_rank0(f"Using precision: {self.model_config.precision}")
        self.model = get_model(self.fabric, 
                                self.model_config.model_path, 
                                self.dtype)
        self.model.set_kv_cache(batch_size=self.inference_config.batch_size, 
                                device=self.device, 
                                dtype=self.dtype)
    
    def generate(
        self, 
        input_tokens: torch.Tensor, 
        tokens_to_generate: int = 50,
    ) -> torch.Tensor:
        """
        Generates tokens from the model, starting with the provided tokenized input.

        Args:
            prompt_tokens (Tensor): The tokenized input to start generation.
            tokens_to_gen (int): Number of tokens to generate after the prompt.

        Returns:
            output_tokens (List[Tensor]): List of generated token IDs.
        """
        output_tokens = [[] for _ in range(input_tokens.size(0))]
        # Start timing the operation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.no_grad(), torch.autocast(self.device, dtype=self.dtype, cache_enabled=False):
            tokens = input_tokens.clone().to(self.device)  # Move prompt tokens to the device
            for step in range(tokens_to_generate):
                if step == 0:  # Prefill step
                    next_token = prefill(self.model, tokens)  # Call prefill function
                    input_pos = torch.tensor([tokens.size(1)], device=self.device, dtype=torch.int64).repeat(tokens.size(0), 1)
                    tokens = next_token.clone()
                else:  # Generation step
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        next_token = generate(self.model, tokens, input_pos)  # Call generate function
                        input_pos.add_(1)  # Increment position
                        tokens.copy_(next_token)  # Copy the new token into tokens

                # Append the generated token to output
                for prompt_no, new_token in enumerate(next_token):
                    output_tokens[prompt_no].append(new_token.clone())

        # End timing and calculate elapsed time
        end.record()
        torch.cuda.synchronize()  # Wait for all events to finish
        time_taken = start.elapsed_time(end) / 1000  # Time in seconds
        tput = sum([len(o) for o in output_tokens]) / time_taken
        #print(f"Generation took {time_taken:.2f} seconds.")
        print_rank0(f"Throughput = {tput:.2f} tok/s")

        return torch.tensor(output_tokens)
 

