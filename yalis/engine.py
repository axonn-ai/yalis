import torch
from typing import Union, List, Optional
from .config import ModelConfig, InferenceConfig
from .model import get_model
from .initialize import init_distributed
from .utils import print_rank0, get_gpu_memory_info
from .external.sampling import sample
import logging
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
import gc
from .timers import Timers

# These flags are taken from the following URL -
# https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False

precision_to_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@torch.no_grad()
@torch.compile()
def prefill(model, tokens, unpadded_prompt_lengths, temperature=1.0, top_k=None, top_p=1.0):
    """
    Prefill function for generating the first token.

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.

    Returns:
        token_id: The next predicted token.
    """

    logits = model(tokens, unpadded_prompt_lengths)["logits"]
    logits = logits[torch.arange(logits.size(0)), unpadded_prompt_lengths - 1]
    token_id = sample(logits=logits, temperature=temperature, top_k=top_k, top_p=top_p)
    return token_id


@torch.no_grad()
@torch.compile(mode="reduce-overhead")
def generate(model, tokens, get_probs=False, temperature=1.0, top_k=None, top_p=1.0):
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
    logits = model(tokens)["logits"]
    token_id = sample(logits=logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p)
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
        device="cuda",
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
        self.device = device
        self.dtype = precision_to_dtype[self.model_config.precision]
        init_distributed(tp_dims=self.inference_config.tp_dims)
        print_rank0(f"Model Config: {self.model_config}")
        print_rank0(f"Inference Config: {self.inference_config}")
        self._initialize_model()
        torch.cuda.empty_cache()  # return extra memory to CUDA. Can prevent NCCL init OOMs
        gc.collect()
        print_rank0(f"Memory Stats After Initializing Model - {get_gpu_memory_info()} ")

    def _make_params_contiguous(self):
        if not self.model:
            print_rank0("Model must be initialized before contiguous parameter buffer can be allocated")
            return
        
        self.model = self.model.to(self.device) 
        return

        total_bytes = 0
        param_info, buf_info = [], []
        
        for name, param in self.model.named_parameters():
            num_bytes = param.numel() * param.element_size()
            param_info.append({
                "name": name,
                "shape": param.shape,
                "dtype": param.dtype,
                "num_bytes": num_bytes,
                "offset": total_bytes,
                "param": param,
            })
            total_bytes += num_bytes
        
        for name, buf in self.model.named_buffers():
            num_bytes = buf.numel() * buf.element_size()
            buf_info.append({
                "name": name,
                "shape": buf.shape,
                "dtype": buf.dtype,
                "num_bytes": num_bytes,
                "offset": total_bytes,
                "buf": buf,
            })
            total_bytes += num_bytes

        # make buffer 128-byte aligned
        total_bytes = total_bytes - (total_bytes % 128) + 128

        gpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")

        for info in param_info:
            param_view = gpu_buffer[info["offset"]: info["offset"] + info["num_bytes"]].view(info["dtype"]).reshape(info["shape"])
            param_view.copy_(info["param"], non_blocking=True)
            info["param"].data = param_view

        for info in buf_info:
            buf_view = gpu_buffer[info["offset"]: info["offset"] + info["num_bytes"]].view(info["dtype"]).reshape(info["shape"])
            buf_view.copy_(info["buf"], non_blocking=True)
            info["buf"].data = buf_view

    def _initialize_model(self):
        """
        Internal method to load and set up the model based on ModelConfig.
        """
        t0 = time.time()
        self.model = get_model(
            self.model_config.model_path,
            self.dtype,
            max_sequence_length=self.inference_config.max_length,
            random_init=False,
            use_intra_head_parallelism=self.inference_config.use_intra_head_parallelism,
            attention_backend=self.inference_config.attention_backend,
            use_paged_kv_caching=self.inference_config.use_paged_kv_caching,
            prestore_kv_cache=self.inference_config.prestore_kv_cache,
        )
        self._make_params_contiguous()
        self.model.set_kv_cache(
            batch_size=self.inference_config.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
        # Check if the tokenizer has a pad token, otherwise use eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print_rank0(
                "Pad token not found in the tokenizer. Using eos_token as pad token."
            )
        print_rank0(f"Initializing Model took {time.time() - t0} seconds")
    
    def reset_kv_cache(self, batch_size):
        if not self.model:
            print_rank0("Model must be initialized before contiguous parameter buffer can be allocated")
            return
        self.model.clear_kv_cache()
        self.model.set_kv_cache(
            batch_size=batch_size,
            device=self.device,
            dtype=self.dtype,
        )


    def generate(
        self,
        prompts: Union[list[str], list[list[int]]],
        tokens_to_generate: int = 50,
        report_throughput: bool = False,
        ignore_eos: Optional[bool] = True
    ) -> torch.Tensor:
        """
        Generate tokens based on input prompts, which can either be a list of strings or a list of token ID lists.

        This method processes the provided prompts, either by tokenizing input strings or directly using tokenized inputs,
        and generates additional tokens based on the model's current state.

        Args:
            prompts (Union[list[str], list[list[int]]]): A list of prompts to generate from.
                - If `list[str]`, each string will be tokenized into input IDs for the model.
                - If `list[list[int]]`, each sublist contains token IDs for the model to process directly.
            tokens_to_generate (int, optional): The number of tokens to generate beyond the input prompts. Defaults to 50.
            report_throughput (bool, optional): A flag indicating whether to report throughput during the generation process. Defaults to False.

        Returns:
            torch.Tensor: A tensor containing the generated tokens, with shape `(batch_size, tokens_to_generate)`.

        """
        timers = Timers()

        timers.start("tokenize")
        if isinstance(prompts, list) and all(isinstance(p, str) for p in prompts):
            prompt_tokens_and_mask = self.tokenizer(
                prompts, return_tensors="pt", padding=True
            )
            prompt_tokens = prompt_tokens_and_mask.input_ids
            # prompt tokens contain padding tokens. Summing the attention mask
            # gives us the actual sequence lengths of each prompt sans padding
            prompt_sequence_lengths = prompt_tokens_and_mask.attention_mask.sum(dim=1)
        elif isinstance(prompts, list) and all(
            isinstance(p, list) and all(isinstance(x, int) for x in p) for p in prompts
        ):
            # Get the maximum length of the sequences
            max_length = max(len(p) for p in prompts)
            prompt_tokens = torch.tensor(
                [
                    (
                        p + [self.tokenizer.pad_token] * (max_length - len(p))
                        if len(p) < max_length
                        else p
                    )
                    for p in prompts
                ]
            )
            prompt_sequence_lengths = torch.tensor([len(p) for p in prompts])
        else:
            raise TypeError(
                "prompts must be either a list of strings or a list of lists of integers"
            )

        if prompt_sequence_lengths.max() > self.model.max_seq_length:
            raise ValueError(
                f"The prompt sequence length ({prompt_sequence_lengths.max()}) exceeds the model's maximum sequence length "
                f"({self.model.max_seq_length}). Unable to proceed."
            )

        if prompt_sequence_lengths.max() + tokens_to_generate > self.model.max_seq_length:
            tokens_to_generate = self.model.max_seq_length - prompt_sequence_lengths.max()
            print_rank0(f"tokens_to_generate has been adjusted to {tokens_to_generate}")

        timers.stop("tokenize")
        print_rank0(
            f"Tokenization took {timers.get_times()[0][('tokenize',)]} ms"
        )

        # Using tensor size instead of inference config object to allow users to choose batch sizes < max batch size
        # (later defined in inference config).
        batch_size = prompt_tokens.size(0)
        ignore_eos = ignore_eos
        # Initialize done mask for multiple batches
        if batch_size > 1 and not ignore_eos:
            done_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        finished_reason = "Max Token Length"
        output_tokens = []
        # Start timing the operations
        timers.start("generate")
        self.model.token_counter.zero_()
        if self.inference_config.use_paged_kv_caching:
            self.model.kv_cache_manager.reset()
        with torch.no_grad(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            current_input_to_model = prompt_tokens.clone().to(
                self.device
            )  # Move prompt tokens to the device
            prompt_sequence_lengths = prompt_sequence_lengths.to(self.device)
            for step in range(tokens_to_generate):
                timer_key = None
                if step == 0:  # Prefill step
                    timer_key = "prefill"
                    timers.start(timer_key)
                    # print_rank0(f"mem before prefill = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    next_token = prefill(
                        self.model, current_input_to_model, prompt_sequence_lengths, 
                        temperature=self.inference_config.temperature, 
                        top_k=self.inference_config.top_k, 
                        top_p=self.inference_config.top_p
                    )  # Call prefill function
                    # print_rank0(f"mem after prefill = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    current_input_to_model = next_token.clone()
                else:  # Generation step
                    timer_key = "decode"
                    timers.start(timer_key)
                    with sdpa_kernel(SDPBackend.MATH):
                        next_token = generate(
                            self.model, current_input_to_model, 
                            temperature=self.inference_config.temperature, 
                            top_k=self.inference_config.top_k, 
                            top_p=self.inference_config.top_p
                        )  # Call generate function
                    # print_rank0(f"mem after generate {step} = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    current_input_to_model.copy_(
                        next_token
                    )  # Copy the new token into tokens

                # EOS Support:
                # Flatten to shape (batch_size,) for element wise comparison
                done_mask |= (next_token.view(-1) == self.tokenizer.eos_token_id)
                # Reshape to match next_token's shape for masked_fill()
                mask = done_mask.view(-1, 1)
                # Force EOS prompts to stay EOS
                next_token.masked_fill_(mask, self.tokenizer.eos_token_id)

                output_tokens.append(next_token.clone())

                # Break only if every sequence is done
                if done_mask.all():
                    print_rank0(f"All batch samples reached EOS at step {step}, stopping.")
                    timers.stop(timer_key)
                    finished_reason = "EOS"
                    break

                timers.stop(timer_key)

        output_tensor = torch.cat(output_tokens, dim=1)
        # End timing and calculate elapsed time
        timers.stop("generate")
        times, events = timers.get_times()
        tput = prompt_tokens.shape[0] * tokens_to_generate / (times[('generate',)] / 1000)
        ttft = (
            (times[('generate', 'prefill')] / events[('generate', 'prefill')])
        )
        tbt = (
            (times[('generate', 'decode')] / events[('generate', 'decode')]) 
        )
        metrics = {
            "BatchSize": prompt_tokens.shape[0],
            "PromptLength": prompt_tokens.shape[1],
            "DecodeLength": tokens_to_generate,
            "Throughput": tput,
            "TTFT": ttft,
            "TBT": tbt,
            "E2E": times[('generate',)],
            "TokenizationTime": times[('tokenize',)],
            "FinishedReason": finished_reason   # NOTE: This should be a list containing reasons for each batch but 
                                                # our EOS stopping currently is all-or-nothing.
        }
        if dist.get_rank() == 0 and report_throughput:
            print (f"[Metrics] BatchSize = {prompt_tokens.shape[0]}, PromptLength = {prompt_tokens.shape[1]}, DecodeLength = {tokens_to_generate}, Throughput = {tput:.2f} tok/s, TTFT = {ttft:.4f} ms, TBT = {tbt:.4f} ms, E2E = {times[('generate',)]:.4f} ms, FinishedReason = {finished_reason}")


        return output_tensor, metrics
