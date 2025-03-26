import torch
from typing import Union, List, Optional
from .config import ModelConfig, InferenceConfig
from .model import get_model
from .initialize import init_distributed
from .utils import print_rank0, get_gpu_memory_info, test_allreduce_bandwidth
from .external.sampling import sample
from .external.rejection_sampler import RejectionSampler
import logging
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
import gc
import time
from .timers import Timers

try:
    from mpi4py import MPI
except ImportError:
    pass

# These flags are taken from the following URL -
# https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True  # Experimental feature to reduce compilation times, will be on by default in future
torch._inductor.config.assert_indirect_indexing = False
torch._inductor.config.combo_kernel_foreach_dynamic_shapes = True

BYTE_TO_GB = 1 / float(1024 * 1024 * 1024)

precision_to_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

@torch.no_grad()
@torch.compile()
def prefill(model, tokens, unpadded_prompt_lengths=None, temperature=1.0, top_k=None, top_p=1.0, get_logits=False, is_verify=False):
    """
    Prefill function for generating the first token.

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.

    Returns:
        token_id: The next predicted token.
    """

    logits = model(tokens, unpadded_prompt_lengths, is_verify)["logits"]

    if unpadded_prompt_lengths is None:
        unpadded_prompt_lengths = 0  # If not provided, assume no padding

    token_id = sample(
        logits=logits[
            torch.arange(logits.size(0)), unpadded_prompt_lengths - 1
        ],
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if get_logits:
        return token_id, logits
    else:
        return token_id


#@torch.compile(mode="reduce-overhead")
#@torch.compile(mode="reduce-overhead")
#@torch.compile()
@torch.no_grad()
def verify(model, tokens, unpadded_prompt_lengths=None, temperature=1.0, top_k=None, top_p=1.0):
    """
    Verify function used in Speculative Decoding

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.

    Returns:
        token_id: The next predicted token.
    """

    logits = model(tokens, unpadded_prompt_lengths, is_verify=True)["logits"]

    num_parallel_drafters = 4
    # Reshape logits to (batch_size, num_parallel_drafters, -1, vocab_size)
    logits = logits.view(logits.size(0), num_parallel_drafters, -1, logits.size(-1))

    if unpadded_prompt_lengths is None:
        unpadded_prompt_lengths = 0 # If not provided, assume no padding

    # Currently supports only batch size = 1 
    token_ids, probs = sample(logits=logits,
                      #[torch.arange(logits.size(0)), unpadded_prompt_lengths - 1], 
                      temperature=temperature, 
                      top_k=top_k,
                      top_p=top_p, 
                      get_probs=True,
                      should_modify_probs=True)
    rank = dist.get_rank()

    # Get token_ids for this rank
    token_ids = token_ids[:, rank, :, :].squeeze(1)
    probs = probs[:, rank, :, :].squeeze(1)


    print (f"TokenIds - {token_ids.shape}")
    print (f"TokenIds - {token_ids}")
    #exit()
    return token_ids[torch.arange(logits.size(0)), unpadded_prompt_lengths - 1], probs

#@torch.compile(mode="reduce-overhead")
@torch.no_grad()
@torch.compile()
def generate(
    model,
    tokens,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    get_probs=False,
    should_modify_probs=False
):
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
    #print_rank0(f"Get probs- {get_probs}")
    return sample(
        logits=logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p, get_probs=get_probs, should_modify_probs=should_modify_probs
    )


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
        test_allreduce_bandwidth()
        self.model, self.tokenizer = self._initialize_model(self.model_config)
        torch.cuda.empty_cache()  # return extra memory to CUDA. Can prevent NCCL init OOMs
        gc.collect()
        print_rank0(f"Memory Stats After Initializing Model - {get_gpu_memory_info()} ")

    def _make_params_contiguous(self, model):
        if not model:
            print_rank0(
                "Model must be initialized before contiguous parameter buffer can be allocated"
            )
            return
        
        return model.to(self.device) 

        total_bytes = 0
        param_info, buf_info = [], []

        for name, param in model.named_parameters():
            num_bytes = param.numel() * param.element_size()
            param_info.append(
                {
                    "name": name,
                    "shape": param.shape,
                    "dtype": param.dtype,
                    "num_bytes": num_bytes,
                    "offset": total_bytes,
                    "param": param,
                }
            )
            total_bytes += num_bytes

        for name, buf in model.named_buffers():
            num_bytes = buf.numel() * buf.element_size()
            buf_info.append(
                {
                    "name": name,
                    "shape": buf.shape,
                    "dtype": buf.dtype,
                    "num_bytes": num_bytes,
                    "offset": total_bytes,
                    "buf": buf,
                }
            )
            total_bytes += num_bytes

        # make buffer 128-byte aligned
        total_bytes = total_bytes - (total_bytes % 128) + 128

        gpu_buffer = torch.empty(total_bytes, dtype=torch.uint8, device="cuda")

        for info in param_info:
            param_view = (
                gpu_buffer[info["offset"] : info["offset"] + info["num_bytes"]]
                .view(info["dtype"])
                .reshape(info["shape"])
            )
            param_view.copy_(info["param"], non_blocking=True)
            info["param"].data = param_view

        for info in buf_info:
            buf_view = (
                gpu_buffer[info["offset"] : info["offset"] + info["num_bytes"]]
                .view(info["dtype"])
                .reshape(info["shape"])
            )
            buf_view.copy_(info["buf"], non_blocking=True)
            info["buf"].data = buf_view

        return model

    def _initialize_model(self, model_config):
        """
        Internal method to load and set up the model based on ModelConfig.
        """
        t0 = time.time()
        print_rank0(f"Initializing model: {model_config.model_name}")
        print_rank0(f"Using precision: {model_config.precision}")
        model = get_model(model_config, self.dtype, max_sequence_length=self.inference_config.max_length)
        print_rank0(f"Making model parameters contiguous")
        model = self._make_params_contiguous(model)
        model.set_kv_cache(
            batch_size=self.inference_config.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        # Check if the tokenizer has a pad token, otherwise use eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print_rank0(
                "Pad token not found in the tokenizer. Using eos_token as pad token."
            )
        print_rank0(f"Initializing Model took {time.time() - t0} seconds")
        return model, tokenizer

    def generate(
        self,
        prompts: Union[list[str], list[list[int]]],
        tokens_to_generate: int = 50,
        report_throughput: bool = False,
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
        if isinstance(prompts, list) and all(
            isinstance(p, str) for p in prompts
        ):
            prompt_tokens_and_mask = self.tokenizer(
                prompts, return_tensors="pt", padding=True
            )
            prompt_tokens = prompt_tokens_and_mask.input_ids
            # prompt tokens contain padding tokens. Summing the attention mask
            # gives us the actual sequence lengths of each prompt sans padding
            prompt_sequence_lengths = (
                prompt_tokens_and_mask.attention_mask.sum(dim=1)
            )
        elif isinstance(prompts, list) and all(
            isinstance(p, list) and all(isinstance(x, int) for x in p)
            for p in prompts
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

        output_tokens = []
        # Start timing the operations
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timers = Timers()
        start.record()
        self.model.token_counter.zero_()
        with torch.no_grad(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            current_input_to_model = prompt_tokens.clone().to(
                self.device
            )  # Move prompt tokens to the device
            prompt_sequence_lengths = prompt_sequence_lengths.to(self.device)
            for step in range(tokens_to_generate):
                if step == 0:  # Prefill step
                    # print_rank0(f"mem before prefill = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    timers.start("prefill")
                    next_token = prefill(
                        self.model,
                        current_input_to_model,
                        prompt_sequence_lengths,
                        temperature=self.inference_config.temperature,
                        top_k=self.inference_config.top_k,
                        top_p=self.inference_config.top_p,
                    )  # Call prefill function
                    timers.stop("prefill")
                    # print_rank0(f"mem after prefill = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    current_input_to_model = next_token.clone()
                else:  # Generation step
                    timers.start("decode")
                    with sdpa_kernel(SDPBackend.MATH):
                        next_token = generate(
                            self.model,
                            current_input_to_model,
                            temperature=self.inference_config.temperature,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                        )  # Call generate function
                    timers.stop("decode")
                    # print_rank0(f"mem after generate {step} = {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    current_input_to_model.copy_(
                        next_token
                    )  # Copy the new token into tokens
                output_tokens.append(next_token.clone())
        output_tensor = torch.cat(output_tokens, dim=1)
        # End timing and calculate elapsed time
        end.record()
        torch.cuda.synchronize()  # Wait for all events to finish
        time_taken = start.elapsed_time(end) / 1000  # Time in seconds
        tput = prompt_tokens.shape[0] * tokens_to_generate / time_taken
        timers,_ = timers.get_times()
        if report_throughput and dist.get_rank() == 0:
            print(f"Throughput = {tput:.2f} tok/s, Time = {time_taken:.2f} s")
            print (f"Timers: {timers}")
        return output_tensor, tput


class SpecDecLLMEngine(LLMEngine):
    def __init__(
        self,
        target_model_config: ModelConfig,
        draft_model_config: ModelConfig,
        inference_config: InferenceConfig,
        parallel_drafting: bool = False,
        device="cuda",
    ):
        super().__init__(target_model_config, inference_config, device)
        self.draft_model_config = draft_model_config
        self.draft_model, _ = super()._initialize_model(draft_model_config)
        #print (f"Draft model: {self.draft_model}")
        self.sampler = RejectionSampler()
        self.parallel_drafting = parallel_drafting
    
    # Function 
    @torch.no_grad()
    def logits_to_probs(self, logits, temperature: float = 1.0):
        logits = logits / max(temperature, 1e-5)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs

    def generate(
        self,
        input_tokens: torch.Tensor,
        tokens_to_gen,
        gamma,
        report_throughput: bool = False,
        iter_idx: Optional[int] = None,
    ):
        #torch.set_deterministic_debug_mode(1)
        output_tokens = []
        # Start timing the operation
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timers = Timers()
        start.record()

        rank = dist.get_rank()
        num_drafters = dist.get_world_size()

        self.model.token_counter.zero_()
        self.draft_model.token_counter.zero_()
        torch.cuda.empty_cache()

        global_accepted_tokens = 0
        generated_draft_tokens = 0

        print_initial_tokens = True
        total_rejected_tokens = []
        
        with torch.no_grad(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            tokens = input_tokens.clone().to(
                self.device
            )  # Move prompt tokens to the device
            generated_tokens = 0
            step = 0
            while generated_tokens < tokens_to_gen:
                print (f"[{rank}] Step: {step}, iteration: {iter_idx}")
                step+=1
                #print_rank0(f"Token Generation - {torch.cuda.memory_allocated() * BYTE_TO_GB}, {torch.cuda.max_memory_allocated() * BYTE_TO_GB}")
                if generated_tokens == 0:  # Prefill step
                    # Only the target model is used for prefilling

                    #timers.start("prefill")
                    _ = prefill(self.draft_model, tokens, top_p=0.0, temperature=0.0)
                    next_token = prefill(self.model, tokens, top_p=0.0, temperature=0.0)
                    #timers.stop("prefill")

                    tokens = next_token.clone()
                    generated_tokens += 1
                    output_tokens.append(next_token.clone())

                    # print (f"Prefill step: {self.model.get_token_count()}, {self.draft_model.get_token_count()}")
                else:  # Generation step

                    # Draft tokens
                    draft_tokens = tokens.clone()
                    draft_output_tokens = []
                    draft_output_tokens.append(draft_tokens.clone())

                    draft_probs = []
                    with sdpa_kernel(SDPBackend.MATH):
                        for draft_step in range(gamma):
                            # Get next draft token and its probabilities
                            #timers.start("draft")
                            #print_rank0(f"Token Generation - {torch.cuda.memory_reserved() * BYTE_TO_GB}, {torch.cuda.max_memory_reserved() * BYTE_TO_GB}")
                            next_token, probs = generate(
                                self.draft_model,
                                draft_tokens,
                                top_p=0.5,
                                temperature=1.0,
                                get_probs=True,
                                should_modify_probs=True
                            )
                            draft_tokens.copy_(next_token)
                            draft_probs.append(probs.unsqueeze(1).clone())
                            generated_draft_tokens += 1

                            draft_output_tokens.append(next_token.clone())
                            #timers.stop("draft")

                        _ = generate(self.draft_model, draft_tokens, top_p=0.5, temperature=1.0)
                        #timers.stop("draft")
                        #print_rank0(f"Draft Probs: {len(draft_probs)}, Draft Probs Size: {draft_probs[0].size()}")

                        #print_rank0(f"Decode step: {self.model.get_token_count()}, {self.draft_model.get_token_count()}")
                        draft_probs = torch.cat(draft_probs, dim=1)
                        #draft_probs = self.logits_to_probs(draft_probs)
                        draft_output_tokens = torch.cat(draft_output_tokens, dim=1)
                        print (f"[{rank},{iter_idx},{step}] Generated {gamma} draft tokens: {draft_output_tokens}")
                        #print_rank0(f"[Decode] Draft Output Tokens: {draft_output_tokens}")
                        #print_rank0(f"Draft Probs: {draft_probs.size()}, Draft Output Tokens: {draft_output_tokens.size()}")

                        #print_rank0(f"Draft Probs: {draft_probs.size()}, Draft Output Tokens: {draft_output_tokens.size()}")
                        if self.parallel_drafting:
                            #if False:
                            # We need to gather the draft tokens from all ranks
                            draft_output_tokens_all = torch.cat(
                                [torch.empty_like(draft_output_tokens) for _ in range(dist.get_world_size())], dim=0
                            )
                            print (f"[{rank}] Draft Output Tokens: {draft_output_tokens}")
                            dist.all_gather_into_tensor(draft_output_tokens_all, draft_output_tokens)

                            draft_output_tokens_all = draft_output_tokens_all.view(1, -1)
                            print(f"[{rank}] Draft Output Tokens: {draft_output_tokens_all}")
                        else:
                            draft_output_tokens_all = draft_output_tokens

                        



                        torch.cuda.synchronize()
                        dist.barrier()
                        MPI.COMM_WORLD.Barrier()


                        # Verify the output of the draft model
                        #timers.start("verify")
                        next_token, target_probs = verify(
                            self.model, draft_output_tokens_all, top_p=0.0, temperature=0.0
                        )
                        #target_probs = target_probs.unsqueeze(1)
                        #print_rank0(f"Verify step: {target_probs}")

                        #target_probs = self.logits_to_probs(target_probs)
                        #timers.stop("verify")

                    # print_rank0(f"Verify step: {self.model.get_token_count()}, {self.draft_model.get_token_count()}")

                    ## Rejection Sampling
                    #timers.start("rejection_sampling")
                    # output_with_bonus_tokens = draft_output_tokens
                    output_with_bonus_tokens = self.sampler(
                        draft_probs,
                        target_probs,
                        draft_output_tokens[:, 1:],
                        next_token,
                    )
                    print (f"[{rank},{iter_idx},{step}] Output with bonus tokens: {output_with_bonus_tokens}")
                    # if print_initial_tokens:
                        # print_rank0(f"Draft Tokens: {draft_output_tokens}")
                        # print_rank0(f"Draft Probs: {draft_probs[:, :5]}")
                        # print_rank0(f"Output with Bonus Tokens: {output_with_bonus_tokens}")
                        # print_initial_tokens = False
                    #timers.stop("rejection_sampling")
                    assert (
                        output_with_bonus_tokens.size(0) == 1
                    )  # Only batch size 1 is supported

                    mask_negative = output_with_bonus_tokens == -1
                    if mask_negative.any():
                        num_accepted_tokens = mask_negative.nonzero(
                            as_tuple=True
                        )[1][0]
                    else:
                        num_accepted_tokens = output_with_bonus_tokens.size(1)
                    print (f"[{rank},{iter_idx},{step}] Number of accepted tokens: {num_accepted_tokens}")

                    # All gather the number of accepted tokens from all the parallel drafters
                    if self.parallel_drafting:
                        num_accepted_tokens_all = torch.zeros(
                            dist.get_world_size(), dtype=torch.int64, device="cuda"
                        )
                        num_accepted_tokens = torch.tensor([num_accepted_tokens], device="cuda", dtype=torch.int64)
                        dist.all_gather_into_tensor(num_accepted_tokens_all, num_accepted_tokens)

                        print (f"[{rank}] Number of accepted tokens: {num_accepted_tokens_all}")

                    #print (f"[{rank},{iter_idx},{step}] Number of accepted tokens: {num_accepted_tokens}")
                    exit()


                    global_accepted_tokens += num_accepted_tokens - 1

                    output_with_bonus_tokens = output_with_bonus_tokens[:,:num_accepted_tokens]
                    accepted_tokens = torch.split(output_with_bonus_tokens, 1, dim=1)
                    num_rejected_tokens = draft_output_tokens.size(1) - num_accepted_tokens

                    total_rejected_tokens.append(num_rejected_tokens)

                    self.model.rewind_kv_cache(num_rejected_tokens)
                    self.draft_model.rewind_kv_cache(num_rejected_tokens)

                    tokens.copy_(accepted_tokens[-1])
                    generated_tokens += num_accepted_tokens
                    output_tokens.extend(accepted_tokens)

                    MPI.COMM_WORLD.Barrier()
        
        print (f"[{rank}]Generated tokens: {len(output_tokens)}")

        output_tensor = torch.cat(output_tokens, dim=1)
        # End timing and calculate elapsed time
        end.record()
        torch.cuda.synchronize()  # Wait for all events to finish

        time_taken = start.elapsed_time(end) / 1000  # Time in seconds
        print (f"[{rank}] Recorded Time")
        #timers,_ = timers.get_times()
        tput = input_tokens.shape[0] * tokens_to_gen / time_taken
        #if report_throughput and dist.get_rank() == 0:
        #    print(f"Throughput = {tput:.2f} tok/s, Acceptance Rate = {global_accepted_tokens / generated_draft_tokens:.2f}, Time = {time_taken:.2f} s")
        #    #print (f"Total Rejected Tokens: {total_rejected_tokens}")
        return output_tensor, tput, float(global_accepted_tokens) / float(generated_draft_tokens)
