import torch
from typing import Union, Optional
from .config import ModelConfig, InferenceConfig
from .model import get_model
from .initialize import init_distributed
from .utils import (
    print_rank0,
    get_gpu_memory_info,
    get_nvtx_funcs,
)
from .external.sampling import sample, sample_top_p
from .external.rejection_sampler import RejectionSampler
import torch.distributed as dist
from transformers import AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel
from .constants import EnginePhase
import time
import gc
from .timers import Timers

import os

# These flags are taken from the following URL -
# https://github.com/pytorch/pytorch/blob/347f96061f1cff603983b9be19ec92b374329a5b/benchmarks/gpt_fast/generate.py#L19

torch._inductor.config.coordinate_descent_tuning = True

torch._inductor.config.triton.unique_kernel_names = True

# Experimental feature to reduce compile times, will be on by default in future
torch._inductor.config.fx_graph_cache = True

torch._inductor.config.assert_indirect_indexing = False

torch._inductor.config.combo_kernel_foreach_dynamic_shapes = True


YALIS_DISABLE_COMPILE = os.environ.get("YALIS_DISABLE_COMPILE", "0") == "1"

YALIS_DECODE_MODE = (
    "default"
    if os.environ.get("YALIS_DISABLE_DECODE_CUDAGRAPHS", "0") == "1"
    else "reduce-overhead"
)

print(
    f"YALIS_DISABLE_COMPILE = {YALIS_DISABLE_COMPILE},"
    f"YALIS_DECODE_MODE = {YALIS_DECODE_MODE}"
)


precision_to_dtype = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@torch.inference_mode()
@torch.compile(disable=YALIS_DISABLE_COMPILE)
def prefill(
    model,
    tokens,
    unpadded_prompt_lengths,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    get_logits=False,
    phase: EnginePhase = EnginePhase.PREFILL,
):
    """
    Prefill function for generating the first token.

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.
        get_logits: If True, returns logits as well.

    Returns:
        token_id: The next predicted token.
        logits: (Optional) The raw logits from the model.
    """

    logits = model(tokens, phase, unpadded_prompt_lengths)["logits"].to(
        torch.float32
    )
    logits = logits[torch.arange(logits.size(0)), unpadded_prompt_lengths - 1]
    token_id = sample(
        logits=logits, temperature=temperature, top_k=top_k, top_p=top_p
    )
    # TODO: We should return a dict to support more return values in the future
    if get_logits:
        return token_id, logits
    else:
        return token_id, None


@torch.inference_mode()
@torch.compile(mode=YALIS_DECODE_MODE, disable=YALIS_DISABLE_COMPILE)
def generate(
    model,
    tokens,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    get_logits=False,
    phase: EnginePhase = EnginePhase.DECODE_SINGLE,
):
    """
    Generate function for producing the next token(s).

    Args:
        model: The model to generate from.
        tokens: Input tokens tensor.
        input_pos: Position indices for the tokens.
        get_logits: If True, returns logits as well.

    Returns:
        token_id: The next predicted token.
        logits: (Optional) The raw logits from the model.
    """
    logits = model(tokens, phase)["logits"].to(torch.float32)
    token_id = sample(
        logits=logits[:, -1], temperature=temperature, top_k=top_k, top_p=top_p
    )
    if get_logits:
        return token_id, logits[:, -1]
    else:
        return token_id, None


@torch.inference_mode()
@torch.compile(
    mode=YALIS_DECODE_MODE, disable=YALIS_DISABLE_COMPILE, fullgraph=True
)
def verify(
    model,
    tokens,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    phase: EnginePhase = EnginePhase.DECODE_MULTI,
):
    # Run the tokens through the model one-by-one
    logits = model(tokens, phase)["logits"].to(torch.float32)

    token_ids = sample(
        logits=logits, temperature=temperature, top_k=top_k, top_p=top_p
    )

    return token_ids, logits


class LLMEngine:
    """
    The core engine for managing and running inference on LLMs.
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
            model_config (ModelConfig): Config for model setup.
            inference_config (InferenceConfig): Config for inference behavior.
        """
        self.model = None  # Placeholder for the loaded model
        self.device = device
        self.dtype = precision_to_dtype[model_config.precision]
        init_distributed(tp_dims=inference_config.tp_dims)
        print_rank0(f"Model Config: {model_config}")
        print_rank0(f"Inference Config: {inference_config}")

        self.model, self.tokenizer = self._initialize_model(
            model_config, inference_config
        )
        self.model_config = model_config
        self.inference_config = inference_config

        # return extra memory to CUDA. Can prevent NCCL init OOMs
        torch.cuda.empty_cache()
        gc.collect()

        print_rank0(
            f"Memory Stats after Initializing Model - {get_gpu_memory_info()} "
        )

    def _make_params_contiguous(self, model):
        if not model:
            print_rank0(
                "Model must be initialized before contiguous parameter buffer can be allocated"  # noqa: E501
            )
            return model

        model = model.to(self.device)
        return model

    def _initialize_model(self, model_config, inference_config):
        """
        Internal method to load and set up the model based on ModelConfig.
        """
        t0 = time.time()
        model = get_model(
            model_config.model_path,
            self.dtype,
            max_sequence_length=inference_config.max_length,
            random_init=False,
            use_intra_head_parallelism=inference_config.use_intra_head_parallelism,  # noqa: E501
            attention_backend=inference_config.attention_backend,
            use_paged_kv_caching=inference_config.use_paged_kv_caching,
            prestore_kv_cache=inference_config.prestore_kv_cache,
            disable_tp=model_config.disable_tp,
        )
        model = self._make_params_contiguous(model)
        model.set_kv_cache(
            max_batch_size=inference_config.max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        print_rank0(
            f"Memory Stats after KV Cache Init - {get_gpu_memory_info()} "
        )
        if inference_config.symmetric_allreduce_strategy is not None:
            model.create_symmetric_memory_pool(
                batch_size=inference_config.batch_size,
                max_seq_length=inference_config.max_length,
                device=torch.device(torch.cuda.current_device()),
                dtype=self.dtype,
                algorithm=inference_config.symmetric_allreduce_strategy,
            )
        print_rank0(
            f"Memory Stats after Symm Pool Creation - {get_gpu_memory_info()} "
        )
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        # Check if the tokenizer has a pad token, otherwise use eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print_rank0(
                "Pad token not found in the tokenizer."
                "Using eos_token as pad token."
            )
        print_rank0(f"Initializing Model took {time.time() - t0} seconds")
        return model, tokenizer

    def reset_kv_cache(self, max_batch_size):
        self._reset_kv_cache(self.model, max_batch_size)

    def _reset_kv_cache(self, model, max_batch_size):
        if not model:
            print_rank0(
                "Model must be initialized before contiguous parameter buffer can be allocated"  # noqa: E501
            )
            return
        model.clear_kv_cache()
        model.set_kv_cache(
            max_batch_size=max_batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        if self.inference_config.symmetric_allreduce_strategy is not None:
            model.create_symmetric_memory_pool(
                max_batch_size=max_batch_size,
                max_seq_length=self.inference_config.max_length,
                device=torch.device(torch.cuda.current_device()),
                dtype=self.dtype,
                algorithm=self.inference_config.symmetric_allreduce_strategy,
            )

    def _tokenize_prompts(self, prompts):
        """Tokenize the input prompts and return tokens and seq lengths."""
        if isinstance(prompts, list) and all(
            isinstance(p, str) for p in prompts
        ):
            prompt_tokens_and_mask = self.tokenizer(
                prompts, return_tensors="pt", padding=True
            )
            prompt_tokens = prompt_tokens_and_mask.input_ids
            prompt_sequence_lengths = (
                prompt_tokens_and_mask.attention_mask.sum(dim=1)
            )
        elif isinstance(prompts, list) and all(
            isinstance(p, list) and all(isinstance(x, int) for x in p)
            for p in prompts
        ):
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
                "prompts must be a list of strings or a list of lists of ints"
            )
        return prompt_tokens, prompt_sequence_lengths

    def _validate_sequence_lengths(
        self, prompt_sequence_lengths, tokens_to_generate
    ):
        """Validate and adjust sequence lengths if necessary."""
        if prompt_sequence_lengths.max() > self.model.max_seq_length:
            raise ValueError(
                f"Prompt sequence length ({prompt_sequence_lengths.max()})"
                " exceeds model's maximum sequence length"
                f" ({self.model.max_seq_length}). Unable to proceed."
            )
        if (
            prompt_sequence_lengths.max() + tokens_to_generate
            > self.model.max_seq_length
        ):
            tokens_to_generate = (
                self.model.max_seq_length - prompt_sequence_lengths.max()
            )
            print_rank0(
                f"tokens_to_generate has been adjusted to {tokens_to_generate}"
            )
        return tokens_to_generate

    def _fake_tokens(self, batch_size, seq_length) -> torch.Tensor:
        """Create tokens with BOS on device with shape (bs, seqlen)."""
        bos = getattr(self.tokenizer, "bos_token_id", None)
        if bos is None:
            raise ValueError("BOS token is none.")
        return torch.full(
            (batch_size, seq_length),
            fill_value=int(bos),
            device=self.device,
            dtype=torch.long,
        )
    
    def _reset_warmup_states(self, batch_size):
        """Per-warmup request, reset token_counter and for paged kv cache, reset the manager."""
        # NOTE: doesn't account for spec dec
        if self.model.token_counter is not None:
            if batch_size is None:
                self.model.token_counter.zero_()
            else:
                self.model.token_counter[:batch_size].zero_()
        else:
            raise ValueError("Token counter is none.")

        if self.inference_config.use_paged_kv_caching:
            self.model.kv_cache_manager.reset()

    def warmup_prefill(
        self,
        batch_sizes,
        seq_lengths,
        iterations,
    ) -> None:
        """Warmup prefill by calling module level prefill path"""
        if batch_sizes is None or seq_lengths is None or iterations is None:
            raise ValueError("batch_sizes, seq_lengths, and iterations must be provided.")
        
        with torch.inference_mode(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            for bs in batch_sizes:
                for sl in seq_lengths:
                    print(f"Warmup prefill for batch size {bs} and sequence length {sl}")
                    for _ in range(iterations):
                        self._reset_warmup_states(bs)

                        tokens = self._fake_tokens(bs, sl)  # (bs, sl)
                        lens = torch.full(
                            (bs,),
                            fill_value=int(sl),
                            dtype=torch.long,
                            device=self.device,
                        )
                        _next_token, _ = prefill(
                            self.model,
                            tokens,
                            lens,
                            temperature=self.inference_config.temperature,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                        )
                        print(f"Warmup prefill for batch size {bs} and sequence length {sl} completed")
        torch.cuda.synchronize()

    def warmup_decode(
        self,
        batch_sizes,
        prompt_length,
        iterations,
    ) -> None:
        """Warmup decode by calling module level generate path"""
        with torch.inference_mode(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            for bs in batch_sizes:
                print(f"Warmup decode for batch size {bs} and prompt length {prompt_length}")
                self._reset_warmup_states(bs)

                # Tiny prefill
                tokens = self._fake_tokens(bs, int(prompt_length))
                lens = torch.full(
                    (bs,),
                    fill_value=int(prompt_length),
                    dtype=torch.long,
                    device=self.device,
                )
                next_token, _ = prefill(
                    self.model,
                    tokens,
                    lens,
                    temperature=self.inference_config.temperature,
                    top_k=self.inference_config.top_k,
                    top_p=self.inference_config.top_p,
                )

                current_input_to_model = next_token.clone()

                # Decode steps
                for _ in range(int(iterations)):
                    with sdpa_kernel(SDPBackend.MATH):
                        next_token, _ = generate(
                            self.model,
                            current_input_to_model,
                            temperature=self.inference_config.temperature,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                        )
                    current_input_to_model.copy_(next_token)
                print(f"Warmup decode for batch size {bs} and prompt length {prompt_length} completed")
        torch.cuda.synchronize()

    def warmup(
        self,
        prefill_batch_sizes,
        prefill_seq_lengths,
        decode_batch_sizes,
        decode_prompt_length=8,
        decode_iterations=2,
    ) -> None:
        """Warmup by calling prefill and decode"""
        print("Prefill warmup start.")
        prefill_start = time.perf_counter()
        self.warmup_prefill(
            batch_sizes=prefill_batch_sizes,
            seq_lengths=prefill_seq_lengths,
            iterations=1,
        )
        prefill_elapsed = time.perf_counter() - prefill_start
        print(f"Prefill warmup end. Elapsed {prefill_elapsed:.2f}s")
        print("Decode warmup start.")
        decode_start = time.perf_counter()
        self.warmup_decode(
            batch_sizes=decode_batch_sizes,
            prompt_length=decode_prompt_length,
            iterations=decode_iterations,
        )
        decode_elapsed = time.perf_counter() - decode_start
        print(f"Decode warmup end. Elapsed {decode_elapsed:.2f}s")
    
    def generate(
        self,
        prompts: Union[list[str], list[list[int]]],
        tokens_to_generate: int = 50,
        report_throughput: bool = False,
        ignore_eos: Optional[bool] = True,
        enable_nvtx: Optional[bool] = False,
        get_logits: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Generate tokens based on input prompts, which can either be
        a list of strings or a list of token ID lists.

        This method processes the provided prompts, either by tokenizing
        input strings or directly using tokenized inputs, and generates
        additional tokens based on the model's current state.

        Args:
            prompts (Union[list[str], list[list[int]]]): List of prompts
                - If `list[str]`, each string will be tokenized
                        into input IDs for the model.
                - If `list[list[int]]`, each sublist contains token IDs
                        for the model to process directly.
            tokens_to_generate (int, optional): Number of tokens to generate.
                Defaults to 50.
            report_throughput (bool, optional): A flag indicating whether to
                report throughput. Defaults to False.
            ignore_eos (bool, optional): Flag to ignore EOS.
            enable_nvtx (bool, optional): Flag to enable NVTX annotations
                around Prefill and Decode steps.
            get_logits (bool, optional): Flag to return logits.
                Defaults to False.

        Returns:
            output_tensor (torch.Tensor): Tensor containing generated tokens,
                with shape `(batch_size, tokens_to_generate)`.
            metrics (dict, optional): Dictionary containing metrics.
            output_logits (torch.Tensor, optional): Tensor containing logits.
        """
        timers = Timers()
        timers.start("tokenize")
        prompt_tokens, prompt_sequence_lengths = self._tokenize_prompts(
            prompts
        )
        tokens_to_generate = self._validate_sequence_lengths(
            prompt_sequence_lengths, tokens_to_generate
        )
        timers.stop("tokenize")
        print_rank0(
            f"Tokenization took {timers.get_times()[0][('tokenize',)]} ms"
        )

        batch_size = prompt_tokens.size(0)
        if not ignore_eos:
            done_mask = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )
        finished_reason = "Max Token Length"

        nvtx_range_push, nvtx_range_pop = get_nvtx_funcs(enable_nvtx)
        output_tokens = []
        output_logits = []
        # Start timing the operations
        timers.start("generate")
        self.model.token_counter.zero_()
        if self.inference_config.use_paged_kv_caching:
            self.model.kv_cache_manager.reset()
        with torch.inference_mode(), torch.autocast(
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
                    nvtx_range_push("Prefill")
                    next_token, logits = prefill(
                        self.model,
                        current_input_to_model,
                        prompt_sequence_lengths,
                        temperature=self.inference_config.temperature,
                        top_k=self.inference_config.top_k,
                        top_p=self.inference_config.top_p,
                        get_logits=get_logits,
                    )  # Call prefill function

                    current_input_to_model = next_token.clone()
                    nvtx_range_pop()
                else:  # Generation step
                    timer_key = "decode"
                    timers.start(timer_key)
                    nvtx_range_push("Decode")
                    with sdpa_kernel(SDPBackend.MATH):
                        next_token, logits = generate(
                            self.model,
                            current_input_to_model,
                            temperature=self.inference_config.temperature,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                            get_logits=get_logits,
                        )  # Call generate function

                    current_input_to_model.copy_(
                        next_token
                    )  # Copy the new token into tokens
                    nvtx_range_pop()

                # EOS Support:
                # Flatten to shape (batch_size,) for element wise comparison
                if not ignore_eos:
                    done_mask |= (
                        next_token.view(-1) == self.tokenizer.eos_token_id
                    )
                    # Reshape to match next_token's shape for masked_fill()
                    mask = done_mask.view(-1, 1)
                    # Force EOS prompts to stay EOS
                    next_token.masked_fill_(mask, self.tokenizer.eos_token_id)

                output_tokens.append(next_token.clone())
                if get_logits:
                    output_logits.append(logits.clone())
                timers.stop(timer_key)

                # Break if every sequence is done
                if not ignore_eos and done_mask.all():
                    finished_reason = "EOS"
                    break

        output_tensor = torch.cat(output_tokens, dim=1)
        # End timing and calculate elapsed time
        timers.stop("generate")
        times, events = timers.get_times()
        tput = (
            prompt_tokens.shape[0]
            * tokens_to_generate
            / (times[("generate",)] / 1000)
        )
        ttft = times[("generate", "prefill")] / events[("generate", "prefill")]
        if events[("generate", "decode")] > 0:
            tbt = (
                times[("generate", "decode")] / events[("generate", "decode")]
            )
        else:
            tbt = 0

        metrics = {
            "BatchSize": prompt_tokens.shape[0],
            "PromptLength": prompt_tokens.shape[1],
            "DecodeLength": tokens_to_generate,
            "Throughput": tput,
            "TTFT": ttft,
            "TBT": tbt,
            "E2E": times[("generate",)],
            "TokenizationTime": times[("tokenize",)],
            # NOTE: This should be a list containing reasons for each batch but
            # our EOS stopping currently is all-or-nothing.
            "FinishedReason": finished_reason,
        }
        if dist.get_rank() == 0 and report_throughput:
            print(
                f"[Metrics] BatchSize = {prompt_tokens.shape[0]},"
                f" PromptLength = {prompt_tokens.shape[1]},"
                f" DecodeLength = {tokens_to_generate},"
                f" Throughput = {tput:.2f} tok/s,"
                f" TTFT = {ttft:.4f} ms,"
                f" TBT = {tbt:.4f} ms,"
                f" E2E = {times[('generate',)]:.4f} ms,"
                f" FinishedReason = {finished_reason}"
            )

        if get_logits:
            return output_tensor, metrics, output_logits
        else:
            return output_tensor, metrics


class SpeculativeLLMEngine(LLMEngine):
    def __init__(
        self,
        target_model_config: ModelConfig,
        draft_model_config: ModelConfig,
        inference_config: InferenceConfig,
        device="cuda",
    ):
        if inference_config.use_paged_kv_caching:
            raise NotImplementedError(
                "Paged KV Caching is not supported for SpeculativeLLMEngine"
            )
        if inference_config.attention_backend not in ["flash", "sdpa"]:
            raise NotImplementedError(
                "Attention backend must be either flash or sdpa"
            )

        super().__init__(target_model_config, inference_config, device)
        print(f"Draft model config: {draft_model_config}")
        self.draft_model, _ = super()._initialize_model(
            draft_model_config, inference_config
        )
        self.draft_model_config = draft_model_config

        print_rank0(
            f"Memory Stats after Initializing Draft Model - "
            f"{get_gpu_memory_info()}"
        )

        self.sampler = RejectionSampler()

    def reset_kv_cache(self, batch_size):
        super().reset_kv_cache(batch_size)
        self._reset_kv_cache(self.draft_model, batch_size)

    # Logits to Probs
    @torch.inference_mode()
    def logits_to_probs(
        self, logits, token_ids, temperature: float = 1.0, top_p: float = 1.0
    ):
        if temperature > 0.0:
            logits = logits / max(temperature, 1e-5)
            if top_p > 0.0 and top_p < 1.0:
                logits = sample_top_p(logits, top_p)
            probs = torch.nn.functional.softmax(logits, dim=-1)
        else:
            probs = torch.zeros_like(logits)
            probs.scatter_(-1, token_ids, 1.0)

        return probs

    def generate_speculative(
        self,
        input_tokens: torch.Tensor,
        tokens_to_generate: int,
        gamma: int,
        report_throughput: bool = False,
        ignore_eos: bool = True,
        enable_nvtx: bool = False,
    ):

        assert (
            tokens_to_generate > 0
        ), "Tokens to generate must be greater than 0"
        assert gamma > 0, "Gamma must be greater than 0"

        timers = Timers()

        timers.start("tokenize")
        prompt_tokens, prompt_sequence_lengths = self._tokenize_prompts(
            input_tokens
        )
        tokens_to_generate = self._validate_sequence_lengths(
            prompt_sequence_lengths, tokens_to_generate
        )
        timers.stop("tokenize")

        batch_size = prompt_tokens.size(0)
        done_mask = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        finished_reason = "Max Token Length"

        nvtx_range_push, nvtx_range_pop = get_nvtx_funcs(enable_nvtx)

        output_tokens = [[] for _ in range(batch_size)]

        global_accepted_tokens = 0
        generated_draft_tokens = 0

        timers.start("generate")
        self.model.token_counter.zero_()
        self.draft_model.token_counter.zero_()

        with torch.inference_mode(), torch.autocast(
            self.device, dtype=self.dtype, cache_enabled=False
        ):
            current_input_to_model = prompt_tokens.clone().to(
                self.device
            )  # Move prompt tokens to the device

            prompt_sequence_lengths = prompt_sequence_lengths.to(self.device)

            # Number of tokens generated per batch can be different,
            # so we need to keep track of the tokens generated per batch
            # Currently, we stop when all the batches have generated the
            # required number of tokens. Might be sub-optimal
            generated_tokens = torch.zeros(
                batch_size, dtype=torch.int64, device=self.device
            )

            step = 0
            while generated_tokens.min() < tokens_to_generate:
                timer_key = None
                step += 1
                if step == 1:  # Prefill step
                    timer_key = "prefill"
                    timers.start(timer_key)
                    nvtx_range_push("Prefill")

                    # Calling prefill on draft model but not using its output
                    draft_next_token, _ = prefill(
                        self.draft_model,
                        current_input_to_model,
                        prompt_sequence_lengths,
                        temperature=self.inference_config.temperature,
                        top_k=self.inference_config.top_k,
                        top_p=self.inference_config.top_p,
                    )

                    # Calling prefill on the target model
                    next_token, _ = prefill(
                        self.model,
                        current_input_to_model,
                        prompt_sequence_lengths,
                        temperature=self.inference_config.temperature,
                        top_k=self.inference_config.top_k,
                        top_p=self.inference_config.top_p,
                    )

                    current_input_to_model = next_token.clone()
                    accepted_tokens = next_token
                    accepted_mask = torch.ones_like(
                        next_token, dtype=torch.bool, device=self.device
                    )
                else:  # Generation step
                    timer_key = "decode"
                    timers.start(timer_key)
                    nvtx_range_push("Decode")

                    # Draft tokens
                    current_input_to_draft_model = (
                        current_input_to_model.clone()
                    )
                    draft_output_tokens = []

                    # Adding the input tokens to the draft output tokens
                    # This is needed in the Rejection Sampling step, we will
                    # ignore this token when we store the final output tokens
                    draft_output_tokens.append(
                        current_input_to_draft_model.clone()
                    )
                    draft_probs = []

                    with sdpa_kernel(SDPBackend.MATH):
                        timers.start("draft_decode")
                        for draft_step in range(gamma):
                            next_token, logits = generate(
                                self.draft_model,
                                current_input_to_draft_model,
                                top_k=self.inference_config.top_k,
                                top_p=self.inference_config.top_p,
                                temperature=self.inference_config.temperature,
                                get_logits=True,
                            )
                            current_input_to_draft_model.copy_(next_token)

                            # TODO: Need to take top_k into account as well
                            probs = self.logits_to_probs(
                                logits,
                                token_ids=next_token,
                                temperature=self.inference_config.temperature,
                                top_p=self.inference_config.top_p,
                            )

                            draft_probs.append(probs.unsqueeze(1).clone())
                            draft_output_tokens.append(next_token.clone())

                        # Needed to add the last token to the draft model's
                        # KV cache because the verify function will generate
                        # one extra bonus token that will be the input to the
                        # next step
                        _ = generate(
                            self.draft_model,
                            current_input_to_draft_model,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                            temperature=self.inference_config.temperature,
                        )

                        draft_probs = torch.cat(draft_probs, dim=1)
                        draft_output_tokens = torch.cat(
                            draft_output_tokens, dim=1
                        )
                        timers.stop("draft_decode")

                        timers.start("verify")
                        # print(f"[{dist.get_rank()}] Verify Step {step}")
                        next_token, target_logits = verify(
                            self.model,
                            draft_output_tokens,
                            top_k=self.inference_config.top_k,
                            top_p=self.inference_config.top_p,
                            temperature=self.inference_config.temperature,
                        )
                        target_probs = self.logits_to_probs(
                            target_logits,
                            token_ids=next_token,
                            temperature=self.inference_config.temperature,
                            top_p=self.inference_config.top_p,
                        )
                        timers.stop("verify")

                    output_with_bonus_tokens = self.sampler(
                        draft_probs,
                        target_probs,
                        draft_output_tokens[:, 1:],
                        next_token[:, -1],
                    )
                    accepted_mask = output_with_bonus_tokens != -1
                    accepted_tokens = output_with_bonus_tokens

                    # Do not accept any tokens if done
                    accepted_mask = accepted_mask & (~done_mask.unsqueeze(-1))
                    num_accepted_tokens = accepted_mask.sum(dim=-1)
                    num_rejected_tokens = (
                        draft_output_tokens.size(-1) - num_accepted_tokens
                    )

                    # Update model's KV cache
                    # Need to handle the case where a certain batch is done
                    self.model.rewind_kv_cache(num_rejected_tokens)
                    self.draft_model.rewind_kv_cache(num_rejected_tokens)

                    # If no tokens are accepted, use the first token as input
                    # This only occurs for finished requests; the token isn’t
                    # added to KV-cache or output, so it’s safe.
                    current_input_to_model.copy_(
                        torch.gather(
                            output_with_bonus_tokens,
                            -1,
                            torch.clamp(
                                num_accepted_tokens.unsqueeze(-1) - 1, min=0
                            ),
                        )
                    )

                    global_accepted_tokens += (
                        torch.clamp(num_accepted_tokens - 1, min=0)
                        .sum()
                        .item()
                    )
                    # Count draft tokens only for unfinished sequences
                    generated_draft_tokens += (gamma * ~done_mask).sum().item()

                if not ignore_eos:
                    # If EOS is found or already done, replace subsequent
                    # tokens with EOS. Also mark the sequence as done
                    eos_mask = (
                        accepted_tokens == self.tokenizer.eos_token_id
                    ).cumsum(dim=-1) > 0 | (done_mask.unsqueeze(-1))
                    done_mask = done_mask | eos_mask.any(dim=-1)
                    accepted_tokens.masked_fill_(
                        eos_mask, self.tokenizer.eos_token_id
                    )

                # Limit the accepted tokens to tokens_to_generate
                limit_mask = (
                    generated_tokens.view(-1, 1)
                    + torch.arange(
                        accepted_tokens.size(-1),
                        device=generated_tokens.device,
                    ).view(1, -1)
                ) < tokens_to_generate
                keep_mask = limit_mask & accepted_mask

                # Select the accepted tokens within limit into a jagged list
                jagged_output_tokens = torch.nested.masked_select(
                    accepted_tokens, keep_mask
                ).unbind()
                for i in range(batch_size):
                    output_tokens[i].append(jagged_output_tokens[i])
                    generated_tokens[i] += jagged_output_tokens[i].size(0)

                    assert generated_tokens[i] <= tokens_to_generate, (
                        f"Generated tokens {generated_tokens[i]} is "
                        f"greater than tokens to generate {tokens_to_generate}"
                    )

                # Update done mask is sequence has finished generating
                done_mask = done_mask | (
                    generated_tokens >= tokens_to_generate
                )

                nvtx_range_pop()
                timers.stop(timer_key)

                # Break if every sequence is done
                if not ignore_eos and done_mask.all():
                    finished_reason = "EOS"
                    break

        output_tokens = [torch.cat(tokens, dim=0) for tokens in output_tokens]

        # If EOS is not ignored, sequences may differ in length; pad to the
        # longest in the batch
        if not ignore_eos:
            output_token_lengths = [
                tokens.shape[0] for tokens in output_tokens
            ]
            max_length = max(output_token_lengths)
            output_tokens = [
                torch.nn.functional.pad(
                    tokens,
                    (0, max_length - tokens.shape[0]),
                    "constant",
                    self.tokenizer.pad_token_id,
                )
                for tokens in output_tokens
            ]

        output_tensor = torch.stack(output_tokens, dim=0)

        timers.stop("generate")
        times, events = timers.get_times()

        tput = (
            prompt_tokens.shape[0]
            * tokens_to_generate
            / (times[("generate",)] / 1000)
        )
        ttft = times[("generate", "prefill")] / events[("generate", "prefill")]
        if events[("generate", "decode")] > 0:
            tbs = (
                times[("generate", "decode")] / events[("generate", "decode")]
            )
            tbs_draft = (
                times[("generate", "decode", "draft_decode")]
                / events[("generate", "decode", "draft_decode")]
            )
            tbs_verify = (
                times[("generate", "decode", "verify")]
                / events[("generate", "decode", "verify")]
            )
        else:
            tbs = 0
            tbs_draft = 0
            tbs_verify = 0
        acceptance_rate = global_accepted_tokens / generated_draft_tokens

        metrics = {
            "BatchSize": prompt_tokens.shape[0],
            "PromptLength": prompt_tokens.shape[1],
            "DecodeLength": tokens_to_generate,
            "Throughput": tput,
            "TTFT": ttft,
            "TBS": tbs,
            "TBS (Draft)": tbs_draft,
            "TBS (Verify)": tbs_verify,
            "E2E": times[("generate",)],
            "TokenizationTime": times[("tokenize",)],
            "AcceptanceRate": acceptance_rate,
            # NOTE: This should be a list containing reasons for each batch but
            # our EOS stopping currently is all-or-nothing.
            "FinishedReason": finished_reason,
        }
        if dist.get_rank() == 0 and report_throughput:
            print(
                f"[Metrics] BatchSize = {prompt_tokens.shape[0]},"
                f" PromptLength = {prompt_tokens.shape[1]},"
                f" DecodeLength = {tokens_to_generate},"
                f" Throughput = {tput:.2f} tok/s,"
                f" TTFT = {ttft:.4f} ms,"
                f" TBS = {tbs:.4f} ms,"
                f" TBS (Draft) = {tbs_draft:.4f} ms,"
                f" TBS (Verify) = {tbs_verify:.4f} ms,"
                f" AcceptanceRate = {acceptance_rate:.4f},"
                f" E2E = {times[('generate',)]:.4f} ms,"
                f" FinishedReason = {finished_reason}"
            )
        return output_tensor, metrics
