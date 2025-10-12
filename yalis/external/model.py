# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.  # noqa: E501

"""
Full definition of a decoder-only transformer language model, all in this file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

from typing import Any, Optional, Tuple
from typing_extensions import Self
from copy import deepcopy
import warnings
import sys

import torch
import torch.nn as nn
from axonn import axonn as ax
from axonn.intra_layer.communication import Drop, Gather

from yalis.attention import attention_wrapper
from yalis.external.config import Config
from yalis.tensor_parallel import TPLinear
from yalis.constants import EnginePhase
from kvcache_manager import KVCacheManager
from yalis.attention.flash import flash_apply_rotary as apply_rotary
from yalis.attention.backends import AttentionBackend
from yalis.attention.masking import create_causal_block_mask_for_flex_attention


# TODO: these should be dynamically set during engine initialization
NUM_BLOCKS, PAGE_BLOCK_SIZE = 512, 256


# switch sequential norm classes to TP norm classes if needed
def get_norm_class(config):
    if not config.tensor_parallel or ax.config.G_intra_c == 1:
        # if not tensor parallel then no need to use tensor parallel norms
        # if tensor parallel and not using column TP then again
        # no need to use TP norms
        return config.norm_class
    from yalis.tensor_parallel import TPRMSNorm

    if config.norm_class_name == "RMSNorm":
        return TPRMSNorm
    else:
        raise NotImplementedError(
            f"TP version of {config.norm_class_name} not implemented"
        )


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:

        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.lm_head_bias
        )
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(
                    Block(config, block_idx)
                    for block_idx in range(config.n_layer)
                ),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.max_seq_length = (
            self.config.block_size
        )  # rope cache is built here
        self.symmetric_memory_pool = None

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than
        the model's context length. This allows setting a smaller number
        to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(
                f"Cannot attend to {value}, block size is only {self.config.block_size}."  # noqa: E501
                " This is likely because the input text exceeds the supported context length of this model."  # noqa: E501
            )
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            cos, sin = self.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`.
        # we cannot update it here as we don't know if the kv cache is expected

    def reset_parameters(self) -> None:
        # Trigger resetting the rope-cache
        self.cos, self.sin = self.rope_cache(device=self.cos.device)

    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        phase: EnginePhase,
        actual_sequence_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        idx = input_ids
        T = idx.size(1)
        if self.max_seq_length < T:
            raise ValueError(
                f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}."  # noqa: E501
            )

        # Update block table
        # assign new pages to each sequence if needed to store new keys/values
        # actual storage will be done by the flash attention kernel.
        # this is just assigning pages to each sequence
        if self.config.use_paged_kv_caching:
            # create pages for T new tokens if needed.
            # Note that T includes padding tokens in prefill.
            # we will readjust the token counters of the block table
            # at the end to exclude padded tokens.
            B = input_ids.shape[0]
            seq_lengths = torch.full(
                (B,),
                T,
                dtype=torch.int64,
                device=self.kvcache_block_table.device,
            )
            torch.ops.yalis.update_block_table_(
                self.kvcache_block_table,
                self.tokens_assigned,
                self.kvcache_next_page,
                self.kvcache_free_pages,
                seq_lengths,
                PAGE_BLOCK_SIZE,
                16384 // PAGE_BLOCK_SIZE,
            )

        x = self.transformer.wte(
            idx
        )  # token embeddings of shape (b, t, n_embd)
        if self.config.scale_embeddings:
            x = x * torch.tensor(self.config.n_embd**0.5, dtype=x.dtype)
        if self.config.tensor_parallel:
            x = Drop.apply(x, ax.comm_handle.inner_intra_layer_parallel_group)

        # flash attention wants the rope cache to be
        # in the same dtype as the query
        # ToDO: confirm if this is okay, or if we should do rope in fp32?
        if self.config.attention_backend == AttentionBackend.FLASH:
            self.cos = self.cos.to(x.dtype)
            self.sin = self.sin.to(x.dtype)

        block_table = (
            self.kvcache_block_table
            if self.config.use_paged_kv_caching
            else None
        )

        B = x.size(0)

        flex_attention_block_mask = (
            create_causal_block_mask_for_flex_attention(
                self.token_counter, self.kv_length, B
            )
            if self.config.attention_backend == AttentionBackend.FLEX
            else None
        )

        for block in self.transformer.h:
            x = block(
                x,
                self.cos,
                self.sin,
                phase,
                self.token_counter,
                block_table,
                flex_attention_block_mask,
            )
        if self.config.tensor_parallel:
            x = Gather.apply(
                x, ax.comm_handle.inner_intra_layer_parallel_group
            )
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)  # (b, t, vocab_size)
        if self.config.final_logit_softcapping is not None:
            x = (
                torch.tanh(x / self.config.final_logit_softcapping)
                * self.config.final_logit_softcapping
            )
        self.token_counter[:B].add_(
            T if actual_sequence_lengths is None else actual_sequence_lengths
        )
        if self.config.use_paged_kv_caching:
            # NOTE: Paged KV: readjusting the token counters of the block table
            # to exclude padded tokens.
            # we can exclude this for generation
            torch.ops.yalis.force_update_tokens_assigned_(
                self.tokens_assigned[:B], self.token_counter[:B]
            )
        return {"logits": x}

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def rope_cache(
        self, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.config.rope_adjustments is None:
            extra_config = None

        else:
            adjusted_params_required = [
                "factor",
                "low_freq_factor",
                "high_freq_factor",
                "original_max_seq_len",
            ]
            params_present = [
                param in self.config.rope_adjustments
                for param in adjusted_params_required
            ]
            num_params_present = sum(params_present)

            if num_params_present == 0:
                extra_config = None  # uses standard RoPE
            elif num_params_present == 4:
                # These parameters should always be used together so that
                # we don't interfere with standard rope
                extra_config = {
                    "original_max_seq_len": self.config.rope_adjustments[
                        "original_max_seq_len"
                    ],
                    "factor": self.config.rope_adjustments["factor"],
                    "low_freq_factor": self.config.rope_adjustments[
                        "low_freq_factor"
                    ],
                    "high_freq_factor": self.config.rope_adjustments[
                        "high_freq_factor"
                    ],
                }
            else:
                # Some but not all parameters are specified; raise an error
                missing_params = [
                    param
                    for param, present in zip(
                        adjusted_params_required, params_present
                    )
                    if not present
                ]
                raise ValueError(
                    f"The following adjusted RoPE parameters are missing in rope_adjustments: {', '.join(missing_params)}. "  # noqa: E501
                    "All adjusted RoPE parameters must be specified together."
                )

        return build_rope_cache(
            seq_len=self.max_seq_length,
            n_elem=self.config.rope_n_elem,
            device=device,
            condense_ratio=self.config.rope_condense_ratio,
            base=self.config.rope_base,
            extra_config=extra_config,
            is_attention_backend_flash=(
                self.config.attention_backend == AttentionBackend.FLASH
            ),
        )

    def set_kv_cache(
        self,
        max_batch_size: int,
        max_seq_length: Optional[int] = None,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
            if self.config.attention_backend == AttentionBackend.FLASH:
                rope_cache_length *= 2

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        self.kv_length = max_seq_length
        self.max_batch_size = max_batch_size

        max_tokens = max_seq_length * batch_size

        # TODO (Prajwal): This is a hack to not over allocated
        # KV-cache by default.Fix with dynamic page calculation logic
        global NUM_BLOCKS
        if self.config.use_paged_kv_caching:
            if max_tokens > PAGE_BLOCK_SIZE * NUM_BLOCKS:
                print("Increasing NUM_BLOCKS to 1024")
                NUM_BLOCKS = 1024

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                max_batch_size,
                max_seq_length,
                rope_cache_length,
                device,
                dtype,
            )
        if self.config.use_paged_kv_caching:
            self.kv_cache_manager = KVCacheManager(
                max_batch_size,
                16384 // PAGE_BLOCK_SIZE,  # ToDo: set these dynamically
                NUM_BLOCKS,
                PAGE_BLOCK_SIZE,
            )
            # TODO: move to separate Python class
            self.tokens_assigned = (
                self.kv_cache_manager.tokens_assigned_tensor()
            )
            self.kvcache_block_table = self.kv_cache_manager.block_table()
            self.kvcache_free_pages = self.kv_cache_manager.free_pages_tensor()
            self.kvcache_next_page = self.kv_cache_manager.next_page_tensor()

        self.token_counter = torch.zeros(
            max_batch_size, device=device, dtype=torch.int32
        )

    def rewind_kv_cache(self, num_tokens: torch.Tensor) -> None:
        """
        Rewind the token counter and KV-cache by the num_tokens.
        Used when rejecting tokens during speculative decoding.
        """
        B = num_tokens.size(0)
        self.token_counter[:B] -= num_tokens

    def clear_kv_cache(self) -> None:
        for block in self.transformer.h:
            block.attn.kv_cache = None
        torch.cuda.empty_cache()

    def create_symmetric_memory_pool(
        self,
        batch_size: int,
        max_seq_length: int,
        device: torch.device,
        dtype: torch.dtype,
        algorithm: str,
    ) -> None:
        """
        This function is used to create a cache of symmetric
        memory tensors within each TP Layer to be used for
        low-latency all-reduce
        """

        self.symmetric_memory_pool = {}

        def _update_symmetric_memory_pool(module):
            if isinstance(module, TPLinear):
                module.set_symmetric_memory_tensor(
                    batch_size,
                    max_seq_length,
                    dtype,
                    device,
                    self.symmetric_memory_pool,
                    algorithm,
                )

        self.transformer.apply(_update_symmetric_memory_pool)

        if len(self.symmetric_memory_pool) == 0:
            warnings.warn(
                "No tensor parallel groups found within the same node."
                "Disabling symmetric memory allreduce"
            )
            self.symmetric_memory_pool = None


class Block(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        if not config.parallel_residual and config.shared_attention_norm:
            raise NotImplementedError(
                "No supported checkpoint uses this configuration"
                " (non-parallel residual and shared attention norm)."
            )

        self.norm_1 = get_norm_class(config)(
            config.n_embd, eps=config.norm_eps
        )
        self.attn = CausalSelfAttention(config, block_idx)
        self.post_attention_norm = (
            get_norm_class(config)(config.n_embd, eps=config.norm_eps)
            if config.post_attention_norm
            else nn.Identity()
        )
        self.norm_2 = (
            None
            if config.shared_attention_norm
            else get_norm_class(config)(config.n_embd, eps=config.norm_eps)
        )
        mlp_class = getattr(sys.modules[__name__], config.mlp_class_name)
        self.mlp = mlp_class(config)
        self.post_mlp_norm = (
            get_norm_class(config)(config.n_embd, eps=config.norm_eps)
            if config.post_mlp_norm
            else nn.Identity()
        )

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        phase: EnginePhase,
        token_counter: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        flex_attention_block_mask=None,
    ) -> torch.Tensor:
        """
        Non-parallel residual       Parallel residual
           ┌─ x                     ┌─ x ──────────────────┐             Note: if `shared_attention_norm` is True,  # noqa: E501
           │  ↓                     │  ↓                   ↓                   the output from `norm_1` is reused   # noqa: E501
           │  norm_1                │  norm_1  ───────►    norm_2
           │  ↓                     │  ↓                   ↓
           │  attn                  │  attn                MLP
           │  ↓                     │  ↓                   ↓
           |  post_attn_norm        |  post_attn_norm      post_mlp_norm
           |  ↓                     |  ↓                   ↓
        ┌─ └► +                     └► + ◄─────────────────┘
        |     ↓
        │     norm_2
        │     ↓
        │     MLP
        │     ↓
        |     post_mlp_norm
        |     ↓
        └───► +
        """

        x_normed = self.norm_1(x)
        attention_output = self.attn(
            x_normed,
            cos,
            sin,
            phase,
            token_counter,
            block_table,
            flex_attention_block_mask,
        )
        attention_output = self.post_attention_norm(attention_output)

        # Currently, MLP does not need to be phase-aware
        # but we might add it in the future
        if self.config.parallel_residual:
            x_normed = (
                x_normed
                if self.config.shared_attention_norm
                else self.norm_2(x)
            )
            x = self.mlp(x_normed) + attention_output + x
        else:
            x = attention_output + x
            x = self.post_mlp_norm(self.mlp(self.norm_2(x))) + x
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config, block_idx: int) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        if not config.tensor_parallel:
            self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        else:
            self.attn = TPLinear(
                config.n_embd,
                shape,
                bias=config.bias,
                init_device=config.init_device,
            )

        # output projection
        # if `head_size` is explicitly specified in the config,
        # `n_embd` might not be equal to `head_size * n_head`
        if not config.tensor_parallel:
            self.proj = nn.Linear(
                config.head_size * config.n_head,
                config.n_embd,
                bias=config.bias,
            )
        else:
            self.proj = TPLinear(
                config.head_size * config.n_head,
                config.n_embd,
                bias=config.bias,
                transpose=True,
                init_device=config.init_device,
            )
        # disabled by default
        self.kv_cache: Optional[KVCache] = None
        self.apply_sliding_window_attention = (
            config.sliding_window_size is not None
            and block_idx % config.sliding_window_layer_placing == 0
        )

        self.config = config
        if config.tensor_parallel:
            # dividing attention heads over the row tensor parallel group
            # currently attention is duplicated across the column TP group
            self.config = deepcopy(self.config)
            attention_world_size = ax.config.G_intra_r
            self.duplicating_kv = attention_world_size > config.n_query_groups
            if self.duplicating_kv:
                assert attention_world_size % config.n_query_groups == 0
                self.duplication_degree = (
                    attention_world_size // config.n_query_groups
                )
            else:
                self.duplication_degree = 1
            assert self.config.n_head % attention_world_size == 0
            # storing number of global heads in the entire model
            self.total_n_head = self.config.n_head
            self.total_n_query_groups = self.config.n_query_groups
            # q per rank
            self.config.n_head //= attention_world_size
            if self.duplicating_kv:
                self.config.n_query_groups = 1
                self.attn.duplicating_kv = True
                self.attn.total_n_head = self.total_n_head
                self.attn.total_n_query_groups = self.total_n_query_groups
                self.attn.duplication_degree = self.duplication_degree
                self.attn.head_size = self.config.head_size
                self.attn.q_per_rank = self.config.n_head
            else:
                assert self.config.n_query_groups % attention_world_size == 0
                self.config.n_query_groups //= attention_world_size

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        phase: EnginePhase,
        token_counter: torch.Tensor,
        block_table: torch.Tensor = None,
        flex_attention_block_mask=None,
    ) -> torch.Tensor:
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support
        # MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = (
            q_per_kv + 2
        )  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=3)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_k, hs)
        v = v.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_v, hs)

        assert (
            self.config.rope_n_elem == self.config.head_size
        ), "partial rope is not supported yet"
        k_cache, v_cache = self.kv_cache.k, self.kv_cache.v
        if self.config.attention_backend == AttentionBackend.FLASH:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            q = apply_rotary(q, cos, sin, token_counter)
            k = apply_rotary(k, cos, sin, token_counter)

            cos, sin = None, None
        else:
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()

        # NOTE: Pass full k_cache, v_cache, and token_counter.
        # Slicing for current batch size is done in the respective backends.
        y = attention_wrapper(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            k=k,
            v=v,
            phase=phase,
            cache_seqlens=token_counter,
            block_table=block_table,
            rotary_cos=cos,
            rotary_sin=sin,
            backend=self.config.attention_backend,
            use_intra_head_parallelism=self.config.use_intra_head_parallelism,
            prestore_kv_cache=self.config.prestore_kv_cache,
            flex_attention_block_mask=flex_attention_block_mask,
        )

        if not self.config.attention_backend == AttentionBackend.FLASH:
            y = y.transpose(1, 2).contiguous()

        y = y.reshape(
            B, T, self.config.head_size * self.config.n_head
        )  # re-assemble all head outputs side by side

        # output projection
        return self.proj(y)

    def build_kv_cache(
        self,
        batch_size: int,
        max_seq_length: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "KVCache":

        heads = self.config.n_query_groups
        if self.config.attention_backend == AttentionBackend.FLASH:
            if self.config.use_paged_kv_caching:
                v_shape = (
                    NUM_BLOCKS,
                    PAGE_BLOCK_SIZE,
                    heads,
                    self.config.head_size,
                )
            else:
                v_shape = (
                    batch_size,
                    max_seq_length,
                    heads,
                    self.config.head_size,
                )

        else:
            v_shape = (
                batch_size,
                heads,
                max_seq_length,
                self.config.head_size,
            )

        if rope_cache_length is None:
            if self.config.rotary_percentage != 1.0:
                raise TypeError(
                    "Please pass the `rope_cache_length=gpt.cos.size(-1)` value"  # noqa: E501
                )
            k_shape = v_shape
        else:
            if self.config.attention_backend == AttentionBackend.FLASH:
                if self.config.use_paged_kv_caching:
                    k_shape = (
                        NUM_BLOCKS,
                        PAGE_BLOCK_SIZE,
                        heads,
                        rope_cache_length
                        + self.config.head_size
                        - self.config.rope_n_elem,
                    )
                else:
                    k_shape = (
                        batch_size,
                        max_seq_length,
                        heads,
                        rope_cache_length
                        + self.config.head_size
                        - self.config.rope_n_elem,
                    )
            else:
                k_shape = (
                    batch_size,
                    heads,
                    max_seq_length,
                    rope_cache_length
                    + self.config.head_size
                    - self.config.rope_n_elem,
                )

        if self.config.use_intra_head_parallelism:
            assert k_shape[-1] % ax.config.G_intra_c == 0
            k_shape = k_shape[:-1] + (k_shape[-1] // ax.config.G_intra_c,)

            assert v_shape[-1] % ax.config.G_intra_c == 0
            v_shape = v_shape[:-1] + (v_shape[-1] // ax.config.G_intra_c,)
        return KVCache(k_shape, v_shape, device=device, dtype=dtype)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert not config.tensor_parallel
        self.fc = nn.Linear(
            config.n_embd, config.intermediate_size, bias=config.bias
        )
        self.proj = nn.Linear(
            config.intermediate_size, config.n_embd, bias=config.bias
        )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(
            x, approximate=self.config.gelu_approximate
        )
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if not config.tensor_parallel:
            self.gate_up_proj = nn.Linear(
                config.n_embd, 2 * config.intermediate_size, bias=config.bias
            )
            self.proj = nn.Linear(
                config.intermediate_size, config.n_embd, bias=config.bias
            )
        else:
            self.gate_up_proj = TPLinear(
                config.n_embd,
                2 * config.intermediate_size,
                bias=config.bias,
                init_device=config.init_device,
            )
            self.proj = TPLinear(
                config.intermediate_size,
                config.n_embd,
                bias=config.bias,
                transpose=True,
                init_device=config.init_device,
            )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x_fc_1, x_fc_2 = x[..., ::2], x[..., 1::2]
        x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        return self.proj(x)


class GemmaMLP(LLaMAMLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gate_up_proj(x)
        x_fc_1, x_fc_2 = x[..., ::2], x[..., 1::2]
        x = (
            torch.nn.functional.gelu(
                x_fc_1, approximate=self.config.gelu_approximate
            )
            * x_fc_2
        )
        return self.proj(x)


class LLaMAMoE(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert not config.tensor_parallel
        self.gate = nn.Linear(config.n_embd, config.n_expert, bias=False)
        self.experts = nn.ModuleList(
            LLaMAMLP(config) for _ in range(config.n_expert)
        )

        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derived from: https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219  # noqa: E501
        See also figure 1 in https://arxiv.org/abs/2211.15841
        """
        B, T, C = (
            x.size()
        )  # batch size, sequence length, embedding dimensionality (n_embd)
        x = x.view(-1, C)  # (B*T, C)
        router = self.gate(x)  # (B*T, n_expert)
        probs, indices = torch.topk(
            router, self.config.n_expert_per_token
        )  # (B*T, n_expert_per_token)
        probs = probs.softmax(dim=1, dtype=torch.float).to(dtype=x.dtype)
        masks = indices.unsqueeze(-1) == torch.arange(
            self.config.n_expert, device=x.device
        )
        masks = masks.permute(2, 0, 1)  # (n_expert, B*T, n_expert_per_token)
        y = torch.zeros_like(x)  # (B*T, C)
        for mask, expert in zip(masks, self.experts):
            token_idx, expert_idx = torch.where(mask)
            y[token_idx] += probs[token_idx, expert_idx, None] * expert(
                x[token_idx]
            )
        return y.view(B, T, C)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    device: Optional[torch.device] = None,
    base: int = 10000,
    condense_ratio: int = 1,
    extra_config: Optional[dict] = None,
    is_attention_backend_flash: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enhanced Transformer with Rotary Position Embedding.

    Args:
        seq_len (int): Sequence length.
        n_elem (int): Number of elements (head dimension).
        device (torch.device, optional): Device for tensor allocations.
        base (int, optional): Base for computing inverse frequencies.
        condense_ratio (int, optional): Ratio to condense the position indices.
        extra_config (dict, optional): Configuration parameters for frequency
                                        adjustments (used by Llama 3.1 and 3.2)
        is_attention_backend_flash (bool, optional): If using flash attention
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Cosine and sine caches for RoPE.
    """

    # Compute the inverse frequencies theta
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem)
    )

    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]
        factor = extra_config["factor"]
        low_freq_factor = extra_config["low_freq_factor"]
        high_freq_factor = extra_config["high_freq_factor"]

        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        smooth_factor = (ratio - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)

        # Compute adjusted_theta without masked indexing
        adjusted_theta = (1 - smooth_factor) * (
            theta / factor
        ) + smooth_factor * theta
        theta = adjusted_theta

    # Create position indices `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(
        seq_idx, theta
    )  # .repeat(1, 2) repeat is not needed for flash attention

    if not is_attention_backend_flash:
        idx_theta = idx_theta.repeat(1, 2)

    return torch.cos(idx_theta), torch.sin(idx_theta)


def batched_index_select(t, dim, idx):
    """index_select for batched index and unbatched t"""
    if idx.dim() == 1:
        return torch.index_select(t, dim, idx)

    *batch_shape, idx_size = idx.shape
    res = torch.index_select(t, dim, idx.reshape(-1))  # flat index
    # split out single batch idx
    res = res.view(*t.shape[:dim], -1, idx_size, *t.shape[dim + 1 :])
    # move batch dim to front, this is np.rollaxis(res, dim, 0) for tensors
    dims = [dim] + list(range(res.dim()))
    # del dims[dim + 1]
    dims = dims[: dim + 1] + dims[dim + 2 :]
    res = res.permute(dims)
    # unflatten batch dims
    res = res.view(*batch_shape, *res.shape[1:])
    return res


def batched_index_copy_(t, dim, idx, val):
    """Index copy for batched t, idx, val"""

    if t.device.type == "mps":
        # Normalize negative dimensions
        if dim < 0:
            dim = t.dim() + dim
        if idx.dim() == 1:
            idx_shape = [1] * val.dim()
            idx_shape[dim] = -1
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)
            t.scatter_(dim, idx_expanded, val)
            return t

        elif idx.dim() == 2:
            assert dim != 0, "Cannot index the batch dimension"
            batch_size = idx.size(0)
            idx_size = idx.size(1)
            assert batch_size == t.size(0) == val.size(0)

            idx_shape = [batch_size] + [1] * (val.dim() - 1)
            idx_shape[dim] = idx_size
            idx_expanded = idx.view(*idx_shape)
            idx_expanded = idx_expanded.expand_as(val)

            t.scatter_(dim, idx_expanded, val)
            return t
        else:
            raise NotImplementedError(
                f"idx.dim() == {idx.dim()} not supported"
            )

    else:
        if idx.dim() == 1:
            return t.index_copy_(dim, idx, val)

        assert idx.dim() == 2, f"multiple batch dims not yet {idx.shape=}"
        assert dim != 0, f"cannot index batch dim {dim=}"
        batch_size, idx_size = idx.shape
        assert batch_size == t.size(0)
        assert batch_size == val.size(0)

        # if we can view the batch and indexed dimensions together, we could
        # do index trickery. This is, sadly, not the case for kvcache so we
        # fall back to for loop
        for i in range(batch_size):
            unbatched_dim = dim if dim < 0 else dim - 1
            t[i].index_copy_(unbatched_dim, idx[i], val[i])
        return t


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    if cos.dim() > 1:
        # batch dimensions must align
        # sin/cos are (B, T, hs) so we unsqeeze -3 for nh
        # we count from back because all of apply_rope does
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class KVCache(nn.Module):
    def __init__(
        self,
        k_shape: Tuple[int, int, int, int],
        v_shape: Tuple[int, int, int, int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "k",
            torch.zeros(k_shape, device=device, dtype=dtype),
            persistent=False,
        )
        self.register_buffer(
            "v",
            torch.zeros(v_shape, device=device, dtype=dtype),
            persistent=False,
        )

    def forward(
        self, input_pos: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # move the buffer to the activation dtype for when AMP is used
        self.k = self.k.to(k.dtype)
        self.v = self.v.to(v.dtype)
        # update the cache
        B = k.size(0)
        # k = batched_index_copy_(self.k[:B, ...], -2, input_pos, k)
        # v = batched_index_copy_(self.v[:B, ...], -2, input_pos, v)
        if input_pos.size(1) > 1:
            # prefill phase
            sequence_length = k.shape[2]
            self.k[:B, :, :sequence_length, :] = k[:B, :, :sequence_length, :]
            self.v[:B, :, :sequence_length, :] = v[:B, :, :sequence_length, :]
        else:
            batched_index_copy_(self.k[:B, ...], -2, input_pos, k)
            batched_index_copy_(self.v[:B, ...], -2, input_pos, v)
        return self.k[:B], self.v[:B]

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.k)
        torch.nn.init.zeros_(self.v)


class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from:
        https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
    BSD 3-Clause License:
        https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE
    """

    def __init__(
        self,
        size: int,
        dim: int = -1,
        eps: float = 1e-6,
        add_unit_offset: bool = False,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim
        self.add_unit_offset = add_unit_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        weight = (1 + self.weight) if self.add_unit_offset else self.weight
        return (x_normed * weight.float()).to(dtype=dtype)

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
