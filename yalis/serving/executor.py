"""
Sync executor for EngineCore.

Builds tensors from requests and calls prefill/generate directly.
Stateless — no access to scheduler or request store.
"""
from __future__ import annotations

from typing import List, Literal

import torch

from yalis import LLMEngine
from yalis.engine import prefill, generate
from yalis.constants import EnginePhase

from .schemas import InternalRequest
from .logger import get_logger

Phase = Literal["PREFILL", "DECODE"]
logger = get_logger("executor")


class Executor:
    """
    Sync executor that runs prefill/decode steps.
    """

    def __init__(self, engine: LLMEngine, kv_slots) -> None:
        self.engine = engine
        self.kv_slots = kv_slots
        self.tokenizer = engine.tokenizer

    def execute_step(self, phase: Phase, batch: List[InternalRequest]) -> torch.Tensor:
        """
        Execute one prefill or decode step.

        Returns next_tokens tensor (B, 1).
        """
        if not batch:
            return torch.empty((0, 1), dtype=torch.long)

        device = self.engine.device
        rows = [req.slot_id for req in batch]

        if phase == "PREFILL":
            return self._prefill(batch, rows, device)
        else:
            return self._decode(batch, rows, device)

    def _prefill(self, batch: List[InternalRequest], rows: List[int], device) -> torch.Tensor:
        lengths_list = [len(req.prompt_token_ids) for req in batch]
        token_counter = torch.zeros(len(rows), dtype=torch.int32, device=device)
        block_table = self.kv_slots.view(rows)

        # Build tokens tensor
        max_len = max(lengths_list)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded = []
        for req in batch:
            ids = list(req.prompt_token_ids)
            row = ids + [pad_id] * (max_len - len(ids))
            padded.append(row)

        tokens = torch.tensor(padded, dtype=torch.long, device=device)
        lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)

        logger.debug(f"prefill batch_size={len(batch)} max_len={max_len} rows={rows} block_table={block_table}")

        next_tok, _ = prefill(
            self.engine.model,
            tokens,
            lengths,
            self.engine.inference_config.temperature,
            self.engine.inference_config.top_k,
            self.engine.inference_config.top_p,
            False,
            EnginePhase.PREFILL,
            block_table,
            token_counter,
        )
        return next_tok

    def _decode(self, batch: List[InternalRequest], rows: List[int], device) -> torch.Tensor:
        # The scheduler updated the kv slots to account for the new tokens generated 
        # in this step. token_counter is the number of tokens until the previous step.
        # We need to subtract 1 to get the correct token counter.
        token_counter = self.kv_slots.lengths(rows) - 1 
        block_table = self.kv_slots.view(rows)

        last_tokens = torch.tensor(
            [[req.last_token_id] for req in batch],
            dtype=torch.long,
            device=device,
        )

        logger.debug(f"decode batch_size={len(batch)} rows={rows} block_table={block_table}")

        next_tok, _ = generate(
            self.engine.model,
            last_tokens,
            self.engine.inference_config.temperature,
            self.engine.inference_config.top_k,
            self.engine.inference_config.top_p,
            False,
            EnginePhase.DECODE_SINGLE,
            block_table,
            token_counter,
        )
        return next_tok
