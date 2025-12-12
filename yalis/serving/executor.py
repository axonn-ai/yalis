import asyncio
from typing import List, Literal, Tuple
import torch

from .logger import get_logger
from .schemas import InternalRequest

Phase = Literal["PREFILL", "DECODE"]


class Executor:
  """
  Batched executor: builds tensors from provided InternalRequests and calls
  the async engine adapter for one phase per tick. Stateless; no access to
  the queue or scheduler state.
  """

  def __init__(self, adapter) -> None:
    self.adapter = adapter
    self.logger = get_logger("executor")

  async def execute_step(self, phase: Phase, requests: List[InternalRequest], block_table: torch.Tensor, token_counter: torch.Tensor) -> Tuple[torch.Tensor, list[int]]:
    """
    Execute a single engine step for the given requests.
    Returns (next_tokens, num_new_tokens) where next_tokens is (B, 1) and
    num_new_tokens is a per-request list of number of new tokens consumed this step.
    """
    if not requests:
      return torch.empty((0, 1), dtype=torch.long), []
    device = self.adapter.engine.device
    tokenizer = self.adapter.tokenizer
    self.logger.info(f"Executor block_table: {block_table}")
    if phase == "PREFILL":
      # Build from pre-tokenized ids (assumed present)
      id_lists = [req.prompt_token_ids for req in requests]  # type: ignore[list-item]
      lengths_list = [len(ids) for ids in id_lists]
      max_len = max(lengths_list) if lengths_list else 0
      if max_len == 0:
        return torch.empty((len(requests), 1), dtype=torch.long, device=device), [0] * len(requests)
      padded = []
      pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
      for ids in id_lists:
        row = ids + [pad_id] * (max_len - len(ids))  # type: ignore[arg-type]
        padded.append(row)
      tokens = torch.tensor(padded, dtype=torch.long, device=device)
      lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)
      self.logger.info(f"Executor prefill tokens: {tokens}, lengths: {lengths}")
      next_tok, _ = await self.adapter.prefill(tokens, lengths, block_table, token_counter)
      return next_tok, lengths.detach().cpu().tolist()

    elif phase == "DECODE":
      last_tokens = torch.tensor(
        [[req.last_token_id] for req in requests],
        dtype=torch.long,
        device=device,
      )
      next_tok, _ = await self.adapter.decode(last_tokens, block_table, token_counter)
      return next_tok, [1 for _ in requests]
    else:
      raise ValueError(f"Unknown phase: {phase}")


