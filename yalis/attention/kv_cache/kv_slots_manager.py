from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from yalis.attention.kv_cache.kv_cache_policy import (
    ContiguousKVCachePolicy,
    PagedKVCachePolicy,
)
from yalis.attention.kv_cache.slot_allocator import SlotAllocator


class KVSlotsManager:
    """
    Unified row/slot + KV-cache manager.
    - Owns row_id assignment (via SlotAllocator)
    - Hides paged vs contiguous cache differences
    - Minimal Engine-facing API:
      * allocate(req_id, prompt_lengths) -> rows, block_table_rows|None
      * update(delta_tokens, rows) -> block_table_rows|None
      * free(req_id) -> freed_rows
      * view(rows) -> block_table_rows|None
      * reset()
    """

    def __init__(
        self,
        capacity: int,
        paged: bool,
        max_num_blocks_per_seq: Optional[int] = None,
        num_blocks: Optional[int] = None,
        page_block_size: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        self.slot_allocator = SlotAllocator(capacity)
        if paged:
            assert (
                max_num_blocks_per_seq is not None
                and num_blocks is not None
                and page_block_size is not None
            ), "paged requires max_num_blocks_per_seq, num_blocks, page_block_size"  # noqa: E501
            self.cache_policy = PagedKVCachePolicy(
                batch_size=capacity,
                max_num_blocks_per_seq=max_num_blocks_per_seq,
                num_blocks=num_blocks,
                page_block_size=page_block_size,
                verbose=verbose,
            )
            self.paged = True
        else:
            self.cache_policy = ContiguousKVCachePolicy(capacity=capacity)
            self.paged = False

    def allocate(
        self,
        req_ids: List[str],
        prompt_lengths: torch.Tensor,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Allocate one row per sequence in prompt_lengths for request req_ids.
        If there are no free slots, return an empty list.
        Returns list of slot_ids for the allocated rows.
        """
        if prompt_lengths.dim() != 1:
            raise ValueError("prompt_lengths must be 1D [B]")
        B = int(prompt_lengths.size(0))
        assert (
            len(req_ids) == B
        ), "req_ids and prompt_lengths must have the same length"
        n = min(B, self.slot_allocator.free_count())
        rows: List[int] = []
        for i in range(n):
            slot_id = self.slot_allocator.allocate(req_ids[i])
            rows.append(slot_id)
        if len(rows) > 0:
            self.cache_policy.allocate(rows, prompt_lengths[: len(rows)])
        return rows

    def update(
        self,
        rows: Union[List[int], torch.Tensor],  # [B] slot_ids
        delta_tokens: Union[int, torch.Tensor],
    ) -> None:
        """
        Update KV-cache page allocations for the provided rows by delta_tokens.
        Returns block_table slice for rows when paged; None for contiguous.
        """
        if isinstance(rows, torch.Tensor):
            rows = rows.tolist()
        if isinstance(delta_tokens, int):
            self.cache_policy.update(rows, delta_tokens)
        elif isinstance(delta_tokens, torch.Tensor):
            self.cache_policy.update(rows, delta_tokens.tolist())
        else:
            raise TypeError("delta_tokens must be int or 1D torch.Tensor")

    def allocate_for_rows(
        self,
        rows: Union[List[int], torch.Tensor],
        prompt_lengths: Union[List[int], torch.Tensor],
    ) -> None:
        """
        Allocate KV-cache pages for already-assigned rows with given prompt
        lengths. Does not change row ownership; only manages page assignment.
        Used for testing.
        """
        if isinstance(rows, torch.Tensor):
            slot_ids = rows.tolist()
        else:
            slot_ids = rows  # type: ignore[assignment]
        if isinstance(prompt_lengths, list):
            lengths_t = torch.tensor(prompt_lengths, dtype=torch.long)
        elif isinstance(prompt_lengths, torch.Tensor):
            lengths_t = prompt_lengths
        else:
            raise TypeError("prompt_lengths must be List[int] or torch.Tensor")
        self.cache_policy.allocate(slot_ids, lengths_t)

    def free(self, req_id: str) -> int | None:
        """
        Free all rows owned by req_id and release paged allocations if any.
        Returns list of freed row ids.
        """
        slot_id = self.slot_allocator.free(req_id)
        if slot_id is not None:
            self.cache_policy.release([slot_id])
        return slot_id

    def view(
        self, ids: List[int] | List[str], is_request_ids: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Return block_table slice for rows when paged; None for contiguous.
        """
        if is_request_ids:
            slot_ids = [
                self.slot_allocator.get_slot_id(req_id) for req_id in ids
            ]
        else:
            slot_ids = ids
        return self.cache_policy.view(slot_ids)

    def reset(self) -> None:
        self.slot_allocator.reset()
        self.cache_policy.reset()

    def lengths(self, rows: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Return current per-row sequence lengths.
        Delegates to policy.
        """
        if isinstance(rows, list):
            rows_t = torch.tensor(rows, dtype=torch.int64, device="cuda")
        else:
            rows_t = rows.to(torch.int64, device="cuda")
        return self.cache_policy.lengths(rows_t.tolist())
