from typing import Optional, Protocol, Union, List

import torch


class KVCachePolicy(Protocol):
    def allocate(
        self, slot_ids: List[int], prompt_lengths: torch.Tensor
    ) -> None: ...

    def update(
        self, slot_ids: List[int], n_new_tokens: Union[int, List[int]]
    ) -> None: ...

    def release(self, slot_ids: List[int]) -> None: ...

    def view(self, slot_ids: List[int]) -> Optional[torch.Tensor]: ...

    def reset(self) -> None: ...

    def lengths(self, slot_ids: List[int]) -> Optional[torch.Tensor]:
        """
        Return per-row sequence lengths for provided slot ids.
        Returns None if the policy does not track lengths internally.
        """
        ...


class ContiguousKVCachePolicy:
    """
    No-op manager for contiguous KV-cache layout.
    Engine can call allocate/update/view to keep a uniform API.
    Tracks per-slot sequence lengths locally.
    """

    def __init__(
        self, capacity: int, device: Optional[torch.device] = None
    ) -> None:
        self._seq_lens = torch.zeros(
            capacity, dtype=torch.int32, device="cuda"
        )

    def allocate(
        self, slot_ids: List[int], prompt_lengths: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if prompt_lengths.numel() == 0 or len(slot_ids) == 0:
            return None
        index = torch.tensor(
            slot_ids, dtype=torch.int64, device=self._seq_lens.device
        )
        self._seq_lens.index_copy_(
            0,
            index,
            prompt_lengths.to(dtype=torch.int32, device=self._seq_lens.device),
        )
        return None

    def update(
        self, slot_ids: List[int], n_new_tokens: Union[int, List[int]]
    ) -> Optional[torch.Tensor]:
        if len(slot_ids) == 0:
            return None
        index = torch.tensor(
            slot_ids, dtype=torch.int64, device=self._seq_lens.device
        )
        if isinstance(n_new_tokens, int):
            self._seq_lens.index_add_(
                0,
                index,
                torch.full(
                    (len(slot_ids),),
                    int(n_new_tokens),
                    dtype=torch.int32,
                    device=self._seq_lens.device,
                ),
            )
        else:
            dt = torch.tensor(
                n_new_tokens, dtype=torch.int32, device=self._seq_lens.device
            )
            self._seq_lens.index_add_(0, index, dt)
        return None

    def release(self, slot_ids: List[int]) -> None:
        if len(slot_ids) == 0:
            return None
        index = torch.tensor(
            slot_ids, dtype=torch.int64, device=self._seq_lens.device
        )
        self._seq_lens.index_fill_(0, index, 0)
        return None

    def view(self, slot_ids: List[int]) -> Optional[torch.Tensor]:
        return None

    def reset(self) -> None:
        self._seq_lens.zero_()
        return None

    def lengths(self, slot_ids: List[int]) -> Optional[torch.Tensor]:
        index = torch.tensor(
            slot_ids, dtype=torch.int64, device=self._seq_lens.device
        )
        return self._seq_lens.index_select(0, index)


class PagedKVCachePolicy:
    """
    Thin Python wrapper over the C++ paged KV cache allocator (kvcache_manager)
    Owned by the Engine. Provides a stable API for page/block table management
    """

    def __init__(
        self,
        batch_size: int,
        max_num_blocks_per_seq: int,
        num_blocks: int,
        page_block_size: int,
        verbose: bool = False,
    ) -> None:
        # Lazy import
        from kvcache_manager import KVCacheManager as _CppKVCacheManager

        self._impl = _CppKVCacheManager(
            batch_size,
            max_num_blocks_per_seq,
            num_blocks,
            page_block_size,
        )
        self._verbose = verbose

    def block_table(self) -> torch.Tensor:
        return self._impl.block_table()

    def allocate(
        self, slot_ids: List[int], prompt_lengths: torch.Tensor
    ) -> None:
        for i, slot in enumerate(slot_ids):
            self._impl.allocate_sequence(
                int(slot), int(prompt_lengths[i].item())
            )

    def update(
        self, slot_ids: List[int], n_new_tokens: Union[int, List[int]]
    ) -> None:
        if isinstance(n_new_tokens, int):
            for slot in slot_ids:
                self._impl.extend_sequence(int(slot), int(n_new_tokens))
        elif isinstance(n_new_tokens, list):
            if len(n_new_tokens) != len(slot_ids):
                raise ValueError(
                    "n_new_tokens list must match slot_ids length"
                )
            for slot, dt in zip(slot_ids, n_new_tokens):
                self._impl.extend_sequence(int(slot), int(dt))
        else:
            raise TypeError("n_new_tokens must be int or list[int]")

    def release(self, slot_ids: List[int]) -> None:
        for slot in slot_ids:
            self._impl.free_sequence(int(slot))

    def view(self, slot_ids: List[int]) -> torch.Tensor:
        bt = self._impl.block_table()
        index = torch.tensor(slot_ids, dtype=torch.int64, device=bt.device)
        return bt.index_select(0, index)

    def reset(self) -> None:
        self._impl.reset()

    def lengths(self, slot_ids: List[int]) -> torch.Tensor:
        """
        Return current per-row token counts from the C++ manager (CPU int64),
        indexed by the provided slot ids.
        """
        all_counts = self._impl.tokens_assigned()
        index = torch.tensor(
            slot_ids, dtype=torch.int32, device=all_counts.device
        )
        return all_counts.index_select(0, index)
