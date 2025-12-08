from __future__ import annotations

from typing import Dict, Set

from yalis.serving.logger import get_logger

logger = get_logger("slot_allocator")

class SlotAllocator:
  """
  Sequence row allocator for online serving.
  - Hands out stable row ids in [0..capacity-1]
  - Smallest-available policy keeps active rows dense
  """

  def __init__(self, capacity: int) -> None:
    if capacity <= 0:
      raise ValueError("capacity must be > 0")
    
    logger.info(f"SlotAllocator capacity: {capacity}")
    self._capacity: int = capacity
    self._free: Set[int] = set(range(capacity))
    self._req_to_slot_id: Dict[str, int] = {}

  @property
  def capacity(self) -> int:
    return self._capacity

  def free_count(self) -> int:
    return len(self._free)

  def allocate(self, req_id: str) -> int:
    if self.free_count() == 0:
      raise RuntimeError(
        "insufficient free slots. Call free_count() first to check if there are any free slots."
      )
    logger.info(f"SlotAllocator allocate req_id: {req_id} free_count: {self.free_count()}, _free: {self._free}")
    slot_id = min(self._free)
    self._free.remove(slot_id)
    self._req_to_slot_id[req_id] = slot_id
    return slot_id

  def free(self, req_id: str) -> int | None:
    slot_id = self._req_to_slot_id.pop(req_id, None)
    if slot_id is not None:
      self._free.add(slot_id)
    return slot_id

  def get_slot_id(self, req_id: str) -> int:
    return self._req_to_slot_id[req_id]

  def reset(self) -> None:
    self._free = set(range(self._capacity))
    self._req_to_slot_id.clear()