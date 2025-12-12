import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set
from typing import List

from .queue import WaitQueue
from .schemas import InternalRequest
from .prompt import build_prompt
from .logger import get_logger
logger = get_logger("scheduler")


@dataclass
class Metrics:
    submitted: int = 0
    admitted: int = 0
    completed: int = 0
    cancelled: int = 0
    ttft_ms_sum: float = 0.0
    ttft_count: int = 0
    step_ms_sum: float = 0.0
    step_count: int = 0

    def to_dict(self):
        avg_ttft = (self.ttft_ms_sum / self.ttft_count) if self.ttft_count else None
        avg_step = (self.step_ms_sum / self.step_count) if self.step_count else None
        return {
            "submitted": self.submitted,
            "admitted": self.admitted,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "avg_ttft_ms": avg_ttft,
            "avg_step_ms": avg_step,
        }


class AsyncScheduler:
    def __init__(self, wait_queue: WaitQueue) -> None:
        self.wait_queue = wait_queue
        self.logger = logger
        # NOTE: Temporary knob for M3 simulation only; delete after real engine lands.
        # YALIS_MAX_CONCURRENCY caps concurrent RUNNING requests (later derived from engine).
        self.max_concurrency = int(os.getenv("YALIS_MAX_CONCURRENCY", "4"))
        # NOTE: Temporary knob for M3 simulation only; delete after real engine lands.
        # YALIS_SCHED_TICK_MS approximates one decode step latency in simulation.
        self.tick_ms = int(os.getenv("YALIS_SCHED_TICK_MS", "10"))

        self.metrics = Metrics()
        self._running: Set[str] = set()
        # Adapter is set by the server on startup (AsyncEngineAdapter)
        self.adapter = None
        # KV slots manager (injected by server on startup)
        self.kv_slots = None

    # EngineLoop will call get_next_batch() each tick to fetch work
    async def get_next_batch(self) -> tuple[str, list[InternalRequest]]:
        await self._admit()
        prefill_ids = self._collect_prefill_ids()
        if prefill_ids:
            return "PREFILL", [self.wait_queue.requests[rid] for rid in prefill_ids]
        decode_ids = self._collect_decode_ids()
        if decode_ids:
            return "DECODE", [self.wait_queue.requests[rid] for rid in decode_ids]
        return "IDLE", []

    async def _admit(self) -> None:
        while len(self._running) < self.max_concurrency and not self.wait_queue.queue.empty():
            # require kv_slots to be set and have capacity for at least one admission
            if self.kv_slots is None:
                raise RuntimeError("kv_slots is not set")
            if self.kv_slots.slot_allocator.free_count() == 0:
                # No free slots to admit new requests
                break
            req_id = await self.wait_queue.queue.get()
            st = self.wait_queue.requests.get(req_id)
            if st is None or st.status != "WAITING":
                continue

            # Assign stable row/slot now to keep batch index == slot mapping
            slot_id = self.kv_slots.slot_allocator.allocate(req_id)

            st.slot_id = slot_id
            st.status = "RUNNING"
            st.started_ts = time.time()
            st.phase = "PREFILL"
            # Endpoint is responsible for setting prompt_text and prompt_token_ids



            self._running.add(req_id)
            self.metrics.admitted += 1
            # Log admission with cap
            logger.info(f"admit req_id={req_id} slot={slot_id} phase=PREFILL max_tokens={st.sampling.max_tokens}")

    def _collect_prefill_ids(self) -> List[str]:
        items: List[tuple[int, str]] = []
        for req_id in list(self._running):
            st = self.wait_queue.requests.get(req_id)
            if st is None or st.status != "RUNNING":
                continue
            if st.phase == "PREFILL":
                items.append((st.slot_id if st.slot_id is not None else 1 << 30, req_id))
        items.sort()
        return [rid for _, rid in items]

    def _collect_decode_ids(self) -> List[str]:
        items: List[tuple[int, str]] = []
        for req_id in list(self._running):
            st = self.wait_queue.requests.get(req_id)
            if st is None or st.status != "RUNNING":
                continue
            if st.phase == "DECODE" and st.generated_count < st.sampling.max_tokens and st.last_token_id is not None:
                items.append((st.slot_id if st.slot_id is not None else 1 << 30, req_id))
        items.sort()
        return [rid for _, rid in items]

    def finalize_batch(self, finished: Set[str]) -> List[int]:
        logger.info(f"Finalize batch finished: {finished}")
        slot_ids: List[int] = []
        for rid in finished:
            self._running.discard(rid)
            # free KV resources and release row
            if self.kv_slots is not None:
                freed = self.kv_slots.free(rid)
                if freed is not None:
                    slot_ids.append(freed)
        return slot_ids


