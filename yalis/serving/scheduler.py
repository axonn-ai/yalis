"""
Sync scheduler for EngineCore.

Manages:
- Request admission (WAITING -> RUNNING)
- Slot allocation via KVSlotsManager
- Batch collection (PREFILL vs DECODE)
- Request finalization
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .schemas import InternalRequest
from .logger import get_logger
import torch

logger = get_logger("scheduler")


@dataclass
class Metrics:
    submitted: int = 0
    admitted: int = 0
    completed: int = 0
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
            "avg_ttft_ms": avg_ttft,
            "avg_step_ms": avg_step,
        }


class Scheduler:
    """
    Sync scheduler that manages request lifecycle and batching.

    Uses:
    - waiting: deque for FCFS ordering of waiting requests
    - running: set of currently running request IDs
    - requests: dict for O(1) lookup by request_id
    """

    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max_concurrency
        self.metrics = Metrics()

        # Request store (for O(1) lookup)
        self.requests: Dict[str, InternalRequest] = {}

        # FCFS queue for waiting requests
        self.waiting: deque[str] = deque()

        # Currently running request IDs
        self.running: Set[str] = set()

        # KV slots manager (injected by EngineCore)
        self.kv_slots = None

    def add_request(self, req: InternalRequest) -> None:
        """Add a new request to the scheduler."""
        self.requests[req.request_id] = req
        self.waiting.append(req.request_id)
        self.metrics.submitted += 1

    def admit(self) -> List[str]:
        """Admit waiting requests (FCFS) up to max_concurrency. Returns list of admitted request_ids."""
        admitted = []

        while self.waiting and len(self.running) < self.max_concurrency:
            if self.kv_slots is None:
                raise RuntimeError("kv_slots not set")
            if self.kv_slots.slot_allocator.free_count() == 0:
                break

            req_id = self.waiting.popleft()
            req = self.requests.get(req_id)

            # TODO: This should never happen right now
            if req is None or req.status != "WAITING":
                raise RuntimeError(f"request {req_id} is not WAITING. This should never happen.")
                continue

            slot_id = self.kv_slots.allocate([req_id], torch.tensor([len(req.prompt_token_ids)], dtype=torch.long))[0]
            req.slot_id = slot_id
            req.status = "RUNNING"
            req.started_ts = time.time()
            req.phase = "PREFILL"
            self.running.add(req_id)
            self.metrics.admitted += 1
            admitted.append(req_id)
            logger.info(f"admit req_id={req_id} slot={slot_id}")

        return admitted

    def schedule(self) -> tuple[str, List[InternalRequest]]:
        """Get next batch of requests to process (PREFILL or DECODE)."""

        # Admit new requests to the running set
        self.admit()

        prefill_reqs = []
        decode_reqs = []

        for req_id in self.running:
            req = self.requests.get(req_id)
            if req is None or req.status != "RUNNING":
                continue
            if req.phase == "PREFILL":
                prefill_reqs.append(req)
            elif req.phase == "DECODE":
                decode_reqs.append(req)

        # Sort by slot_id for consistent ordering
        if prefill_reqs:
            prefill_reqs.sort(key=lambda r: r.slot_id or 0)
            return "PREFILL", prefill_reqs
        if decode_reqs:
            decode_reqs.sort(key=lambda r: r.slot_id or 0)
            return "DECODE", decode_reqs

        return "IDLE", []
    
    def process_outputs(self, batch: List[InternalRequest], outputs: List[OutputItem]) -> None:
        """Process outputs for a batch of requests."""
        finished = []
        now = time.time()

        for req, tok_list in zip(batch, outputs):
            # TODO(Prajwal): This can be multiple tokens in case of 
            # scenarios like speculative decoding
            if not isinstance(tok_list, list):
                tok_list = [tok_list]
            #req.output_token_ids.extend(tok_list)
            #req.generated_count += len(tok_list)

            for tok in tok_list:
                req.output_token_ids.append(int(tok))
                req.last_token_id = int(tok)
                req.generated_count += 1

                stopped = self.check_stop(req, int(tok))

                if stopped:
                    req.status = "DONE"
                    req.finished_ts = now
                    logger.info(f"completed req_id={req.request_id} tokens={req.generated_count}")
                    finished.append(req)
                    break
                
            
            # Update the KV slots
            logger.debug(f"update req_id={req.request_id} slot={req.slot_id} len(tok_list)={len(tok_list)}")
            self.kv_slots.update([req.slot_id], len(tok_list))

            # Transition to DECODE phase
            if req.phase == "PREFILL":
                if req.started_ts is not None:
                    self.metrics.ttft_ms_sum += (time.time() - req.started_ts) * 1000.0
                    self.metrics.ttft_count += 1
                req.phase = "DECODE"
            

        # Finalize the requests and clean up the slots
        for req in finished:
            self.finalize(req.request_id)
        return finished
  

    def check_stop(self, req: InternalRequest, tok: int) -> bool:
        if req.generated_count >= req.sampling.max_tokens:
            req.finish_reason = "length"
            return True

        if tok in req.sampling.stop_token_ids:
            req.finish_reason = "stop"
            return True

        return False

    def finalize(self, request_id: str) -> Optional[int]:
        """Remove request from running set and free slot. Returns freed slot_id."""
        self.running.discard(request_id)
        self.metrics.completed += 1
        if self.kv_slots is not None:
            return self.kv_slots.free(request_id)
        return None
