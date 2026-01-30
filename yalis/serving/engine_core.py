"""
EngineCore: sync loop running on a dedicated OS thread.

Orchestrates:
- Inbox draining (input transport -> scheduler)
- Scheduler (admit, get_batch, finalize)
- Executor (prefill/decode steps)
- Output (completed requests -> output transport)
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

from yalis import ModelConfig, InferenceConfig, LLMEngine
from yalis.attention.kv_cache.kv_slots_manager import KVSlotsManager

from .scheduler import Scheduler
from .executor import Executor
from .schemas import InternalRequest
from .transport import InProcTransport
from .logger import get_logger


@dataclass
class SubmitItem:
    req: InternalRequest


@dataclass
class OutputItem:
    request_id: str
    req: InternalRequest


class EngineCore:
    """
    Sync engine loop running on a dedicated thread.

    Owns: LLMEngine, KVSlotsManager, Scheduler, Executor
    """

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config: InferenceConfig,
        input_transport: InProcTransport,
        output_transport: InProcTransport,
    ) -> None:
        self.model_config = model_config
        self.inference_config = inference_config
        self.input_transport = input_transport
        self.output_transport = output_transport

        self.stop_event = threading.Event()
        self.ready_event = threading.Event()  # Set when engine is ready
        self.startup_error: Optional[Exception] = None
        self.thread: Optional[threading.Thread] = None
        self.logger = get_logger("engine_core")

        # Built on start()
        self.engine: Optional[LLMEngine] = None
        self.kv_slots: Optional[KVSlotsManager] = None
        self.scheduler: Optional[Scheduler] = None
        self.executor: Optional[Executor] = None

    def start(self, timeout: float = 300.0) -> None:
        """Start the engine core thread and wait for it to be ready."""
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.ready_event.clear()
        self.startup_error = None
        self.thread = threading.Thread(target=self.run_forever, name="yalis-engine-core", daemon=True)
        self.thread.start()

        # Wait for engine to be ready
        # TODO(Prajwal): The engine can be built in init itself
        if not self.ready_event.wait(timeout=timeout):
            raise RuntimeError(f"EngineCore did not become ready within {timeout}s")
        if self.startup_error is not None:
            raise self.startup_error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=10.0)
        self.thread = None

    def run_forever(self) -> None:
        self.logger.info("EngineCore starting, building engine...")
        try:
            self._build_components()
        except Exception as e:
            self.logger.exception("EngineCore failed to build components")
            self.startup_error = e
            self.ready_event.set()
            return

        self.logger.info("EngineCore ready, entering main loop")
        self.ready_event.set()

        while not self.stop_event.is_set():
            self._step()

        self.logger.info("EngineCore stopped")

    def _build_components(self) -> None:
        """Build engine, KV manager, scheduler, executor on this thread."""
        # Build engine (on this thread for CUDA TLS consistency)
        self.engine = LLMEngine(
            model_config=self.model_config,
            inference_config=self.inference_config,
        )

        self.kv_slots = self.engine.kv_slots_manager

        # Build scheduler and executor
        self.scheduler = Scheduler(max_concurrency=self.inference_config.max_batch_size)
        self.scheduler.kv_slots = self.kv_slots

        self.executor = Executor(engine=self.engine, kv_slots=self.kv_slots)

    def _step(self) -> None:
        """One iteration of the main loop."""
        # 1. Drain inbox
        self._drain_inbox()

        # 2. Get the next scheduled set of requests
        phase, batch = self.scheduler.schedule()

        if not batch:
            # No requests to schedule, sleep for a bit
            time.sleep(0.001)
            return

        # 4. Execute step
        t0 = time.time()
        try:
            next_tokens = self.executor.execute_step(phase, batch)
        except Exception:
            self.logger.exception(f"execute_step failed phase={phase}")
            self._handle_batch_error(batch)
            return

        # 5. Post-process
        finished = self.scheduler.process_outputs(batch, next_tokens)

        self._post_process(finished)

        # 6. Record step timing
        self.scheduler.metrics.step_ms_sum += (time.time() - t0) * 1000.0
        self.scheduler.metrics.step_count += 1

    def _drain_inbox(self) -> None:
        """Pull all pending submits from input transport."""
        while True:
            try:
                item: SubmitItem = self.input_transport.get_sync_nowait()
                self.scheduler.add_request(item.req)
                self.logger.info(f"received req_id={item.req.request_id}")
            except Exception:
                break

    def _post_process(self, phase: str, batch: List[InternalRequest], next_tokens) -> None:
        """Update requests with generated tokens and check for completion."""
        tokenizer = self.engine.tokenizer
        eos_id = tokenizer.eos_token_id
        now = time.time()

        next_list = next_tokens.detach().cpu().tolist()

        for req, tok_list in zip(batch, next_list):
            tok = tok_list[0] if isinstance(tok_list, list) else tok_list
            req.last_token_id = int(tok)
            req.output_token_ids.append(req.last_token_id)
            req.generated_count += 1

            # Decode text
            text = tokenizer.decode(req.output_token_ids, skip_special_tokens=True)

            # Check stop conditions
            stop_hit = False
            reason = None

            if req.last_token_id == eos_id:
                stop_hit = True
                reason = "stop"

            if not stop_hit and req.stop:
                for s in req.stop:
                    if s and text.endswith(s):
                        text = text[:-len(s)]
                        stop_hit = True
                        reason = "stop"
                        break

            req.output_text = text

            if phase == "PREFILL":
                req.has_first_token = True
                if req.started_ts:
                    self.scheduler.metrics.ttft_ms_sum += (now - req.started_ts) * 1000.0
                    self.scheduler.metrics.ttft_count += 1

            # Check if done
            if stop_hit or req.generated_count >= req.sampling.max_tokens:
                self._finish_request(req, reason, now)
            else:
                req.phase = "DECODE"

    def _post_process(self, finished: List[InternalRequest]) -> None:
        """Post-process the finished requests."""
        # Send to output transport
        for req in finished:
            out = OutputItem(request_id=req.request_id, req=req)
            self.output_transport.put_sync(out)

    def _handle_batch_error(self, batch: List[InternalRequest]) -> None:
        """Mark all requests in batch as done with error."""
        now = time.time()
        for req in batch:
            req.status = "DONE"
            req.finish_reason = "error"
            req.finished_ts = now
            out = OutputItem(request_id=req.request_id, req=req)
            self.output_transport.put_sync(out)
            self.scheduler.finalize(req.request_id)
