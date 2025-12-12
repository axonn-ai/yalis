import asyncio
import time
from typing import Optional, Set
import torch

from .scheduler import AsyncScheduler
from .executor import Executor
from .logger import get_logger


class EngineLoop:
  def __init__(self, scheduler: AsyncScheduler, executor: Executor, tick_ms: int = 0) -> None:
    self.scheduler = scheduler
    self.executor = executor
    # Remove artificial tick pacing; engine step cadence drives the loop
    # TODO: Remove tick_s altogether
    self.tick_s = 0.0
    self._task: Optional[asyncio.Task] = None
    self._stop = asyncio.Event()
    self.logger = get_logger("engine_loop")

  async def start(self) -> None:
    if self._task is None or self._task.done():
      self._stop.clear()
      self._task = asyncio.create_task(self._loop())

  async def stop(self) -> None:
    if self._task is not None:
      self._stop.set()
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      self._task = None

  async def _loop(self) -> None:
    self.logger.info("EngineLoop loop started")
    while not self._stop.is_set():
      t0 = time.time()
      phase, reqs = await self.scheduler.get_next_batch()
      if phase == "IDLE" or not reqs:
        # Cooperative yield when there's no work; prevents busy-loop starvation
        await asyncio.sleep(0.001)
        continue

      # Ensure KV rows are prepared before the engine step
      if phase == "PREFILL":
        # Allocate pages for each row based on precomputed prompt lengths
        rows = [int(req.slot_id) for req in reqs if req.slot_id is not None]
        self.logger.info(f"EngineLoop loop PREFILL rows: {rows}")
        if len(rows) != len(reqs):
          self.logger.error("Missing slot_id on one or more requests during PREFILL")
          await asyncio.sleep(0)
          continue
        lengths_list = [len(req.prompt_token_ids) for req in reqs]  # type: ignore[arg-type]
        lengths_t = torch.tensor(lengths_list, dtype=torch.long)
        try:
          token_counter = torch.zeros(len(rows), dtype=torch.int32, device=self.executor.adapter.engine.device)
          self.scheduler.kv_slots.allocate_for_rows(rows, lengths_t)  # type: ignore[union-attr]
        except Exception:
          self.logger.exception("kv_slots.allocate_for_rows failed")
          await asyncio.sleep(0)
          continue
      elif phase == "DECODE":
        # Extend rows by one token per request
        self.logger.info("EngineLoop loop DECODE")
        rows = [int(req.slot_id) for req in reqs if req.slot_id is not None]
        if len(rows) != len(reqs):
          self.logger.error("Missing slot_id on one or more requests during DECODE")
          await asyncio.sleep(0)
          continue
        try:
          token_counter = self.scheduler.kv_slots.lengths(rows)
          self.scheduler.kv_slots.update(rows, 1)  # type: ignore[union-attr]
        except Exception:
          self.logger.exception("kv_slots.update failed")
          await asyncio.sleep(0)
          continue

      # Execute one step after KV is prepared
      t_step0 = time.time()
      try:
        block_table = self.scheduler.kv_slots.view(rows)  # type: ignore[union-attr]
        next_tok, num_new_tokens = await self.executor.execute_step(phase, reqs, block_table, token_counter)
      except Exception as e:
        self.logger.exception(f"execute_step failed phase={phase} batch_size={len(reqs)}")
        now = time.time()
        finished = set()
        for req in reqs:
          req.status = "DONE"
          req.phase = "DONE"
          req.finished_ts = now
          finished.add(req.request_id)
        self.scheduler.finalize_batch(finished)
        continue
      now = time.time()
      finished: Set[str] = set()
      # Post-process
      tokenizer = self.scheduler.adapter.tokenizer  # type: ignore[union-attr]
      eos_id = tokenizer.eos_token_id
      next_list = next_tok.detach().cpu().tolist()
      for req, tok_list in zip(reqs, next_list):
        tok = tok_list[0] if isinstance(tok_list, list) else tok_list
        req.last_token_id = int(tok)
        req.output_token_ids.append(req.last_token_id)
        # Inline decode and stop condition compute
        stop_hit = False
        eos_hit = (req.last_token_id == eos_id)
        reason: Optional[str] = None
        # Decode full text so far
        text = tokenizer.decode(req.output_token_ids, skip_special_tokens=True)
        # Stop by EOS
        if eos_hit:
          stop_hit = True
          reason = "stop"
        # Stop by stop sequences (suffix trim)
        if not stop_hit:
          stop_list = getattr(req, "stop", None)
          if stop_list:
            for s in stop_list:
              if s and text.endswith(s):
                text = text[: -len(s)]
                stop_hit = True
                reason = "stop"
                break
        req.output_text = text

        if phase == "PREFILL":
          req.has_first_token = True
          # TTFT once on first prefill
          if req.started_ts is not None:
            self.scheduler.metrics.ttft_ms_sum += (now - req.started_ts) * 1000.0
            self.scheduler.metrics.ttft_count += 1
        req.generated_count += 1
        # Finish or phase transition
        if stop_hit or req.generated_count >= req.sampling.max_tokens:
          # Finalize finish reason
          if reason:
            req.finish_reason = reason
          elif req.generated_count >= req.sampling.max_tokens:
            req.finish_reason = "length"
          else:
            req.finish_reason = "stop"
          self.logger.info(f"phase transition req_id={req.request_id} phase=DONE tokens={req.generated_count}, output_token_ids={req.output_token_ids}")
          req.status = "DONE"
          req.phase = "DONE"
          req.finished_ts = now
          finished.add(req.request_id)
        else:
          self.logger.info(f"phase transition req_id={req.request_id} phase=DECODE, last_token_id={req.last_token_id}")
          req.phase = "DECODE"

      for rid in list(finished):
        self.scheduler.metrics.completed += 1
      slot_ids = self.scheduler.finalize_batch(finished)
      # No per-request processor state to cleanup (stateless)


      self.scheduler.metrics.step_ms_sum += (time.time() - t_step0) * 1000.0
      self.scheduler.metrics.step_count += 1

      # Tick pacing
      # Always yield once per loop to keep event loop responsive
      await asyncio.sleep(0)


