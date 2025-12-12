from __future__ import annotations

import threading
import queue
from dataclasses import dataclass
from typing import Optional

from .schemas import InternalRequest


@dataclass(frozen=True)
class SubmitItem:
    """
    A single submission into the engine core.

    The engine core is deliberately API-agnostic: it only ever accepts InternalRequest.
    """

    req: InternalRequest
    # The frontend event loop will provide a handle so the core can complete it
    # via loop.call_soon_threadsafe(...). We keep this typed as `object` here to
    # avoid importing asyncio types into the sync engine core.
    result_handle: object


class EngineCore:
    """
    vLLM-style "engine side" owner.

    - Runs on a dedicated OS thread.
    - Core implementation is synchronous (no async defs).
    - Communicates with the async frontend via thread-safe queues + callbacks.

    Step 1: skeleton only (no scheduling/execution yet).
    """

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._in_q: "queue.Queue[SubmitItem]" = queue.Queue()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_forever, name="yalis-engine-core", daemon=True)
        self._thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join_timeout_s)

    def submit(self, item: SubmitItem) -> None:
        """
        Thread-safe: called by the async frontend to submit an InternalRequest.
        """
        self._in_q.put(item)

    def _run_forever(self) -> None:
        """
        Dedicated-thread entrypoint.

        In later steps this will:
        - drain admissions,
        - run scheduler,
        - execute prefill/decode,
        - publish status/results back to the frontend.
        """
        while not self._stop.is_set():
            try:
                _item = self._in_q.get(timeout=0.05)
            except queue.Empty:
                continue
            # Step 1: do nothing yet (wiring comes later).
            # Intentionally drop on the floor for now.
            # The CoreClient will not be wired until the core can return results.
            continue


