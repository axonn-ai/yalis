from __future__ import annotations

import asyncio
from typing import Optional

from .engine_core import EngineCore, SubmitItem
from .schemas import InternalRequest


class CoreClient:
    """
    vLLM-style "frontend client" used by the FastAPI layer.

    - Async API (because FastAPI/ASGI is async).
    - Talks to a sync EngineWorker running on a dedicated OS thread.
    - Only deals in InternalRequest objects (API-agnostic).

    Step 1: skeleton only (not wired into server yet).
    """

    def __init__(self, core: EngineCore) -> None:
        self._core = core

    async def start(self) -> None:
        # Starting a thread is cheap; do it synchronously.
        self._core.start()

    async def stop(self) -> None:
        self._core.stop()

    async def submit(self, req: InternalRequest) -> str:
        """
        Submit an InternalRequest to the engine.

        Returns request_id once accepted by the frontend.

        In later steps, this will also return an awaitable for completion/streaming,
        similar to vLLM's add_request + output subscription.
        """
        loop = asyncio.get_running_loop()
        accepted = loop.create_future()
        item = SubmitItem(req=req, result_handle=accepted)
        self._core.submit(item)
        # Step 1: we don't have engine-side handling yet; just acknowledge enqueue.
        accepted.set_result(True)
        return req.request_id

    async def wait_done(self, request_id: str, *, timeout_s: Optional[float] = None) -> InternalRequest:
        """
        Placeholder: in later steps this will await completion from the engine.
        """
        raise NotImplementedError("CoreClient.wait_done will be implemented after EngineWorker can return results.")


