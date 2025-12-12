from __future__ import annotations

from typing import Optional

from .core_client import CoreClient
from .engine_core import EngineCore
from .schemas import InternalRequest


class AsyncYalis:
    """
    Lifecycle owner for serving.

    - Owns the engine-side Core (sync, dedicated thread) + the async CoreClient facade.
    - The FastAPI app should depend on this object, not directly on scheduler/executor/etc.
    - Later we can swap the transport (in-proc queues -> ZMQ) without changing this API.
    """

    def __init__(self) -> None:
        self._core = EngineCore()
        self._client = CoreClient(self._core)
        self._started = False

    @property
    def client(self) -> CoreClient:
        return self._client

    async def start(self) -> None:
        if self._started:
            return
        await self._client.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self._client.stop()
        self._started = False

    async def add_request(self, req: InternalRequest) -> str:
        """
        API-agnostic submission: only accepts InternalRequest.
        """
        if not self._started:
            await self.start()
        return await self._client.submit(req)

    async def wait_done(self, request_id: str, *, timeout_s: Optional[float] = None) -> InternalRequest:
        if not self._started:
            await self.start()
        return await self._client.wait_done(request_id, timeout_s=timeout_s)


