"""
Thread-safe queue for async frontend <-> sync core communication.

Uses janus.Queue for async put/get and sync put/get across threads.
This is the seam where we can later swap to ZMQ.
"""
from __future__ import annotations

from typing import Any

import janus


class InProcTransport:
    """Thread-safe queue: async or sync on either side."""

    def __init__(self, maxsize: int = 0) -> None:
        self.q: janus.Queue[Any] = janus.Queue(maxsize=maxsize)

    # -- Async side (client / API server) --

    def full(self) -> bool:
        return self.q.async_q.full()

    async def put_async(self, item: Any) -> None:
        await self.q.async_q.put(item)

    def try_put(self, item: Any) -> bool:
        try:
            self.q.async_q.put_nowait(item)
            return True
        except janus.AsyncQueueFull:
            return False

    async def get_async(self) -> Any:
        return await self.q.async_q.get()

    # -- Sync side (engine core thread) --

    def put_sync(self, item: Any) -> None:
        self.q.sync_q.put(item)

    def get_sync(self, block: bool = True, timeout: float | None = None) -> Any:
        return self.q.sync_q.get(block=block, timeout=timeout)

    def get_sync_nowait(self) -> Any:
        return self.q.sync_q.get_nowait()

    def empty(self) -> bool:
        return self.q.sync_q.empty()

    def close(self) -> None:
        self.q.close()
