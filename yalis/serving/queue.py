import asyncio
import os
from typing import Dict, Tuple

from .schemas import InternalRequest


class WaitQueue:
    def __init__(self, max_queue_size: int | None = None) -> None:
        if max_queue_size is None:
            max_queue_size = int(os.getenv("YALIS_MAX_QUEUE_SIZE", "1024"))
        self.queue: "asyncio.Queue[str]" = asyncio.Queue(maxsize=max_queue_size)
        self.requests: Dict[str, InternalRequest] = {}

    def depth(self) -> int:
        return self.queue.qsize()

    def capacity(self) -> int:
        return self.queue.maxsize

    async def enqueue(self, req: InternalRequest) -> Tuple[str, InternalRequest]:
        if self.queue.full():
            raise asyncio.QueueFull
        self.requests[req.request_id] = req
        await self.queue.put(req.request_id)
        return req.request_id, req

    def get_status(self, request_id: str) -> InternalRequest | None:
        return self.requests.get(request_id)


