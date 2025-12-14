from __future__ import annotations

import asyncio
from typing import Dict, Optional, Tuple

from .engine_core import SubmitItem, OutputItem
from .schemas import InternalRequest
from .transport import InProcTransport
from .logger import get_logger


class CoreClient:
    """
    Async client that talks to the sync EngineCore via transports.

    - Submits requests to input transport
    - Reads completed requests from output transport
    - Routes outputs to per-request futures
    - Keeps all requests for status lookup
    """

    def __init__(self, input_transport: InProcTransport, output_transport: InProcTransport) -> None:
        self.input_transport = input_transport
        self.output_transport = output_transport

        self.pending: Dict[str, asyncio.Future[InternalRequest]] = {}

        # TODO(Prajwal): Use a proper request store instead of a dict
        self.requests: Dict[str, InternalRequest] = {}  # all requests for status lookup

        self.output_task: Optional[asyncio.Task] = None
        self.logger = get_logger("core_client")

    def start(self) -> None:
        """Start the output reader task."""
        if self.output_task is None or self.output_task.done():
            self.output_task = asyncio.create_task(self.read_outputs())

    def stop(self) -> None:
        """Stop the output reader task."""
        if self.output_task is not None:
            self.output_task.cancel()
            self.output_task = None

    def get_request(self, request_id: str) -> Optional[InternalRequest]:
        """Get request by ID (for status endpoint)."""
        # TODO(Prajwal): This works for now because both the client and the core
        # are running in the same process and janus.Queue stores the reference
        # to the request object When we switch to ZMQ, and the core is running
        # in a different process, the core needs to return snapshot of the
        # request object everytime so that our copy of the request object is
        # updated with the latest state
        return self.requests.get(request_id)

    async def submit(self, req: InternalRequest) -> Tuple[str, asyncio.Future[InternalRequest]]:
        """
        Submit a request.

        Returns (request_id, future) so caller can await the future directly.
        """
        fut: asyncio.Future[InternalRequest] = asyncio.get_running_loop().create_future()
        self.pending[req.request_id] = fut

        # store for status lookup
        self.requests[req.request_id] = req  

        item = SubmitItem(req=req)
        if not self.input_transport.try_put(item):
            del self.pending[req.request_id]
            del self.requests[req.request_id]
            raise RuntimeError("input queue full")
        self.logger.info(f"CoreClient submitted request_id={req.request_id}")
        return req.request_id, fut

    async def read_outputs(self) -> None:
        """Background task: read from output transport and complete futures."""
        self.logger.info("CoreClient output reader started")
        while True:
            try:
                # Wait until an output item is available
                out: OutputItem = await self.output_transport.get_async()
            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("output reader error")
                continue

            # Update stored request with completed state
            self.requests[out.request_id] = out.req

            fut = self.pending.pop(out.request_id, None)
            if fut is not None and not fut.done():
                # Complete the future with the completed request
                fut.set_result(out.req)
            self.logger.info(f"CoreClient delivered request_id={out.request_id}")

        self.logger.info("CoreClient output reader stopped")
