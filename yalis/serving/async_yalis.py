from __future__ import annotations

import asyncio
from typing import Optional, Tuple

from transformers import AutoTokenizer

from .core_client import CoreClient
from .engine_core import EngineCore
from .schemas import InternalRequest
from .transport import InProcTransport
from .prompt import build_prompt
from yalis import ModelConfig, InferenceConfig


class AsyncYalis:
    """
    Lifecycle owner for serving.

    - Creates input/output transports
    - Owns EngineCore (sync thread) and CoreClient (async facade)
    - Handles request preprocessing (prompt building, tokenization)
    """

    def __init__(self, model_config: ModelConfig, inference_config: InferenceConfig, max_queue_size: int = 0) -> None:
        self.model_config = model_config
        self.inference_config = inference_config

        # Load tokenizer for preprocessing
        # TODO(Prajwal): Add an output processor to handle the output
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        # TODO(Prajwal): Make max_queue_size configurable
        self.input_transport = InProcTransport(maxsize=max_queue_size)
        self.output_transport = InProcTransport(maxsize=0)  # unbounded output

        self.core = EngineCore(
            model_config=model_config,
            inference_config=inference_config,
            input_transport=self.input_transport,
            output_transport=self.output_transport,
        )
        self.client = CoreClient(self.input_transport, self.output_transport)
        self.started = False

    def start(self) -> None:
        """Start the engine core thread and client output reader."""
        if self.started:
            return
        self.core.start()
        self.client.start()
        self.started = True

    def stop(self) -> None:
        if not self.started:
            return
        self.client.stop()
        self.core.stop()
        self.started = False

    async def add_request(self, req: InternalRequest) -> Tuple[str, asyncio.Future[InternalRequest]]:
        """Preprocess and submit a request. Returns (request_id, future)."""
        await self._prepare_request(req)
        return await self.client.submit(req)

    async def generate(self, req: InternalRequest, *, timeout_s: Optional[float] = None) -> InternalRequest:
        """Preprocess, submit, and wait for completion."""
        await self._prepare_request(req)
        request_id, fut = await self.client.submit(req)
        return await asyncio.wait_for(fut, timeout=timeout_s)

    def get_request(self, request_id: str) -> Optional[InternalRequest]:
        """Get request by ID (for status endpoint)."""
        return self.client.get_request(request_id)

    async def _prepare_request(self, req: InternalRequest) -> None:
        """Build prompt and tokenize based on input_kind."""
        # Build prompt text
        if req.input_kind == "chat":
            req.prompt_text = build_prompt(self.tokenizer, req)
        elif req.input_kind == "text":
            req.prompt_text = req.input  # type: ignore
        elif req.input_kind == "tokens":
            # Already tokenized
            req.prompt_token_ids = req.input  # type: ignore
            return
        else:
            raise ValueError(f"Unsupported input_kind: {req.input_kind}")

        # Tokenize
        enc = await asyncio.to_thread(
            self.tokenizer,
            req.prompt_text or "",
            return_tensors="pt",
            padding=False,
        )
        req.prompt_token_ids = enc.input_ids[0].tolist()
