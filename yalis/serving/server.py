import time
import uuid
from typing import Callable
import asyncio

from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from importlib.metadata import version, PackageNotFoundError
import functools

from .schemas import ChatCompletionsRequest, ErrorResponse, InternalRequest
from .queue import WaitQueue
from .scheduler import AsyncScheduler
import os
from .engine_adapter import AsyncEngineAdapter
from yalis import ModelConfig, InferenceConfig
from .executor import Executor
from .engine_loop import EngineLoop
from yalis.attention.kv_cache.kv_slots_manager import KVSlotsManager
from .logger import get_logger
from .prompt import build_prompt
from .output import build_openai_chat_response

router = APIRouter()
logger = get_logger("server")

async def req_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000.0
    response.headers["X-Request-ID"] = req_id
    # minimal access log
    method = request.method
    path = request.url.path
    status = response.status_code
    print(f"{req_id} {method} {path} {status} {duration_ms:.2f}ms")
    return response


@router.get("/healthz")
async def healthz_handler():
    return JSONResponse({"status": "ok"})


@router.get("/version")
async def version_handler():
    try:
        pkg_ver = version("yalis")
    except PackageNotFoundError:
        pkg_ver = "dev"
    info = {"package": "yalis", "version": pkg_ver}
    return JSONResponse(info)


@router.post("/v1/chat/completions")
async def chat_completions_handler(request: Request, payload: ChatCompletionsRequest):
    wait_queue: WaitQueue = request.app.state.wait_queue
    adapter: AsyncEngineAdapter = request.app.state.engine_adapter
    # Build internal request and enqueue
    internal = InternalRequest.from_chat_request(payload)
    internal.prompt_text = build_prompt(adapter.tokenizer, internal)
    enc = await asyncio.to_thread(adapter.tokenizer, internal.prompt_text or "", return_tensors="pt", padding=False)
    internal.prompt_token_ids = enc.input_ids[0].tolist()
    try:
        req_id, _ = await wait_queue.enqueue(internal)
        logger.info(f"enqueue chat_completions req_id={req_id}")
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="queue full")
    # Poll until completion (non-streaming)
    while True:
        st = wait_queue.get_status(req_id)
        if st is None:
            raise HTTPException(status_code=404, detail="request vanished")
        if st.status == "DONE":
            resp = build_openai_chat_response(st, adapter.tokenizer)
            return JSONResponse(resp)
        if st.status == "CANCELLED":
            return JSONResponse({"error": {"message": "cancelled"}}, status_code=499)
        await asyncio.sleep(0.01)

@router.post("/v1/requests")
async def submit_request(request: Request, payload: ChatCompletionsRequest):
    wait_queue: WaitQueue = request.app.state.wait_queue
    adapter: AsyncEngineAdapter = request.app.state.engine_adapter
    try:
        internal = InternalRequest.from_chat_request(payload)
        # Build prompt string and token ids at the endpoint
        internal.prompt_text = build_prompt(adapter.tokenizer, internal)
        enc = await asyncio.to_thread(adapter.tokenizer, internal.prompt_text or "", return_tensors="pt", padding=False)
        internal.prompt_token_ids = enc.input_ids[0].tolist()
        request_id, _ = await wait_queue.enqueue(internal)
        logger.info(f"enqueue req_id={request_id} model={internal.model}")
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="queue full")
    return JSONResponse({"request_id": request_id, "status": "WAITING"})


@router.get("/v1/requests/{request_id}/status")
async def get_request_status(request: Request, request_id: str):
    wait_queue: WaitQueue = request.app.state.wait_queue
    st: InternalRequest | None = wait_queue.get_status(request_id)
    if st is None:
        raise HTTPException(status_code=404, detail="request_id not found")
    return JSONResponse(
        {
            "request_id": st.request_id,
            "status": st.status,
            "created_ts": st.created_ts,
            "started_ts": st.started_ts,
            "finished_ts": st.finished_ts,
        }
    )


@router.get("/v1/metrics")
async def get_metrics(request: Request):
    sched: AsyncScheduler = request.app.state.scheduler
    return JSONResponse(sched.metrics.to_dict())


def create_app(model_config: ModelConfig, inference_config: InferenceConfig) -> FastAPI:
    app = FastAPI()
    app.middleware("http")(req_id_middleware)

    # waiting queue instance stored in app state
    app.state.wait_queue = WaitQueue()



    app.state.scheduler = AsyncScheduler(app.state.wait_queue)
    # Build the engine adapter from configs; start it on app startup
    app.state.engine_adapter = AsyncEngineAdapter(model_config, inference_config)
    # lifecycle
    async def _startup():
        # Start engine adapter (builds the engine)
        await app.state.engine_adapter.start()
        # Concurrency cap: min(env override, engine capacity)
        engine_cap = app.state.engine_adapter.inference_config.max_batch_size
        env_cap = int(os.getenv("YALIS_MAX_CONCURRENCY", str(engine_cap)))
        app.state.scheduler.max_concurrency = max(1, min(env_cap, engine_cap))

        # Build or obtain KVSlotsManager (paged-only serving)
        if not inference_config.use_paged_kv_caching:
          raise RuntimeError("Online serving requires use_paged_kv_caching=True")
        engine = app.state.engine_adapter.engine
        kv_slots = getattr(engine, "kv_slots_manager", None)
        if kv_slots is None:
            # Derive from model buffers
            block_table = getattr(engine.model, "kvcache_block_table", None)
            free_pages = getattr(engine.model, "kvcache_free_pages", None)
            if block_table is None or free_pages is None:
                raise RuntimeError("KV cache tensors not found on model; ensure set_kv_cache executed.")
            max_num_blocks_per_seq = int(block_table.size(1))
            num_blocks = int(free_pages.size(0))
            first_layer = engine.model.transformer.h[0]
            page_block_size = int(first_layer.attn.kv_cache.v.size(1))
            kv_slots = KVSlotsManager(
                capacity=engine_cap,
                paged=True,
                max_num_blocks_per_seq=max_num_blocks_per_seq,
                num_blocks=num_blocks,
                page_block_size=page_block_size,
            )
        app.state.kv_slots_manager = kv_slots
        app.state.scheduler.kv_slots = app.state.kv_slots_manager

        # Link adapter and executor to scheduler and start engine loop
        app.state.scheduler.adapter = app.state.engine_adapter
        app.state.executor = Executor(app.state.engine_adapter)
        app.state.engine_loop = EngineLoop(app.state.scheduler, app.state.executor)
        await app.state.engine_loop.start()

    async def _shutdown():
        await app.state.engine_loop.stop()
        await app.state.engine_adapter.stop()

    app.add_event_handler("startup", _startup)
    app.add_event_handler("shutdown", _shutdown)
    app.include_router(router)
    return app


