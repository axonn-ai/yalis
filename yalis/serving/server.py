import time
import uuid

from fastapi import FastAPI, Request, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from importlib.metadata import version, PackageNotFoundError

from yalis import ModelConfig, InferenceConfig
from .schemas import ChatCompletionsRequest, InternalRequest
from .async_yalis import AsyncYalis
from .logger import get_logger
from .output import build_openai_chat_response

router = APIRouter()
logger = get_logger("server")


async def req_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000.0
    response.headers["X-Request-ID"] = req_id
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
    return JSONResponse({"package": "yalis", "version": pkg_ver})


@router.post("/v1/chat/completions")
async def chat_completions_handler(request: Request, payload: ChatCompletionsRequest):
    """ChatCompletions API handler."""
    # TODO (Prajwal): Add streaming support
    yalis: AsyncYalis = request.app.state.yalis
    req = InternalRequest.from_chat_request(payload, yalis.tokenizer)

    try:
        result = await yalis.generate(req)
    except RuntimeError as e:
        if "queue full" in str(e):
            raise HTTPException(status_code=429, detail="queue full")
        raise

    logger.info(f"completed req_id={result.request_id}")
    resp = build_openai_chat_response(result, yalis.tokenizer)
    return JSONResponse(resp)


@router.post("/v1/requests")
async def submit_request(request: Request, payload: ChatCompletionsRequest):
    yalis: AsyncYalis = request.app.state.yalis
    req = InternalRequest.from_chat_request(payload, yalis.tokenizer)

    try:
        request_id, fut = await yalis.add_request(req)
    except RuntimeError as e:
        if "queue full" in str(e):
            raise HTTPException(status_code=429, detail="queue full")
        raise

    logger.info(f"enqueue req_id={request_id}")
    return JSONResponse({"request_id": request_id, "status": "WAITING"})


@router.get("/v1/requests/{request_id}/status")
async def get_request_status(request: Request, request_id: str):
    yalis: AsyncYalis = request.app.state.yalis
    req = yalis.get_request(request_id)
    if st is None:
        raise HTTPException(status_code=404, detail="request_id not found")
    return JSONResponse(
        {
            "request_id": req.request_id,
            "status": req.status,
            "created_ts": req.created_ts,
            "started_ts": req.started_ts,
            "finished_ts": req.finished_ts,
        }
    )


def create_app(model_config: ModelConfig, inference_config: InferenceConfig) -> FastAPI:
    """Create the FastAPI app."""
    app = FastAPI()
    app.middleware("http")(req_id_middleware)

    app.state.yalis = AsyncYalis(model_config, inference_config)

    # Startup and shutdown handlers
    async def _startup():
        app.state.yalis.start()
        logger.info("AsyncYalis started")

    async def _shutdown():
        app.state.yalis.stop()
        logger.info("AsyncYalis stopped")

    app.add_event_handler("startup", _startup)
    app.add_event_handler("shutdown", _shutdown)
    app.include_router(router)
    return app
