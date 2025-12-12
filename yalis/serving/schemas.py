from typing import List, Optional, Dict, Any, Literal, Union
from pydantic import BaseModel, Field
import time
import uuid

from .logger import get_logger
logger = get_logger("schemas")

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=128, alias="max_completion_tokens")
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None


class ErrorResponse(BaseModel):
    error: Dict[str, Any]


class SamplingParams(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_tokens: int = 128


InputKind = Literal["text", "tokens", "embeds", "chat"]


class InternalRequest(BaseModel):
    request_id: str
    model: str
    input_kind: InputKind
    input: Union[str, List[int], List[float], List[ChatMessage], Dict[str, Any]]
    sampling: SamplingParams
    status: Literal["WAITING", "RUNNING", "DONE", "CANCELLED"] = "WAITING"
    created_ts: float = Field(default_factory=lambda: time.time())
    started_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    # M4.4 runtime fields
    phase: Literal["PREFILL", "DECODE", "DONE"] = "PREFILL"
    slot_id: Optional[int] = None
    prompt_text: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    # Accumulates generated token ids for this request (excludes EOS)
    output_token_ids: List[int] = Field(default_factory=list)
    # Incremental decoding/output fields
    output_text: Optional[str] = None
    finish_reason: Optional[str] = None
    # Stop sequences for output processor
    stop: Optional[List[str]] = None

    has_first_token: bool = False
    last_token_id: Optional[int] = None
    generated_count: int = 0

    @staticmethod
    def from_chat_request(req: ChatCompletionsRequest) -> "InternalRequest":
        logger.debug(f"from_chat_request req={req}")
        sampling = SamplingParams(
            temperature=float(req.temperature or 1.0),
            top_p=float(req.top_p or 1.0),
            top_k=req.top_k,
            max_tokens=int(req.max_tokens or 128),
        )
        stop_list: Optional[List[str]]
        if req.stop is None:
            stop_list = None
        elif isinstance(req.stop, str):
            stop_list = [req.stop]
        else:
            stop_list = list(req.stop)
        return InternalRequest(
            request_id=str(uuid.uuid4()),
            model=req.model,
            input_kind="chat",
            input=req.messages,
            sampling=sampling,
            status="WAITING",
            stop=stop_list,
        )



