from typing import List
from .schemas import InternalRequest, ChatMessage


def build_prompt(tokenizer, req: InternalRequest) -> str:
    """
    Map InternalRequest to a single prompt string for the engine.
    Currently supports:
      - input_kind='chat': expects List[ChatMessage]
      - input_kind='text': expects str
    """
    if req.input_kind == "chat":
        messages: List[ChatMessage] = req.input  # type: ignore[assignment]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    if req.input_kind == "text":
        return str(req.input)
    raise NotImplementedError(f"Unsupported input_kind: {req.input_kind}")


