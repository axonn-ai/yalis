from __future__ import annotations

import time
from typing import Dict, Any

from .schemas import InternalRequest


def build_openai_chat_response(req: InternalRequest, tokenizer) -> Dict[str, Any]:
	"""
	Build a minimal non-streaming OpenAI Chat Completions response from InternalRequest.
	"""
	created = int(req.created_ts)
	model = req.model
	request_id = req.request_id
	# Decode assistant text from generated token ids
	if req.output_text is not None:
		text = req.output_text
	else:
		gen_ids = req.output_token_ids or []
		text = tokenizer.decode(gen_ids, skip_special_tokens=True) if gen_ids else ""
	# Finish reason
	if req.finish_reason:
		finish_reason = req.finish_reason
	else:
		eos_id = tokenizer.eos_token_id
		finish_reason = "stop" if (req.last_token_id == eos_id) else "length"
	# Usage
	prompt_tokens = len(req.prompt_token_ids or [])
	completion_tokens = len(req.output_token_ids or [])
	total_tokens = prompt_tokens + completion_tokens

	return {
		"id": f"chatcmpl-{request_id}",
		"object": "chat.completion",
		"created": created,
		"model": model,
		"choices": [
			{
				"index": 0,
				"message": {"role": "assistant", "content": text},
				"finish_reason": finish_reason,
				"logprobs": None,
			}
		],
		"usage": {
			"prompt_tokens": prompt_tokens,
			"completion_tokens": completion_tokens,
			"total_tokens": total_tokens,
		},
	}


