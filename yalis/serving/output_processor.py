from __future__ import annotations

from typing import List, Optional, Tuple


class OutputProcessor:
	"""
	Stateless helper that derives decoded text and stop condition from request data.
	State (tokens, text, finish_reason) should live on the request.
	"""

	def __init__(self, tokenizer, *, skip_special_tokens: bool = True) -> None:
		self.tokenizer = tokenizer
		self.skip_special_tokens = skip_special_tokens

	def process(
		self,
		token_ids: List[int],
		stop_sequences: Optional[List[str]] = None,
		*,
		eos_hit: bool = False,
	) -> Tuple[str, bool, Optional[str]]:
		"""
		Given all generated token_ids so far, returns:
		(decoded_text, finished, finish_reason)
		- Trims trailing stop sequence if matched (suffix-based).
		- eos_hit immediately marks finished with reason="stop".
		"""
		text = self.tokenizer.decode(token_ids, skip_special_tokens=self.skip_special_tokens)
		finished = False
		reason: Optional[str] = None
		if eos_hit:
			finished = True
			reason = "stop"
		if not finished and stop_sequences:
			for s in stop_sequences:
				if s and text.endswith(s):
					text = text[: -len(s)]
					finished = True
					reason = "stop"
					break
		return text, finished, reason