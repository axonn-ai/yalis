import logging
import os
from typing import Optional

_configured = False


def _ensure_configured() -> None:
	if not hasattr(logging, "_yalis_configured") or not getattr(logging, "_yalis_configured"):  # type: ignore[attr-defined]
		level_str = os.getenv("YALIS_LOG_LEVEL", "INFO").upper()
		level = getattr(logging, level_str, logging.INFO)
		logging.basicConfig(
			level=level,
			format="%(asctime)s %(levelname)s %(name)s: %(message)s",
		)
		setattr(logging, "_yalis_configured", True)  # type: ignore[attr-defined]


def get_logger(component: str) -> logging.Logger:
	"""
	Return a namespaced logger for a given component.
	Respects YALIS_LOG_LEVEL env var (default INFO).
	"""
	_ensure_configured()
	return logging.getLogger(f"yalis.{component}")


