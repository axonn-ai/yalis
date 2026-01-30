import logging
import os
import sys


class ColoredFormatter(logging.Formatter):
    """Logging formatter with ANSI colors."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            color = self.COLORS.get(record.levelname, "")
            reset = self.RESET
            dim = self.DIM
            bold = self.BOLD
        else:
            color = reset = dim = bold = ""

        # Format: timestamp | level | component | message
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = f"{color}{record.levelname:<8}{reset}"
        name = f"{dim}{record.name}{reset}"
        message = record.getMessage()

        # Exception info
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return f"{dim}{timestamp}{reset} {level} {name}: {message}"


def _ensure_configured() -> None:
    if getattr(logging, "_yalis_configured", False):
        return

    level_str = os.getenv("YALIS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)

    # Detect if we're in a TTY (terminal) for colors
    use_colors = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    use_colors = use_colors or os.getenv("YALIS_LOG_COLORS", "").lower() in ("1", "true", "yes")

    # Create handler with colored formatter
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ColoredFormatter(use_colors=use_colors))

    # Configure root logger for yalis namespace
    root_logger = logging.getLogger("yalis")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False  # Don't propagate to root

    setattr(logging, "_yalis_configured", True)


def get_logger(component: str) -> logging.Logger:
    """
    Return a namespaced logger for a given component.

    Respects:
    - YALIS_LOG_LEVEL env var (default INFO)
    - YALIS_LOG_COLORS env var (force colors on/off)
    """
    _ensure_configured()
    return logging.getLogger(f"yalis.{component}")
