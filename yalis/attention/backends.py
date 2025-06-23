# These imports trigger @register_attention decorators
from . import sdpa_and_flex  # noqa: F401
from . import flash  # noqa: F401

from enum import Enum


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
