# These imports trigger @register_attention decorators
from yalis.attention.backend_impl.sdpa_and_flex import sdpa_attention  # noqa: F401
from yalis.attention.backend_impl.flash import flash_attention  # noqa: F401

from enum import Enum


class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
