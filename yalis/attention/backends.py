# These imports trigger @register_attention decorators
from . import sdpa_and_flex
from . import flash
from . import flex

from enum import Enum

class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
    SDPA_AND_FLEX = "sdpa_and_flex"