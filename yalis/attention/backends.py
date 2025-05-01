# These imports trigger @register_attention decorators
from . import sdpa_and_flex
from . import flash
from . import fa3

from enum import Enum

class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
    FA3 = "fa3"