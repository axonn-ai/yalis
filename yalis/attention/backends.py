# These imports trigger @register_attention decorators
from . import sdpa_and_flex
from . import flash

import os
if os.environ.get("USE_FA3", '0') == '1':
    from . import fa3

from enum import Enum

class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
    FA3 = "fa3"