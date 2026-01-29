# These imports trigger @register_attention decorators
from . import sdpa_and_flex
from . import flash
from . import thresh
from . import thresh_nowmp
from . import topk
from . import sparge
from . import double_sparse

from enum import Enum

class AttentionBackend(str, Enum):
    SDPA = "sdpa"
    FLASH = "flash"
    FLEX = "flex"
    THRESH = "thresh"
    THRESH_ATTN_NOWMP = "thresh_attn_nowmp"
    TOPK = "topk"
    SPARGE = "sparge"
    DOUBLE_SPARSE = "double_sparse"
