from enum import Enum, auto


class EnginePhase(Enum):
    PREFILL = auto()
    DECODE_SINGLE = auto()
    DECODE_MULTI = auto()  # For speculative decoding
