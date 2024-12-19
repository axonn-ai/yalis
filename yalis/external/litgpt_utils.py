from litgpt.utils import lazy_load
import torch.nn as nn
from pathlib import Path

def load_checkpoint(model: nn.Module, 
                    checkpoint_path: Path, 
                    strict: bool = True) -> None:
    state_dict = lazy_load(checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=strict)