from litgpt.utils import lazy_load
from collections.abc import Sequence
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from typing_extensions import override
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable, Optional
import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from typing_extensions import override

# From https://lernapparat.de/faster-model-init by Thomas Viehmann
class _EmptyInit(TorchFunctionMode):
    """Initialize `nn.Module` with empty tensors, i.e., uninitialized memory.

    Example::

        with _EmptyInit():
            model = BigModel()
        model.load_state_dict(torch.load("checkpoint.pt"))

    """

    def __init__(self, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = enabled

    @override
    def __torch_function__(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        kwargs = kwargs or {}
        if not self.enabled:
            return func(*args, **kwargs)
        if getattr(func, "__module__", None) == "torch.nn.init":
            if "tensor" in kwargs:
                return kwargs["tensor"]
            return args[0]
        return func(*args, **kwargs)


def load_checkpoint(model: nn.Module,
                    checkpoint_path: Path,
                    strict: bool = True) -> None:
    state_dict = lazy_load(checkpoint_path)
    state_dict = state_dict.get("model", state_dict)
    model.load_state_dict(state_dict, strict=strict)
