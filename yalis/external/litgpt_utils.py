from litgpt.utils import lazy_load
from collections.abc import Sequence
from typing import Any, Callable, Optional
import torch.nn as nn
from torch.overrides import TorchFunctionMode
from typing_extensions import override
from pathlib import Path
from yalis.utils import print_rank0
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("once", category=DeprecationWarning)


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


def load_litgpt_checkpoint(
    model: nn.Module, checkpoint_path: Path, strict: bool = True
) -> None:
    warnings.warn(
        "Loading .pth checkpoints is deprecated. Move to safetensors instead.",
        DeprecationWarning,
    )
    print_rank0(f"Loading checkpoint from {checkpoint_path}")
    state_dict = lazy_load(checkpoint_path)
    state_dict = state_dict.get("model", state_dict)

    modules_to_hook = [
        m
        for m in model.modules()
        if any(True for _ in m.parameters(recurse=False))  # module has direct params
        or any(True for _ in m.buffers(recurse=False))  # or direct buffers
    ]
    assert (
        len(modules_to_hook) > 0
    ), "Could not find modules with direct parameters or buffers"

    pbar = tqdm(total=len(modules_to_hook), desc="Loading State Dict")

    def _post_hook(module, incompatible_keys):
        # one tick per module
        pbar.update(1)
        # you *can* still mutate incompatible_keys here if you want

    for m in modules_to_hook:
        m.register_load_state_dict_post_hook(_post_hook)

    try:
        model.load_state_dict(state_dict, strict=strict)
    except Exception as e:
        print_rank0(f"Error loading checkpoint: {e}")
        raise e
    finally:
        pbar.close()
