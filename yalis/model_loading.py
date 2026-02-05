import json
from pathlib import Path
from typing import Dict, Iterable, Tuple, Optional, Any
from collections.abc import MutableMapping

import torch
import torch.nn as nn
from safetensors import safe_open
from yalis.utils import print_rank0
from tqdm import tqdm


class LazySafeTensorDict(MutableMapping):
    """
    Dict-like view over safetensors that:
      - Enumerates all keys without loading tensors.
      - Loads an individual tensor only when accessed (get/pop).
      - Loads it on the device/dtype of the target param/buffer.

    Supports:
      - Single file: /cpt/model.safetensors
      - Sharded: /cpt/model.safetensors.index.json + model-00001.safetensors,..
    """

    def __init__(
        self,
        ckpt_path: Path,
        *,
        name_to_device_dtype: Dict[str, Tuple[torch.device, torch.dtype]],
    ) -> None:
        self._root = Path(ckpt_path)
        self._map: Dict[str, str] = (
            {}
        )  # name -> shard filename (or single file)
        self._keys: Iterable[str] = []
        self._single_file: Optional[Path] = None
        self._opened: Dict[str, Any] = {}  # shard path -> safe_open handle
        self._cache: Dict[str, torch.Tensor] = {}
        self._name_to_devdtype = name_to_device_dtype

        if self._root.is_file() and self._root.suffix == ".safetensors":
            # Single-file checkpoint
            self._single_file = self._root
            with safe_open(
                str(self._single_file), framework="pt", device="cpu"
            ) as f:
                self._keys = list(f.keys())
            for k in self._keys:
                self._map[k] = self._single_file.name
        else:
            # Sharded (expect an index json nearby)
            index_json = None
            if self._root.is_file() and self._root.name.endswith(
                ".index.json"
            ):
                index_json = self._root
                base_dir = self._root.parent
            elif self._root.is_dir():
                # look for *.safetensors.index.json
                idxs = [p for p in self._root.glob("*.safetensors.index.json")]
                if not idxs:
                    raise FileNotFoundError(
                        f"No *.safetensors.index.json in {self._root}"
                    )
                index_json = idxs[0]
                base_dir = self._root
            else:
                raise FileNotFoundError(f"Checkpoint not found: {self._root}")

            with open(index_json, "r") as f:
                idx = json.load(f)
            wm = idx.get("weight_map", {})
            if not wm:
                raise RuntimeError("Index JSON missing weight_map")

            # Names may come like "model.xxx". Provide both aliases if needed.
            self._map = dict(wm)
            # Build key list and alias stripping 'model.' prefix if all have it
            self._keys = list(self._map.keys())

            # Make sure shard files exist
            for shard in set(self._map.values()):
                if not (base_dir / shard).is_file():
                    raise FileNotFoundError(
                        f"Shard missing: {base_dir / shard}"
                    )

            self._base_dir = base_dir

        # If most keys are prefixed with "model.", support alias without it
        if self._keys and all(k.startswith("model.") for k in self._keys):
            extra = {k[6:]: self._map[k] for k in list(self._map.keys())}
            self._map.update(extra)
            self._keys = list(self._map.keys())

    # ---- Minimal mapping API used by nn.Module._load_from_state_dict ----

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._map

    def __getitem__(self, key: str) -> torch.Tensor:
        # Allow get without loading: but when asked,
        # load into correct device/dtype
        if key in self._cache:
            return self._cache[key]
        if key not in self._map:
            raise KeyError(key)
        t = self._load_key(key)
        self._cache[key] = t
        return t

    def get(self, key: str, default=None):
        return self[key] if key in self else default

    def pop(self, key: str, default=None):
        if key in self:
            v = self[key]  # loads if needed
            del self._map[key]
            self._cache.pop(key, None)
            return v
        if default is not None:
            return default
        raise KeyError(key)

    def keys(self):
        return self._map.keys()

    def __iter__(self):
        return iter(self._map)

    def __len__(self) -> int:
        return len(self._map)

    # We don’t support mutation into the checkpoint
    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        # PyTorch won't call this directly; we implement pop above.
        raise NotImplementedError

    # ---- internals ----

    def _open_handle(self, shard_file: str):
        path = (
            self._single_file
            if self._single_file and self._single_file.name == shard_file
            else (self._base_dir / shard_file)
        )
        spath = str(path)
        if spath not in self._opened:
            # Keep handle open to avoid re-parsing the header repeatedly
            self._opened[spath] = safe_open(
                spath, framework="pt", device="cpu"
            )
        return self._opened[spath]

    def _load_key(self, name: str) -> torch.Tensor:
        shard = self._map[name]
        f = self._open_handle(shard)
        # Load CPU tensor first
        t = f.get_tensor(name)

        # Move & cast to expected device/dtype if known
        dev, dt = self._name_to_devdtype.get(name, (None, None))
        if (
            dev is None
            and name.startswith("model.")
            and name[6:] in self._name_to_devdtype
        ):
            dev, dt = self._name_to_devdtype[name[6:]]
        if dt is not None and t.dtype != dt:
            t = t.to(dtype=dt)
        if dev is not None and t.device != dev:
            # non-blocking move if CUDA target
            non_blocking = dev.type == "cuda"
            t = t.to(device=dev, non_blocking=non_blocking)
        try:
            _rank = torch.distributed.get_rank()
        except Exception:
            _rank = -1
        try:
            shard_info = f"{shard}"
        except Exception:
            shard_info = shard
        print_rank0(
            f"[rank {_rank}] Loaded key '{name}' from shard '{shard_info}' -> "
            f"shape={tuple(t.shape)}, dtype={t.dtype}, target_dev={dev}, "
            f"target_dtype={dt}"
        )
        return t

    def close(self):
        self._opened.clear()


def _collect_param_buffer_devdtype(
    model: nn.Module,
) -> Dict[str, Tuple[torch.device, torch.dtype]]:
    """
    Map each state_dict key -> (device, dtype) from the current model,
    so we can load tensors directly onto the right device/dtype.
    """
    mapping = {}
    for n, p in model.named_parameters(recurse=True):
        dev = None if p.device.type == "meta" else p.device
        mapping[n] = (dev, p.dtype)
    for n, b in model.named_buffers(recurse=True):
        dev = None if b.device.type == "meta" else b.device
        mapping[n] = (dev, b.dtype)
    return mapping


def load_checkpoint_safetensors(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool = True,
) -> None:
    print_rank0(
        f"Loading checkpoint (lazy safetensors) from {checkpoint_path}"
    )

    # Build per-key device/dtype targets from the CURRENT model
    name_to_devdtype = _collect_param_buffer_devdtype(model)

    # Build a lazy dict view over the safetensors checkpoint
    lazy_sd = LazySafeTensorDict(
        checkpoint_path, name_to_device_dtype=name_to_devdtype
    )

    # Progress hook (unchanged from your current code)
    modules_to_hook = [
        m
        for m in model.modules()
        if any(True for _ in m.parameters(recurse=False))
        or any(True for _ in m.buffers(recurse=False))
    ]
    assert (
        len(modules_to_hook) > 0
    ), "Could not find modules with direct parameters or buffers"

    pbar = tqdm(total=len(modules_to_hook), desc="Loading State Dict")

    def _post_hook(module, incompatible_keys):
        pbar.update(1)

    hooks = [
        m.register_load_state_dict_post_hook(_post_hook)
        for m in modules_to_hook
    ]

    try:
        model.load_state_dict(lazy_sd, strict=strict, assign=True)
    except Exception as e:
        print_rank0(f"Error loading checkpoint: {e}")
        raise
    finally:
        for h in hooks:
            h.remove()
        pbar.close()
        lazy_sd.close()
