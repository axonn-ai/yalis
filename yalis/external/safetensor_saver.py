import os
import json
from typing import Dict, Iterable, Tuple, Optional, Any
import torch
from safetensors.torch import save_file as save_safetensors
from dataclasses import dataclass


@dataclass(frozen=True)
class _AlreadyStored:
    name: str


# SafeTensor Shard Writer
class _SafeTensorShardWriter:
    def __init__(
        self,
        out_dir: str,
        filename_prefix: str = "model",
        max_shard_size_bytes: int = 4 * (1024**3),  # 4 GB default
        per_shard_metadata: Optional[Dict[str, str]] = None,
        make_contiguous: bool = True,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = filename_prefix
        self.max_shard = max_shard_size_bytes
        self.meta = per_shard_metadata or {}
        self.make_contiguous = make_contiguous

        self.buf: Dict[str, torch.Tensor] = {}
        self.buf_size = 0
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0
        self.shard_idx = 0
        self.closed = False
        self.per_tensor_overhead = 1024  # ~1KB JSON/name fudge

    def _next_shard_name(self) -> str:
        self.shard_idx += 1
        return f"{self.prefix}-{self.shard_idx:05d}.safetensors"

    def _flush(self):
        if not self.buf:
            return
        shard_name = self._next_shard_name()
        shard_path = os.path.join(self.out_dir, shard_name)
        # write atomically
        save_safetensors(self.buf, shard_path, metadata=self.meta)
        for k in self.buf.keys():
            self.weight_map[k] = shard_name
        self.buf.clear()
        self.buf_size = 0

    def add(self, name: str, t: torch.Tensor):
        if self.closed:
            raise RuntimeError("Shard writer already closed")

        t = t.detach()
        if t.device.type != "cpu":
            t = t.to("cpu", copy=False)
        if not t.is_contiguous() and self.make_contiguous:
            t = t.contiguous()

        nbytes = t.element_size() * t.numel()
        est = nbytes + self.per_tensor_overhead

        self.buf[name] = t
        self.buf_size += est
        self.total_size += nbytes

        if self.buf_size > self.max_shard:
            self._flush()

    def close(self):
        if self.closed:
            return
        self._flush()
        self.closed = True
        index = {
            "metadata": {},
            "weight_map": self.weight_map,
            "total_size": self.total_size,
            "shard_count": self.shard_idx,
            "prefix": self.prefix,
        }
        with open(
            os.path.join(
                self.out_dir, f"{self.prefix}.safetensors.index.json"
            ),
            "w",
        ) as f:
            json.dump(index, f, indent=2)


# Incremental SafeTensor Saver
class incremental_save:
    """
    Usage:
        with incremental_save(out_dir, max_shard_size_bytes=8*(1024**3)) as saver: # noqa: E501
            for name, p in model.named_parameters():  # you MUST know `name`
                saver.store_early(name, p)            # adds to shard, flushes as needed
            for name, b in model.named_buffers():
                saver.store_early(name, b)
            saver.save()  # finalize (writes index); `obj` is optional
    """

    def __init__(
        self,
        name: str,
        *,
        max_shard_size_bytes: int = 4 * (1024**3),
        per_shard_metadata: Optional[Dict[str, str]] = None,
        filename_prefix: str = "model",
        make_contiguous: bool = True,
    ):
        out_dir = (
            name
            if os.path.isdir(name) or name.endswith("/")
            else os.path.dirname(name) or "."
        )
        self._writer = _SafeTensorShardWriter(
            out_dir=out_dir,
            filename_prefix=filename_prefix,
            max_shard_size_bytes=max_shard_size_bytes,
            per_shard_metadata=per_shard_metadata,
            make_contiguous=make_contiguous,
        )
        self.has_saved = False
        self._committed_names: set[str] = set()

    def __enter__(self):
        return self

    def _store_early(self, tensor: torch.Tensor):
        raise TypeError("store_early expects (name: str, tensor: Tensor)")

    def store_early(self, name: str, tensor: torch.Tensor):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError("store_early expects (name: str, tensor: Tensor)")
        # Allows overwriting of the same name so that last mapping is respected

        self._writer.add(name, tensor)  # may flush the shard
        self._committed_names.add(name)
        return _AlreadyStored(name)

    def _iter_named_tensors(
        self, obj: Any
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        # Optional: allow adding late extras via a mapping in save(obj)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in self._committed_names:
                    continue  # already written via store_early
                if isinstance(v, torch.Tensor):
                    yield k, v
            return
        # Optional: iterable of (name, tensor)
        try:
            iterator = iter(obj)
            first = next(iterator)
        except TypeError:
            raise TypeError(
                "save(obj) expects a mapping or iterable of (name, tensor)"
            )
        except StopIteration:
            return

        def _emit(pair):
            k, v = pair
            if k not in self._committed_names and isinstance(v, torch.Tensor):
                yield k, v

        for out in _emit(first):
            yield out
        for pair in iterator:
            for out in _emit(pair):
                yield out

    def save(self, obj: Optional[Any] = None):
        """
        Finalize. If `obj` is provided, any (name, tensor) not already stored
        via `store_early` will be added now (and may trigger final flushes).
        """
        if self.has_saved:
            raise RuntimeError("have already saved")
        if obj is not None:
            for name, tensor in self._iter_named_tensors(obj):
                self._writer.add(name, tensor)
                self._committed_names.add(name)
        self._writer.close()
        self.has_saved = True

    def __exit__(self, et, ev, tb):
        # finalize if user forgot to call save()
        if not self.has_saved:
            self.save()
