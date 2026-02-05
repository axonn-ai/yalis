import os
import json
from typing import Dict, Iterable, Tuple, Optional, Any
import torch
from safetensors.torch import save_file as save_safetensors
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, Future
import threading


@dataclass(frozen=True)
class _AlreadyStored:
    name: str


def _write_shard(
    buf: Dict[str, torch.Tensor],
    shard_path: str,
    metadata: Dict[str, str],
) -> Tuple[float, str]:
    """
    Write a shard to disk. Called from background thread.
    Returns (elapsed_time, shard_path) for logging.
    """
    start = time.time()
    save_safetensors(buf, shard_path, metadata=metadata)
    elapsed = time.time() - start
    return elapsed, shard_path


# SafeTensor Shard Writer with Async Flush
class _SafeTensorShardWriter:
    def __init__(
        self,
        out_dir: str,
        filename_prefix: str = "model",
        max_shard_size_bytes: int = 4 * (1024**3),  # 4 GB default
        per_shard_metadata: Optional[Dict[str, str]] = None,
        make_contiguous: bool = True,
        async_write: bool = True,
        max_pending_writes: int = 2,
    ):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = filename_prefix
        self.max_shard = max_shard_size_bytes
        self.meta = per_shard_metadata or {}
        self.make_contiguous = make_contiguous
        self.async_write = async_write

        self.buf: Dict[str, torch.Tensor] = {}
        self.buf_size = 0
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0
        self.shard_idx = 0
        self.closed = False
        self.per_tensor_overhead = 1024  # ~1KB JSON/name fudge

        # Async write infrastructure
        self._pending_futures: list[Future] = []
        self._pending_weight_maps: list[Tuple[Future, Dict[str, str]]] = []
        self._write_executor: Optional[ThreadPoolExecutor] = None
        self._max_pending = max_pending_writes
        self._write_errors: list[Exception] = []
        self._lock = threading.Lock()

        if self.async_write:
            # Single worker to serialize writes (avoid overwhelming I/O)
            self._write_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="shard_writer"
            )

    def _next_shard_name(self) -> str:
        self.shard_idx += 1
        return f"{self.prefix}-{self.shard_idx:05d}.safetensors"

    def _check_pending_writes(self, wait_all: bool = False):
        """
        Check status of pending async writes.
        If wait_all=True, block until all complete.
        Otherwise, just collect completed ones and check for errors.
        """
        if not self._pending_weight_maps:
            return

        still_pending = []
        for future, wmap in self._pending_weight_maps:
            if wait_all:
                # Block until this write completes
                try:
                    elapsed, shard_path = future.result()
                    with self._lock:
                        self.weight_map.update(wmap)
                except Exception as e:
                    self._write_errors.append(e)
            elif future.done():
                # Collect completed write
                try:
                    elapsed, shard_path = future.result()
                    with self._lock:
                        self.weight_map.update(wmap)
                except Exception as e:
                    self._write_errors.append(e)
            else:
                still_pending.append((future, wmap))

        self._pending_weight_maps = still_pending

        # Raise if any errors occurred
        if self._write_errors:
            errors = self._write_errors
            self._write_errors = []
            raise RuntimeError(
                f"Async shard write failed: {errors[0]}"
            ) from errors[0]

    def _wait_if_too_many_pending(self):
        """Block if we have too many pending writes (backpressure)."""
        while len(self._pending_weight_maps) >= self._max_pending:
            # Wait for at least one to complete
            if self._pending_weight_maps:
                future, wmap = self._pending_weight_maps[0]
                try:
                    elapsed, shard_path = future.result()
                    with self._lock:
                        self.weight_map.update(wmap)
                except Exception as e:
                    self._write_errors.append(e)
                self._pending_weight_maps.pop(0)

        # Check for errors
        if self._write_errors:
            errors = self._write_errors
            self._write_errors = []
            raise RuntimeError(
                f"Async shard write failed: {errors[0]}"
            ) from errors[0]

    def _flush(self):
        if not self.buf:
            return

        shard_name = self._next_shard_name()
        shard_path = os.path.join(self.out_dir, shard_name)

        # Build weight map for this shard
        shard_weight_map = {k: shard_name for k in self.buf.keys()}

        if self.async_write and self._write_executor is not None:
            # Check/collect completed writes and apply backpressure
            self._check_pending_writes(wait_all=False)
            self._wait_if_too_many_pending()

            # Hand off buffer to background thread
            # We must give the thread its own copy since we'll clear self.buf
            buf_to_write = self.buf
            self.buf = {}  # Create new buffer for main thread
            self.buf_size = 0

            future = self._write_executor.submit(
                _write_shard, buf_to_write, shard_path, self.meta
            )
            self._pending_weight_maps.append((future, shard_weight_map))
        else:
            # Synchronous write
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

        # Flush any remaining buffer
        self._flush()

        # Wait for all pending async writes to complete
        if self.async_write:
            self._check_pending_writes(wait_all=True)

        # Shutdown executor
        if self._write_executor is not None:
            self._write_executor.shutdown(wait=True)
            self._write_executor = None

        self.closed = True

        # Write index file
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

    Args:
        name: Output directory or file path.
        max_shard_size_bytes: Maximum size per shard file (default 4GB).
        per_shard_metadata: Optional metadata dict to include in each shard.
        filename_prefix: Prefix for shard filenames (default "model").
        make_contiguous: Whether to make tensors contiguous before saving.
        async_write: If True (default), write shards in background thread.
            This allows processing to continue while I/O happens.
        max_pending_writes: Maximum number of pending async writes before
            blocking (backpressure). Default is 2.
    """

    def __init__(
        self,
        name: str,
        *,
        max_shard_size_bytes: int = 4 * (1024**3),
        per_shard_metadata: Optional[Dict[str, str]] = None,
        filename_prefix: str = "model",
        make_contiguous: bool = True,
        async_write: bool = True,
        max_pending_writes: int = 2,
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
            async_write=async_write,
            max_pending_writes=max_pending_writes,
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
