"""
CPU Offload Manager - coordinates layer-by-layer execution with prefetching.
"""

import warnings

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set, Callable
from contextlib import contextmanager

from yalis.utils import print_rank0
from .constants import compiler_disable, PrefetchMode
from .buffer_manager import GPUBufferManager

from yalis.constants import EnginePhase


class CPUOffloadManager:
    """
    Manages CPU<->GPU transfers for layer-by-layer model execution.

    Two independent axes:
    - **What to offload** (offload_modules): which submodules live on CPU.
      None = offload everything in each block.
    - **How to prefetch** (prefetch_mode): ALL, SELECTIVE, or NONE.

    Features:
    - Prefetches next layers while current layer executes
    - Uses separate CUDA streams for compute and transfer
    - Module-path prefix matching for flexible offloading
    - Optional pre-allocated GPU buffers for zero-allocation transfers
    - Discovery: prints block structure with sizes at init
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cuda"),
        num_prefetch_layers: int = 1,
        pin_memory: bool = True,
        use_preallocated_buffers: bool = False,
        offload_modules: Optional[List[str]] = None,
        prefetch_mode: PrefetchMode = PrefetchMode.ALL,
    ):
        self.model = model
        self.device = device
        self.num_prefetch_layers = num_prefetch_layers
        self.pin_memory = pin_memory
        self.use_preallocated_buffers = use_preallocated_buffers
        self.prefetch_mode = prefetch_mode
        self.offload_modules = offload_modules

        # CUDA streams
        self.compute_stream = torch.cuda.Stream(device=device)
        self.transfer_stream = torch.cuda.Stream(device=device, priority=-1)

        # State tracking
        self.layers_on_gpu: Set[int] = set()
        self.transfer_events: Dict[int, torch.cuda.Event] = {}

        # Model structure
        self.blocks = self._get_transformer_blocks()
        self.num_layers = len(self.blocks)

        # CPU storage
        self.cpu_state_dicts: Dict[int, Dict[str, torch.Tensor]] = {}
        self.layer_buffer_idx: Dict[int, int] = {}

        # GPU buffer manager
        self.gpu_buffer_manager: Optional[GPUBufferManager] = None

        # Callback for sparse row indices
        self.row_indices_callback: Optional[
            Callable[[int], Optional[torch.Tensor]]
        ] = None

        # Install default random row selector for selective mode
        if self.prefetch_mode == PrefetchMode.SELECTIVE:
            self._install_default_row_selector()

        self.event_pool = {}
        for i in range(self.num_layers):
            self.event_pool[i] = torch.cuda.Event()

    def _should_offload(self, name: str) -> bool:
        """Check if a parameter should be offloaded based on module path.

        When offload_modules is None, everything is offloaded.
        Otherwise, a parameter is offloaded if its dotted name matches
        any of the specified module prefixes.
        """
        if self.offload_modules is None:
            return True
        return any(
            name == m or name.startswith(m + ".") for m in self.offload_modules
        )

    def _get_transformer_blocks(self) -> nn.ModuleList:
        """Get transformer blocks from model."""
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "h"
        ):
            return self.model.transformer.h
        raise ValueError("Expected model.transformer.h")

    def _install_default_row_selector(self):
        """Install a random row selector as fallback for selective mode."""
        n_expert = getattr(
            getattr(self.model, "config", None), "n_expert", None
        )
        n_expert_per_token = getattr(
            getattr(self.model, "config", None),
            "n_expert_per_token",
            None,
        )

        if n_expert is None or n_expert_per_token is None:
            warnings.warn(
                "[CPUOffloadManager] prefetch_mode='selective' but"
                " model.config.n_expert / n_expert_per_token not"
                " found. No default row selector installed."
                " Supply a callback via set_row_indices_callback()."
            )
            return

        total = n_expert
        k = n_expert_per_token

        def _random_row_selector(layer_idx: int):
            experts = torch.randint(0, total, (k,), device="cuda")
            return torch.sort(experts).values

        self.row_indices_callback = _random_row_selector
        warnings.warn(
            f"[CPUOffloadManager] Using random row selector as"
            f" fallback (n_expert={total}, k={k}). Supply real"
            f" expert IDs via set_row_indices_callback() or"
            f" prefetch_expert_ids in layer_context()."
        )

    def print_offloadable_modules(self):
        """Print block structure with parameter sizes for discovery."""
        if not self.blocks:
            return

        block = self.blocks[0]
        # Collect sizes per module path
        module_sizes: Dict[str, int] = {}
        for name, param in block.named_parameters():
            size_bytes = param.numel() * param.element_size()
            # Accumulate for each prefix level
            parts = name.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                module_sizes.setdefault(prefix, 0)
                module_sizes[prefix] += size_bytes

        total = sum(p.numel() * p.element_size() for p in block.parameters())

        print_rank0("[CPUOffloadManager] Block structure (per layer):")
        # Sort by path for readable output
        for path in sorted(module_sizes.keys()):
            depth = path.count(".")
            indent = "  " * (depth + 1)
            size_gb = module_sizes[path] / 1e9
            offloaded = self._should_offload(path)
            marker = "offload" if offloaded else "keep"
            print_rank0(f"{indent}{path:40s} {size_gb:.3f} GB" f"  [{marker}]")

        print_rank0(f"  Total per layer: {total / 1e9:.3f} GB")

        if self.offload_modules is None:
            print_rank0("  Config: cpu_offload_modules=None")
            print_rank0("  Offloading: all modules")
        else:
            print_rank0(
                f"  Config: cpu_offload_modules=" f"{self.offload_modules}"
            )
            offloaded_size = sum(
                v
                for k, v in module_sizes.items()
                if self._should_offload(k)
                and "." not in k  # top-level only for summary
            )
            print_rank0(f"  Offloading: {offloaded_size / 1e9:.3f} GB")
        print_rank0(f"  Prefetch mode: {self.prefetch_mode.name.lower()}")

    def prepare_for_offloading(self, dtype: torch.dtype = torch.bfloat16):
        """Prepare model for CPU offloading."""
        self.dtype = dtype

        # Print discovery info
        self.print_offloadable_modules()

        if self.offload_modules is None:
            print_rank0("[CPUOffloadManager] Offloading: all modules")
        else:
            print_rank0(
                f"[CPUOffloadManager] Offloading modules:"
                f" {self.offload_modules}"
            )

        # Keep embeddings/output on GPU
        self._move_fixed_components_to_gpu()

        # Initialize buffer manager if using preallocated buffers
        if self.use_preallocated_buffers:
            num_buffer_sets = self.num_prefetch_layers + 1
            print_rank0(
                f"[CPUOffloadManager] Creating"
                f" {num_buffer_sets} buffer sets"
            )
            self.gpu_buffer_manager = GPUBufferManager(
                layer_template=self.blocks[0],
                device=self.device,
                dtype=dtype,
                should_offload=self._should_offload,
                num_buffer_sets=num_buffer_sets,
            )

        # Process each block
        for idx, block in enumerate(self.blocks):
            self._prepare_block(idx, block)

        # Load first layer
        self._move_to_gpu(0, non_blocking=False)

        print_rank0(
            f"[CPUOffloadManager] Prepared" f" {self.num_layers} layers"
        )

    def _move_fixed_components_to_gpu(self):
        """Move embeddings, final norm, lm_head to GPU."""
        if hasattr(self.model, "transformer"):
            if hasattr(self.model.transformer, "wte"):
                self.model.transformer.wte.to(self.device)
            if hasattr(self.model.transformer, "ln_f"):
                self.model.transformer.ln_f.to(self.device)

        if hasattr(self.model, "lm_head"):
            self.model.lm_head.to(self.device)

        # RoPE cache
        if hasattr(self.model, "cos"):
            self.model.cos = self.model.cos.to(self.device)
        if hasattr(self.model, "sin"):
            self.model.sin = self.model.sin.to(self.device)

    def _prepare_block(self, idx: int, block: nn.Module):
        """Prepare a single block for offloading."""
        self.cpu_state_dicts[idx] = {}

        # Process parameters
        for name, param in block.named_parameters():
            if self._should_offload(name):
                self._offload_param(idx, name, param)
            else:
                self._keep_param_on_gpu(param)

        # Process buffers (KV cache must stay on GPU)
        for name, buf in block.named_buffers():
            if "kv_cache" in name:
                if buf.device.type != "cuda":
                    buf.data = buf.data.to(self.device)
                continue

            if self._should_offload(name):
                self._offload_buffer(idx, name, buf)
            elif buf.device.type != "cuda":
                buf.data = buf.data.to(self.device)

        self.layers_on_gpu.discard(idx)

    def _offload_param(self, idx: int, name: str, param: nn.Parameter):
        """Move param to CPU (or store CPU copy for preallocated)."""
        if self.use_preallocated_buffers:
            cpu_tensor = self._to_cpu_pinned(param.data)
            self.cpu_state_dicts[idx][name] = cpu_tensor

            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(idx)
            buf_set = self.gpu_buffer_manager.get_buffer_set(buffer_idx)

            if name in buf_set.tensors:
                param.data = buf_set.tensors[name]

            self.layer_buffer_idx[idx] = buffer_idx
        else:
            if param.device.type != "cpu":
                param.data = param.data.to("cpu")
            param.data = self._pin_if_enabled(param.data)
            self.cpu_state_dicts[idx][name] = param.data

    def _offload_buffer(self, idx: int, name: str, buf: torch.Tensor):
        """Move buffer to CPU (or store CPU copy for preallocated)."""
        if self.use_preallocated_buffers:
            cpu_tensor = buf.data.to("cpu")
            self.cpu_state_dicts[idx][name] = cpu_tensor

            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(idx)
            buf_set = self.gpu_buffer_manager.get_buffer_set(buffer_idx)

            if name in buf_set.tensors:
                buf.data = buf_set.tensors[name]
        else:
            if buf.device.type != "cpu":
                buf.data = buf.data.to("cpu")
            self.cpu_state_dicts[idx][name] = buf

    def _keep_param_on_gpu(self, param: nn.Parameter):
        """Keep param on GPU."""
        if param.device.type != "cuda":
            param.data = param.data.to(self.device)

    def _to_cpu_pinned(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to CPU with optional pinning."""
        cpu_tensor = tensor.to("cpu")
        if self.pin_memory:
            pinned = cpu_tensor.pin_memory()
            del cpu_tensor
            return pinned
        return cpu_tensor

    def _pin_if_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin tensor memory if enabled."""
        if self.pin_memory and torch.cuda.is_available():
            if tensor.is_contiguous() and not tensor.is_pinned():
                try:
                    return tensor.pin_memory()
                except RuntimeError:
                    pass
        return tensor

    @compiler_disable()
    def _move_to_gpu(
        self,
        layer_idx: int,
        non_blocking: bool = True,
    ):
        """Move offloaded params to GPU."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        block = self.blocks[layer_idx]
        cpu_state = self.cpu_state_dicts.get(layer_idx, {})

        if self.use_preallocated_buffers and self.gpu_buffer_manager:
            buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(
                layer_idx
            )
            self.gpu_buffer_manager.copy_all(
                block,
                cpu_state,
                self.transfer_stream,
                buffer_idx,
                non_blocking,
            )
        else:
            for name, param in block.named_parameters():
                if name in cpu_state:
                    param.data = cpu_state[name].to(
                        self.device, non_blocking=non_blocking
                    )

            for name, buf in block.named_buffers():
                if name in cpu_state:
                    buf.data = cpu_state[name].to(
                        self.device, non_blocking=non_blocking
                    )

        self.layers_on_gpu.add(layer_idx)

    @compiler_disable()
    def _move_rows_to_gpu(
        self,
        layer_idx: int,
        row_indices: torch.Tensor,
        non_blocking: bool = True,
        stream=None,
    ):
        """Move specific rows to GPU (requires preallocated buffers)."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        row_indices = (
            row_indices.cpu()
            if row_indices.device.type != "cpu"
            else row_indices
        )

        if stream is None:
            stream = self.transfer_stream

        if not self.use_preallocated_buffers or not self.gpu_buffer_manager:
            return self._move_to_gpu(layer_idx, non_blocking)

        buffer_idx = self.gpu_buffer_manager.get_buffer_idx_for_layer(
            layer_idx
        )
        self.gpu_buffer_manager.copy_rows(
            self.blocks[layer_idx],
            self.cpu_state_dicts[layer_idx],
            row_indices,
            stream,
            buffer_idx,
            non_blocking=non_blocking,
        )

    @compiler_disable()
    def _restore_layer_to_cpu(self, layer_idx: int):
        """Restore layer to CPU (or just update tracking)."""
        if layer_idx not in self.layers_on_gpu:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        cpu_state = self.cpu_state_dicts.get(layer_idx, {})
        if not cpu_state:
            self.layers_on_gpu.discard(layer_idx)
            return

        # With preallocated buffers, don't reassign
        if not self.use_preallocated_buffers:
            block = self.blocks[layer_idx]
            for name, param in block.named_parameters():
                if name in cpu_state:
                    param.data = cpu_state[name]
            for name, buf in block.named_buffers():
                if name in cpu_state:
                    buf.data = cpu_state[name]

        self.layers_on_gpu.discard(layer_idx)

    def fetch_layer(
        self,
        layer_idx: int,
        row_indices: torch.Tensor = None,
    ):
        """Fetch layer synchronously."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return

        current_stream = torch.cuda.current_stream()

        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_stream(current_stream)
            if row_indices is not None:
                self._move_rows_to_gpu(
                    layer_idx,
                    row_indices,
                    non_blocking=True,
                )
            else:
                self._move_to_gpu(layer_idx, non_blocking=True)

        current_stream.wait_stream(self.transfer_stream)

    def prefetch_layer(
        self,
        layer_idx: int,
        row_indices: torch.Tensor = None,
    ):
        """Prefetch layer asynchronously."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx in self.layers_on_gpu:
            return

        current_stream = torch.cuda.current_stream()

        with torch.cuda.stream(self.transfer_stream):
            self.transfer_stream.wait_stream(current_stream)

            if row_indices is not None:
                self._move_rows_to_gpu(
                    layer_idx,
                    row_indices,
                    non_blocking=True,
                )
            else:
                self._move_to_gpu(layer_idx, non_blocking=True)

            event = self.event_pool[layer_idx]
            event.record(self.transfer_stream)
            self.transfer_events[layer_idx] = event

    def wait_for_layer(self, layer_idx: int):
        """Wait for layer transfer to complete."""
        if layer_idx in self.transfer_events:
            torch.cuda.current_stream().wait_event(
                self.transfer_events[layer_idx]
            )
            self.layers_on_gpu.add(layer_idx)
            del self.transfer_events[layer_idx]

    def offload_layer(self, layer_idx: int):
        """Offload layer back to CPU."""
        self._restore_layer_to_cpu(layer_idx)

    def set_row_indices_callback(
        self,
        callback: Callable[[int], Optional[torch.Tensor]],
    ):
        """Set callback for sparse row prefetch."""
        self.row_indices_callback = callback

    def is_inline_mode(self) -> bool:
        """Check if in no-prefetch (inline) mode."""
        return self.prefetch_mode == PrefetchMode.NONE

    @contextmanager
    def layer_context(
        self,
        layer_idx: int,
        phase: EnginePhase,
        prefetch_expert_ids=None,
        next_prefetch_expert_ids=None,
    ):
        """
        Context manager for layer execution with overlapped prefetching.

        Usage:
            with manager.layer_context(i, phase):
                output = layer(input)
        """
        if self.prefetch_mode == PrefetchMode.NONE:
            yield
            return

        if layer_idx == 0:
            self.transfer_stream.wait_stream(torch.cuda.current_stream())

        # Wait for current layer
        self.wait_for_layer(layer_idx)

        if layer_idx not in self.layers_on_gpu:
            row_indices = None
            if self.row_indices_callback and phase != EnginePhase.PREFILL:
                if prefetch_expert_ids is not None:
                    row_indices = prefetch_expert_ids.squeeze(0)
                else:
                    row_indices = self.row_indices_callback(layer_idx)
            self.fetch_layer(layer_idx, row_indices=row_indices)

        # Start prefetching next layers
        for offset in range(1, self.num_prefetch_layers + 1):
            next_idx = layer_idx + offset
            if next_idx < self.num_layers:
                row_indices = None
                if self.row_indices_callback and phase != EnginePhase.PREFILL:
                    if next_prefetch_expert_ids is not None:
                        row_indices = next_prefetch_expert_ids.squeeze(0)
                    else:
                        row_indices = self.row_indices_callback(next_idx)
                self.prefetch_layer(next_idx, row_indices=row_indices)

        yield

        # Offload previous layer
        prev_idx = layer_idx - 1
        if prev_idx >= 0 and prev_idx in self.layers_on_gpu:
            self.offload_layer(prev_idx)

    def cleanup(self):
        """Clean up resources."""
        for idx in list(self.layers_on_gpu):
            self.offload_layer(idx)

        if self.gpu_buffer_manager:
            self.gpu_buffer_manager.cleanup()
            self.gpu_buffer_manager = None

        torch.cuda.empty_cache()

    def reset(self):
        """Reset the manager."""
        assert len(self.transfer_events) == 0
        self.layers_on_gpu = set()
