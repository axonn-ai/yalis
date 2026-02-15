"""CUDA Graph Manager for explicit graph capture and replay.

Manages CUDA graph lifecycle for decode phase of LLM inference. Captures graphs
for user-specified batch sizes and routes actual batch sizes to the nearest
suitable graph.
"""

from __future__ import annotations

import os
import torch
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from math import log2


@dataclass
class CUDAGraphEntry:
    """Stores a captured CUDA graph with static buffers."""

    graph: torch.cuda.CUDAGraph
    batch_size: int
    # Static input buffers
    static_tokens: torch.Tensor
    static_block_table: Optional[torch.Tensor]
    static_token_counter: torch.Tensor
    # Static output buffers
    static_output_token: torch.Tensor
    static_output_logits: Optional[torch.Tensor]
    replay_count: int = 0


class CUDAGraphManager:
    """Manages CUDA graph capture and dispatch for decode inference.

    Captures CUDA graphs for specified batch sizes during initialization,
    then routes actual batch sizes to the nearest suitable graph during
    inference.

    Args:
        max_batch_size: Maximum batch size supported
        device: CUDA device
        cuda_graph_capture_sizes: Batch sizes to pre-capture graphs for.
            If None, defaults to powers of 2 up to max_batch_size.
            If empty list, lazy mode (capture on demand).
    """

    def __init__(
        self,
        max_batch_size: int,
        device: torch.device,
        cuda_graph_capture_sizes: Optional[List[int]] = None,
    ):
        self.max_batch_size = max_batch_size
        self.device = device
        self.graph_pool: Dict[int, CUDAGraphEntry] = {}
        self.enabled = (
            os.environ.get("YALIS_DISABLE_DECODE_CUDAGRAPHS", "0") != "1"
        )

        # Determine which batch sizes to capture
        if cuda_graph_capture_sizes is None:
            # Default: powers of 2 up to max_batch_size
            self.capture_sizes = [
                2**i
                for i in range(int(log2(max_batch_size)) + 1)
                if 2**i <= max_batch_size
            ]
        elif len(cuda_graph_capture_sizes) == 0:
            # Empty list = lazy mode (no pre-capture)
            self.capture_sizes = []
        else:
            # User-specified list (sorted, filtered to valid batch sizes)
            self.capture_sizes = sorted(
                [bs for bs in cuda_graph_capture_sizes if bs <= max_batch_size]
            )

    def capture_for_decode(
        self,
        batch_size: int,
        model: torch.nn.Module,
        generate_fn,
        max_num_blocks_per_seq: int,
        temperature: float,
        top_k: Optional[int],
        top_p: float,
        get_logits: bool = False,
        use_paged_kv_caching: bool = True,
    ) -> None:
        """Capture CUDA graph for decode phase.

        Args:
            batch_size: Batch size to capture for
            model: The model
            generate_fn: The generate function (torch.compiled)
            max_num_blocks_per_seq: Max blocks per sequence for block table
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p sampling
            get_logits: Whether to capture logits output
            use_paged_kv_caching: Whether paged KV caching is used
        """
        if batch_size in self.graph_pool:
            return  # Already captured

        # Allocate static input buffers
        static_tokens = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=self.device
        )
        # Only allocate block table if using paged KV caching
        if use_paged_kv_caching:
            static_block_table = torch.zeros(
                (batch_size, max_num_blocks_per_seq),
                dtype=torch.int32,
                device=self.device,
            )
        else:
            static_block_table = None
        static_token_counter = torch.zeros(
            batch_size, dtype=torch.int32, device=self.device
        )

        # Warmup iterations to stabilize memory and trigger torch.compile
        for _ in range(3):
            generate_fn(
                model,
                static_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                get_logits=get_logits,
                block_table=static_block_table,
                token_counter=static_token_counter,
            )

        # Capture graph
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output_token, static_output_logits = generate_fn(
                    model,
                    static_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    get_logits=get_logits,
                    block_table=static_block_table,
                    token_counter=static_token_counter,
                )
        torch.cuda.current_stream().wait_stream(stream)

        # Store in pool
        self.graph_pool[batch_size] = CUDAGraphEntry(
            graph=graph,
            batch_size=batch_size,
            static_tokens=static_tokens,
            static_block_table=static_block_table,
            static_token_counter=static_token_counter,
            static_output_token=static_output_token,
            static_output_logits=static_output_logits,
        )

    def find_suitable_batch_size(self, batch_size: int) -> Optional[int]:
        """Find the smallest captured batch size >= actual batch size."""
        suitable = [bs for bs in self.graph_pool.keys() if bs >= batch_size]
        return min(suitable) if suitable else None

    def run_decode(
        self,
        batch_size: int,
        tokens: torch.Tensor,
        block_table: Optional[torch.Tensor],
        token_counter: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run decode step using CUDA graph replay.

        Args:
            batch_size: Actual batch size
            tokens: Input tokens [batch_size, 1]
            block_table: Block table for KV cache (None if not using paged)
            token_counter: Token counters per sequence

        Returns:
            Tuple of (output_token, output_logits)
        """
        graph_bs = self.find_suitable_batch_size(batch_size)
        if graph_bs is None:
            raise RuntimeError(
                f"No suitable CUDA graph for batch_size={batch_size}. "
                f"Available: {list(self.graph_pool.keys())}"
            )

        entry = self.graph_pool[graph_bs]

        # Copy inputs to static buffers (only copy actual batch_size rows)
        entry.static_tokens[:batch_size].copy_(tokens)
        if block_table is not None and entry.static_block_table is not None:
            entry.static_block_table[:batch_size].copy_(block_table)
        entry.static_token_counter[:batch_size].copy_(token_counter)

        # Replay graph
        entry.graph.replay()
        entry.replay_count += 1

        # Extract outputs (only actual batch_size rows)
        output_token = entry.static_output_token[:batch_size].clone()
        output_logits = None
        if entry.static_output_logits is not None:
            output_logits = entry.static_output_logits[:batch_size].clone()

        return output_token, output_logits
