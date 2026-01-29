"""DoubleSparse backend wrapper (kernel lives in attention/double_sparse)."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional, List

import torch

from .registry import register_attention

_KERNEL_PKG = "yalis.attention._double_sparse_kernels"
_KERNEL_DIR = Path(__file__).resolve().parent / "double_sparse"


def _load_kernel_module(name: str):
    if _KERNEL_PKG not in sys.modules:
        spec = importlib.util.spec_from_loader(_KERNEL_PKG, loader=None, is_package=True)
        pkg = importlib.util.module_from_spec(spec)
        pkg.__path__ = [str(_KERNEL_DIR)]
        sys.modules[_KERNEL_PKG] = pkg
    return importlib.import_module(f"{_KERNEL_PKG}.{name}")


_channel = _load_kernel_module("channel")
_sparse = _load_kernel_module("sparse")

get_label_tensor = _channel.get_label_tensor
fwd_sparse_no_mask = _sparse.fwd_sparse_no_mask


def permute_channel_config(sorted_channel: torch.Tensor) -> torch.Tensor:
    head_dim = sorted_channel.shape[1]
    return (sorted_channel * 2) % head_dim + (sorted_channel * 2) // head_dim


def resolve_double_sparse_config(
    head_dim: int,
    max_seq_length: int,
    sparsity: Optional[int],
    heavy_channel_num: Optional[int],
    heavy_const: Optional[int],
) -> tuple[int, int]:
    if sparsity is None or sparsity <= 0:
        raise ValueError("double_sparse_sparsity must be a positive integer.")
    if heavy_channel_num is None:
        heavy_channel_num = max(1, head_dim // sparsity)
    if heavy_const is None:
        heavy_const = max(1, max_seq_length // sparsity)
    if heavy_channel_num <= 0 or heavy_const <= 0:
        raise ValueError("double_sparse heavy parameters must be positive.")
    if heavy_channel_num > head_dim:
        raise ValueError("double_sparse_heavy_channel_num exceeds head_dim.")
    if heavy_const > max_seq_length:
        raise ValueError("double_sparse_heavy_const exceeds max_seq_length.")
    return heavy_channel_num, heavy_const


def load_channel_config(
    channel_config_path: str,
    n_layers: int,
    channel_type: str,
    heavy_channel_num: int,
    device: torch.device,
) -> List[torch.Tensor]:
    path = Path(channel_config_path)
    if not path.is_file():
        raise FileNotFoundError(f"DoubleSparse channel config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        channel_config = json.load(handle)
    sorted_channels: List[torch.Tensor] = []
    for layer_idx in range(n_layers):
        key = f"model.layers.{layer_idx}.self_attn.{channel_type}_proj"
        if key not in channel_config:
            raise KeyError(f"Missing channel config entry: {key}")
        channel = torch.tensor(channel_config[key], dtype=torch.int64, device=device)
        channel = permute_channel_config(channel)[:, :heavy_channel_num].contiguous()
        sorted_channels.append(channel)
    return sorted_channels


def init_double_sparse_state(
    batch_size: int,
    max_seq_length: int,
    heads: int,
    head_dim: int,
    heavy_const: int,
    heavy_channel_num: int,
    sorted_channel: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    if sorted_channel.shape[0] != heads:
        raise ValueError(
            "DoubleSparse channel config head count mismatch: "
            f"{sorted_channel.shape[0]} != {heads}."
        )
    state = {
        "sorted_channel": sorted_channel.to(device=device, dtype=torch.int64).contiguous(),
        "k_label": torch.zeros(
            (batch_size, max_seq_length, heads, heavy_channel_num),
            device=device,
            dtype=dtype,
        ),
        "attn_out": torch.zeros((batch_size, heads, head_dim), device=device, dtype=dtype),
        "heavy_const": heavy_const,
        "heavy_channel_num": heavy_channel_num,
    }
    return state


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    if cos is not None and sin is not None:
        if cos.dim() > 1:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        roped = []
        for x in (q, k):
            head_size = x.size(-1)
            x1 = x[..., : head_size // 2]
            x2 = x[..., head_size // 2 :]
            rotated = torch.cat((-x2, x1), dim=-1)
            roped.append((x * cos + rotated * sin).to(dtype=x.dtype))
        q, k = roped
    return q, k


def _index_into_rope_cache(cache: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    assert index.dim() == 1, "this method is only for the generation phase"
    return torch.index_select(cache, 0, index.view(-1)).reshape(index.size(0), 1, -1)


def _build_mask_from_index(index: torch.Tensor, t_max: int) -> torch.Tensor:
    arange_t = torch.arange(t_max, device=index.device).unsqueeze(0)
    return arange_t <= index.unsqueeze(1)


def _double_sparse_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    state: dict,
) -> torch.Tensor:
    B, q_heads, T, D = q.shape
    kv_heads = k.shape[1]
    q_per_kv = q_heads // kv_heads
    if rotary_cos is not None and rotary_sin is not None:
        cos = rotary_cos[:T][None, None, :, :]
        sin = rotary_sin[:T][None, None, :, :]
        q, k = _apply_rope(q, k, cos, sin)

    k_cache[:, :, :T, :] = k
    v_cache[:, :, :T, :] = v

    heavy_channel_num = state["heavy_channel_num"]
    sorted_channel = state["sorted_channel"]
    if q_per_kv > 1:
        k_for_labels = k.repeat_interleave(q_per_kv, dim=1)
    else:
        k_for_labels = k
    k_tokens = k_for_labels.transpose(1, 2).contiguous().view(B * T, q_heads, D)
    labels = torch.empty((B * T, q_heads, heavy_channel_num), device=k.device, dtype=k.dtype)
    get_label_tensor(k_tokens, sorted_channel, labels, heavy_channel_num)
    state["k_label"][:, :T, :, :] = labels.view(B, T, q_heads, heavy_channel_num)

    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=(q_heads != kv_heads)
    )


def _double_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    token_counter: torch.Tensor,
    rotary_cos: Optional[torch.Tensor],
    rotary_sin: Optional[torch.Tensor],
    state: dict,
    retain_perc: Optional[torch.Tensor],
) -> torch.Tensor:
    if rotary_cos is not None and rotary_sin is not None:
        cos = _index_into_rope_cache(rotary_cos, token_counter)
        sin = _index_into_rope_cache(rotary_sin, token_counter)
        q, k = _apply_rope(q, k, cos, sin)

    B, q_heads, _, D = q.shape
    kv_heads = k.shape[1]
    q_per_kv = q_heads // kv_heads
    b_indices = torch.arange(B, device=k_cache.device)
    t_indices = token_counter.view(-1)

    k_cache[b_indices, :, t_indices, :] = k[:, :, 0, :]
    v_cache[b_indices, :, t_indices, :] = v[:, :, 0, :]

    heavy_channel_num = state["heavy_channel_num"]
    sorted_channel = state["sorted_channel"]
    k_labels = state["k_label"]

    q_flat = q[:, :, 0, :].contiguous()
    q_label = torch.empty((B, q_heads, heavy_channel_num), device=q.device, dtype=q.dtype)
    get_label_tensor(q_flat, sorted_channel, q_label, heavy_channel_num)

    if q_per_kv > 1:
        k_for_labels = k.repeat_interleave(q_per_kv, dim=1)
    else:
        k_for_labels = k
    k_flat = k_for_labels[:, :, 0, :].contiguous()
    k_label = torch.empty((B, q_heads, heavy_channel_num), device=k.device, dtype=k.dtype)
    get_label_tensor(k_flat, sorted_channel, k_label, heavy_channel_num)
    k_labels[b_indices, t_indices, :, :] = k_label

    label_scores = torch.matmul(
        q_label.view(B, 1, q_heads, heavy_channel_num).transpose(1, 2),
        k_labels.view(B, -1, q_heads, heavy_channel_num).transpose(1, 2).transpose(2, 3),
    ).view(B, q_heads, 1, -1)

    mask = _build_mask_from_index(token_counter, t_max=k_labels.size(1))
    label_scores = label_scores.masked_fill(~mask[:, None, None, :], float("-inf"))
    _, label_index = torch.topk(label_scores, state["heavy_const"], dim=-1)
    heavy_list = label_index.view(B, q_heads, state["heavy_const"])

    k_full = k_cache.transpose(1, 2)
    v_full = v_cache.transpose(1, 2)
    if q_per_kv > 1:
        k_full = k_full.repeat_interleave(q_per_kv, dim=2)
        v_full = v_full.repeat_interleave(q_per_kv, dim=2)
    k_full = k_full.contiguous().view(-1, q_heads, D)
    v_full = v_full.contiguous().view(-1, q_heads, D)
    fwd_sparse_no_mask(q_flat, k_full, v_full, state["attn_out"], heavy_list)

    if retain_perc is not None:
        denom = token_counter.to(torch.float32) + 1.0
        retain = (state["heavy_const"] / denom).view(-1, 1) * 100.0
        retain_perc.add_(retain)

    return state["attn_out"].unsqueeze(-2)


def double_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    prestore_kv_cache: bool = True,
    double_sparse_state: Optional[dict] = None,
    retain_perc: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if "block_table" in kwargs and kwargs["block_table"] is not None:
        raise ValueError("'block_table' or paged kv-caching is not compatible with DoubleSparse.")
    if k_cache is None or v_cache is None:
        raise ValueError("double_sparse requires k_cache and v_cache")
    if double_sparse_state is None:
        raise ValueError("double_sparse requires an initialized state dict")
    if q.size(-1) not in {16, 32, 64, 128}:
        raise ValueError("double_sparse only supports head_dim in {16, 32, 64, 128}.")

    T = q.shape[-2]
    if T == 1:
        if cache_seqlens is None:
            raise ValueError("double_sparse requires cache_seqlens for decode.")
        return _double_sparse_decode(
            q=q,
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            token_counter=cache_seqlens,
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            state=double_sparse_state,
            retain_perc=retain_perc,
        )

    return _double_sparse_prefill(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        state=double_sparse_state,
    )


@register_attention("double_sparse")
def double_sparse_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: Optional[torch.Tensor] = None,
    v_cache: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    use_intra_head_parallelism: bool = False,
    prestore_kv_cache: bool = True,
    **kwargs,
):
    if use_intra_head_parallelism:
        raise ValueError("double_sparse does not support intra head parallelism.")
    retain_perc = kwargs.pop("retain_perc", None)
    double_sparse_state = kwargs.pop("double_sparse_state", None)
    return double_sparse_attention(
        q=q,
        k=k,
        v=v,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        rotary_cos=rotary_cos,
        rotary_sin=rotary_sin,
        prestore_kv_cache=prestore_kv_cache,
        double_sparse_state=double_sparse_state,
        retain_perc=retain_perc,
        **kwargs,
    )
