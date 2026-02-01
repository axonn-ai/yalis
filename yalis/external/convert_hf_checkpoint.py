# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from tqdm import tqdm
import torch
from lightning.fabric.utilities.load import (
    _NotYetLoadedTensor as NotYetLoadedTensor,
)
from safetensors.torch import load_file as load_safetensors
from config import Config
from litgpt.utils import (
    extend_checkpoint_dir,
    lazy_load,
    save_config,
    # incremental_save,
)
from safetensor_saver import incremental_save


# =============================================================================
# Memory Profiling Utilities
# =============================================================================
def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def count_dict_tensors(d: dict, depth: int = 0) -> Tuple[int, int, int]:
    """
    Count tensors in a nested dict structure.
    Returns: (num_tensors, num_loaded_tensors, estimated_bytes_loaded)
    """
    num_tensors = 0
    num_loaded = 0
    bytes_loaded = 0

    for v in d.values():
        if isinstance(v, dict):
            nt, nl, bl = count_dict_tensors(v, depth + 1)
            num_tensors += nt
            num_loaded += nl
            bytes_loaded += bl
        elif isinstance(v, list):
            for item in v:
                if item is None:
                    continue
                if isinstance(item, torch.Tensor):
                    num_tensors += 1
                    num_loaded += 1
                    bytes_loaded += item.element_size() * item.numel()
                elif hasattr(item, "_load_tensor"):
                    num_tensors += 1
                elif isinstance(item, dict):
                    nt, nl, bl = count_dict_tensors(item, depth + 1)
                    num_tensors += nt
                    num_loaded += nl
                    bytes_loaded += bl
        elif isinstance(v, torch.Tensor):
            num_tensors += 1
            num_loaded += 1
            bytes_loaded += v.element_size() * v.numel()
        elif hasattr(v, "_load_tensor"):
            # NotYetLoadedTensor - not actually in memory
            num_tensors += 1

    return num_tensors, num_loaded, bytes_loaded


class ConversionProfiler:
    """Profiler to track memory and timing during checkpoint conversion."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.snapshots = []
        self.start_time = time.time()

    def snapshot(
        self,
        label: str,
        qkv_weights: dict = None,
        gate_up_proj_weights: dict = None,
        down_proj_weights: dict = None,
        state_dict: dict = None,
    ):
        if not self.enabled:
            return

        mem_mb = get_memory_usage_mb()
        elapsed = time.time() - self.start_time

        snap = {
            "label": label,
            "time_s": elapsed,
            "memory_mb": mem_mb,
        }

        if qkv_weights is not None:
            nt, nl, bl = count_dict_tensors(qkv_weights)
            snap["qkv_weights"] = {
                "num_layers": len(qkv_weights),
                "num_tensors": nt,
                "num_loaded": nl,
                "loaded_mb": bl / (1024 * 1024),
            }

        if gate_up_proj_weights is not None:
            nt, nl, bl = count_dict_tensors(gate_up_proj_weights)
            snap["gate_up_proj_weights"] = {
                "num_layers": len(gate_up_proj_weights),
                "num_tensors": nt,
                "num_loaded": nl,
                "loaded_mb": bl / (1024 * 1024),
            }

        if down_proj_weights is not None:
            nt, nl, bl = count_dict_tensors(down_proj_weights)
            snap["down_proj_weights"] = {
                "num_layers": len(down_proj_weights),
                "num_tensors": nt,
                "num_loaded": nl,
                "loaded_mb": bl / (1024 * 1024),
            }

        if state_dict is not None:
            num_saved = len(state_dict)
            snap["state_dict_entries"] = num_saved

        self.snapshots.append(snap)

    def print_summary(self):
        if not self.enabled or not self.snapshots:
            return

        print("\n" + "=" * 70)
        print("CONVERSION PROFILING SUMMARY")
        print("=" * 70)

        for snap in self.snapshots:
            print(f"\n[{snap['time_s']:.1f}s] {snap['label']}")
            print(f"  Process memory: {snap['memory_mb']:.1f} MB")

            for key in [
                "qkv_weights",
                "gate_up_proj_weights",
                "down_proj_weights",
            ]:
                if key in snap:
                    d = snap[key]
                    print(
                        f"  {key}: {d['num_layers']} layers, "
                        f"{d['num_tensors']} tensors "
                        f"({d['num_loaded']} loaded, {d['loaded_mb']:.1f} MB)"
                    )

            if "state_dict_entries" in snap:
                print(f"  state_dict entries: {snap['state_dict_entries']}")

        print("=" * 70 + "\n")


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "gpt_neox.embed_in.weight": "transformer.wte.weight",
        "gpt_neox.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "gpt_neox.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "gpt_neox.layers.{}.attention.query_key_value.bias": "transformer.h.{}.attn.attn.bias",
        "gpt_neox.layers.{}.attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "gpt_neox.layers.{}.attention.dense.bias": "transformer.h.{}.attn.proj.bias",
        "gpt_neox.layers.{}.attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "gpt_neox.layers.{}.attention.rotary_emb.inv_freq": None,
        "gpt_neox.layers.{}.attention.bias": None,
        "gpt_neox.layers.{}.attention.masked_bias": None,
        "gpt_neox.layers.{}.post_attention_layernorm.bias": "transformer.h.{}.norm_2.bias",
        "gpt_neox.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.bias": "transformer.h.{}.mlp.fc.bias",
        "gpt_neox.layers.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.bias": "transformer.h.{}.mlp.proj.bias",
        "gpt_neox.layers.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "gpt_neox.final_layer_norm.bias": "transformer.ln_f.bias",
        "gpt_neox.final_layer_norm.weight": "transformer.ln_f.weight",
        "embed_out.weight": "lm_head.weight",
    }

    for name, param in hf_weights.items():
        if "gpt_neox.layers" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param


def copy_weights_falcon(
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "transformer.word_embeddings.weight": "transformer.wte.weight",
        "transformer.h.{}.self_attention.query_key_value.weight": "transformer.h.{}.attn.attn.weight",
        "transformer.h.{}.self_attention.dense.weight": "transformer.h.{}.attn.proj.weight",
        "transformer.h.{}.mlp.dense_h_to_4h.weight": "transformer.h.{}.mlp.fc.weight",
        "transformer.h.{}.mlp.dense_4h_to_h.weight": "transformer.h.{}.mlp.proj.weight",
        "transformer.ln_f.bias": "transformer.ln_f.bias",
        "transformer.ln_f.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    # the original model definition is different for each size
    if "7b" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
            }
        )
    elif "40b" in model_name or "180B" in model_name:
        weight_map.update(
            {
                "transformer.h.{}.ln_attn.bias": "transformer.h.{}.norm_1.bias",
                "transformer.h.{}.ln_attn.weight": "transformer.h.{}.norm_1.weight",
                "transformer.h.{}.ln_mlp.bias": "transformer.h.{}.norm_2.bias",
                "transformer.h.{}.ln_mlp.weight": "transformer.h.{}.norm_2.weight",
            }
        )
    else:
        raise NotImplementedError

    for name, param in hf_weights.items():
        if "transformer.h" in name:
            from_name, number = layer_template(name, 2)
            to_name = weight_map[from_name].format(number)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param


def copy_weights_qwen_3(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[Any, List[Optional[NotYetLoadedTensor]]],
    down_proj_weights: Dict[Any, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.q_norm.weight": "transformer.h.{l}.attn.norm_q.weight",
        "model.layers.{}.self_attn.k_norm.weight": "transformer.h.{l}.attn.norm_k.weight",
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.mlp.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.mlp.experts.{}.gate_proj.weight": None,
                "model.layers.{}.mlp.experts.{}.up_proj.weight": None,
                "model.layers.{}.mlp.experts.{}.down_proj.weight": None,
                # "transformer.h.{l}.mlp.experts.{e}.proj.weight",
            }
        )
    elif config.mlp_class_name == "LLaMAMLP":
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": None,
                "model.layers.{}.mlp.up_proj.weight": None,
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    transformer_wte_weight = None

    for from_name, param in hf_weights.items():

        if "model.layers" in from_name:
            name_template, l = layer_template(from_name, 2)
            if "experts" in from_name:
                name_template, e = layer_template(name_template, 5)
            else:
                e = None

            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                qkv = qkv_weights.setdefault(l, defaultdict(dict))
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param

            if any(w in from_name for w in ("gate_proj", "up_proj")):
                if e is not None:
                    # MoE case
                    gate_up_proj = gate_up_proj_weights.setdefault(
                        l, defaultdict(lambda: defaultdict(dict))
                    )
                    weight_name, weight_type = from_name.split(".")[-2:]
                    gate_up_proj[weight_type][weight_name][e] = param
                else:
                    # Non-MoE case
                    gate_up_proj = gate_up_proj_weights.setdefault(
                        l, defaultdict(dict)
                    )
                    weight_name, weight_type = from_name.split(".")[-2:]
                    gate_up_proj[weight_type][weight_name] = param

            if any(w in from_name for w in ("down_proj",)) and e is not None:
                # Down projections need to be combined for all experts in MoE case
                down_proj = down_proj_weights.setdefault(
                    l, defaultdict(lambda: defaultdict(dict))
                )
                weight_name, weight_type = from_name.split(".")[-2:]
                down_proj[weight_type][weight_name][e] = param

            to_name = weight_map[name_template]
            if to_name is None:
                continue
            if e is None:
                to_name = to_name.format(l=l)
            else:
                to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[from_name]
        param = load_param(param, from_name, dtype, verbose=debug_mode)

        if to_name == "transformer.wte.weight":
            transformer_wte_weight = param.clone().detach()

        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight.clone()
            )
            state_dict["lm_head.weight"] = param_saved

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is split across different .bin files
                continue
            q = load_param(
                qkv["q_proj"],
                f"layer {i} q {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            k = load_param(
                qkv["k_proj"],
                f"layer {i} k {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            v = load_param(
                qkv["v_proj"],
                f"layer {i} v {weight_type}",
                dtype,
                verbose=debug_mode,
            )

            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)

            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

    for i in list(gate_up_proj_weights):
        for weight_type in list(gate_up_proj_weights[i]):
            gate_up_proj = gate_up_proj_weights[i][weight_type]

            if ("gate_proj" not in gate_up_proj) or (
                "up_proj" not in gate_up_proj
            ):
                continue
            gate_proj = gate_up_proj["gate_proj"]
            up_proj = gate_up_proj["up_proj"]

            if isinstance(gate_proj, dict) and isinstance(up_proj, dict):
                # MoE case
                # Check if all experts are present
                num_gate_proj_experts = len(gate_proj)
                num_up_proj_experts = len(up_proj)
                if (
                    num_gate_proj_experts != config.n_expert
                    or num_up_proj_experts != config.n_expert
                ):
                    continue

                # Pre-allocate after loading first expert to get shape
                first_gate = load_param(
                    gate_proj[0],
                    f"layer {i} gate_proj 0",
                    dtype,
                    verbose=debug_mode,
                )
                first_up = load_param(
                    up_proj[0],
                    f"layer {i} up_proj 0",
                    dtype,
                    verbose=debug_mode,
                )
                intermediate_size, hidden_size = first_gate.shape

                # Pre-allocate combined tensor
                gate_up_proj_combined = torch.empty(
                    (config.n_expert, 2 * intermediate_size, hidden_size),
                    dtype=first_gate.dtype,
                )

                # Fill first expert
                gate_up_proj_combined[0] = torch.stack(
                    (first_gate, first_up), dim=1
                ).reshape(2 * intermediate_size, -1)
                del first_gate, first_up

                # Fill remaining experts directly
                for e in range(1, config.n_expert):
                    gate_proj_e = load_param(
                        gate_proj[e],
                        f"layer {i} gate_proj {e}",
                        dtype,
                        verbose=debug_mode,
                    )
                    up_proj_e = load_param(
                        up_proj[e],
                        f"layer {i} up_proj {e}",
                        dtype,
                        verbose=debug_mode,
                    )
                    gate_up_proj_combined[e] = torch.stack(
                        (gate_proj_e, up_proj_e), dim=1
                    ).reshape(2 * intermediate_size, -1)
                    del gate_proj_e, up_proj_e

                # Use incremental save if available
                if saver is not None:
                    gate_up_proj_ref = saver.store_early(
                        f"transformer.h.{i}.mlp.experts.gate_up_proj",
                        gate_up_proj_combined,
                    )
                    state_dict[
                        f"transformer.h.{i}.mlp.experts.gate_up_proj"
                    ] = gate_up_proj_ref
                else:
                    state_dict[
                        f"transformer.h.{i}.mlp.experts.gate_up_proj"
                    ] = gate_up_proj_combined
                del gate_up_proj_combined
                del gate_up_proj_weights[i][weight_type]
            else:
                # Non-MoE case
                gate_proj = load_param(
                    gate_proj,
                    f"layer {i} gate_proj",
                    dtype,
                    verbose=debug_mode,
                )
                up_proj = load_param(
                    up_proj, f"layer {i} up_proj", dtype, verbose=debug_mode
                )
                gate_up_proj = torch.stack(
                    (gate_proj, up_proj), dim=1
                ).reshape(2 * gate_proj.size(0), -1)
                state_name = f"transformer.h.{i}.mlp.gate_up_proj.weight"
                state_dict[state_name] = gate_up_proj
                del gate_up_proj_weights[i][weight_type]

    for i in list(down_proj_weights):
        for weight_type in list(down_proj_weights[i]):
            down_proj = down_proj_weights[i][weight_type]["down_proj"]
            if len(down_proj) != config.n_expert:
                continue

            # Pre-allocate after loading first expert to get shape
            first_down = load_param(
                down_proj[0],
                f"layer {i} down_proj 0",
                dtype,
                verbose=debug_mode,
            )
            hidden_size, intermediate_size = first_down.shape

            # Pre-allocate combined tensor
            down_proj_combined = torch.empty(
                (config.n_expert, hidden_size, intermediate_size),
                dtype=first_down.dtype,
            )
            down_proj_combined[0] = first_down
            del first_down

            # Fill remaining experts directly
            for e in range(1, config.n_expert):
                down_proj_e = load_param(
                    down_proj[e],
                    f"layer {i} down_proj {e}",
                    dtype,
                    verbose=debug_mode,
                )
                down_proj_combined[e] = down_proj_e
                del down_proj_e

            # Use incremental save if available
            if saver is not None:
                down_proj_ref = saver.store_early(
                    f"transformer.h.{i}.mlp.experts.proj", down_proj_combined
                )
                state_dict[f"transformer.h.{i}.mlp.experts.proj"] = (
                    down_proj_ref
                )
            else:
                state_dict[f"transformer.h.{i}.mlp.experts.proj"] = (
                    down_proj_combined
                )
            del down_proj_combined
            del down_proj_weights[i]


def copy_weights_qwen_2_5(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.mlp.gate_proj.weight": None,
        "model.layers.{}.mlp.up_proj.weight": None,
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    transformer_wte_weight = None

    for from_name, param in hf_weights.items():
        if "model.layers" in from_name:
            name_template, *ids = layer_template(from_name, 2)
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                qkv = qkv_weights.setdefault(ids[0], defaultdict(dict))
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param

            if any(w in from_name for w in ("gate_proj", "up_proj")):
                gate_up_proj = gate_up_proj_weights.setdefault(
                    ids[0], defaultdict(dict)
                )
                weight_name, weight_type = from_name.split(".")[-2:]
                gate_up_proj[weight_type][weight_name] = param

            to_name = weight_map[name_template]
            if to_name is None:
                continue
            to_name = to_name.format(l=ids[0])

        else:
            to_name = weight_map[from_name]
        param = load_param(param, from_name, dtype, verbose=debug_mode)

        if to_name == "transformer.wte.weight":
            transformer_wte_weight = param.clone().detach()

        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight.clone()
            )
            state_dict["lm_head.weight"] = param_saved

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # qkv is split across different .bin files
                continue
            q = load_param(
                qkv["q_proj"],
                f"layer {i} q {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            k = load_param(
                qkv["k_proj"],
                f"layer {i} k {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            v = load_param(
                qkv["v_proj"],
                f"layer {i} v {weight_type}",
                dtype,
                verbose=debug_mode,
            )

            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

    for i in list(gate_up_proj_weights):
        for weight_type in list(gate_up_proj_weights[i]):
            gate_up_proj = gate_up_proj_weights[i][weight_type]
            if ("gate_proj" not in gate_up_proj) or (
                "up_proj" not in gate_up_proj
            ):
                continue
            gate_proj = gate_up_proj["gate_proj"]
            up_proj = gate_up_proj["up_proj"]

            gate_proj = load_param(
                gate_proj, f"layer {i} gate_proj", dtype, verbose=debug_mode
            )
            up_proj = load_param(
                up_proj, f"layer {i} up_proj", dtype, verbose=debug_mode
            )
            gate_up_proj = torch.stack((gate_proj, up_proj), dim=1).reshape(
                2 * gate_proj.size(0), -1
            )
            state_dict[f"transformer.h.{i}.mlp.gate_up_proj.weight"] = (
                gate_up_proj
            )
            del gate_up_proj_weights[i]


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    down_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{l}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{l}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{l}.attn.proj.weight",
        "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{l}.norm_2.weight",
        "model.layers.{}.post_attention_layernorm.bias": "transformer.h.{l}.norm_2.bias",
        "model.norm.weight": "transformer.ln_f.weight",
        "model.norm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
    }
    if config.mlp_class_name == "LLaMAMoE":
        weight_map.update(
            {
                "model.layers.{}.block_sparse_moe.gate.weight": "transformer.h.{l}.mlp.gate.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": None,  # "transformer.h.{l}.mlp.experts.{e}.fc_1.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": None,  # "transformer.h.{l}.mlp.experts.{e}.fc_2.weight",
                "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": None,  # "transformer.h.{l}.mlp.experts.{e}.proj.weight",
            }
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP"):
        weight_map.update(
            {
                "model.layers.{}.mlp.gate_proj.weight": None,  # "transformer.h.{l}.mlp.fc_1.weight",
                "model.layers.{}.mlp.up_proj.weight": None,  # "transformer.h.{l}.mlp.fc_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{l}.mlp.proj.weight",
            }
        )
    else:
        raise NotImplementedError

    transformer_wte_weight = None
    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l = layer_template(name, 2)
            e = None
            if "block_sparse_moe.experts" in name:
                from_name, e = layer_template(from_name, 5)
            qkv = qkv_weights.setdefault(l, [None, None, None])
            if e is not None and "experts" in name:
                # Two level dictionary for gate_up_proj and down_proj
                # First level is the layer index
                # Second level is the expert index
                gate_up_proj = gate_up_proj_weights.setdefault(
                    l, defaultdict(lambda: [None, None])
                )
                down_proj = down_proj_weights.setdefault(
                    l, defaultdict(lambda: [None])
                )
            elif "mlp" in name:
                gate_up_proj = gate_up_proj_weights.setdefault(l, [None, None])
            else:
                gate_up_proj = None

            if "q_proj" in name:
                qkv[0] = param
            elif "k_proj" in name:
                qkv[1] = param
            elif "v_proj" in name:
                qkv[2] = param
            elif "gate_proj" in name:
                gate_up_proj[0] = param
            elif "up_proj" in name:
                gate_up_proj[1] = param
            elif "experts" in name and "w1" in name:
                gate_up_proj[e][0] = param
            elif "experts" in name and "w3" in name:
                gate_up_proj[e][1] = param
            elif "experts" in name and "w2" in name:
                down_proj[e][0] = param
            # Here I can directly check if a layer is completed in which case I trigger the concatenation and the split for the reshaped tensor, store it to the disk and then delete the information fromt he qkv_weights dictionary which might be the one getting overloaded

            # incremental QKV reshaping
            # Here if None not in in qkv means that all qkv tensors for the particular layer being
            # loaded right now have been temporarily loaded onto the qkv loading dictionary
            # which means that we can write the converted tensrors for ll to the disk and free up the space from the RAM for future layers' tensors
            if None not in qkv and saver is not None:

                q, k, v = qkv
                q = load_param(q, f"layer {l} q", dtype, verbose=debug_mode)
                k = load_param(k, f"layer {l} k", dtype, verbose=debug_mode)
                v = load_param(v, f"layer {l} v", dtype, verbose=debug_mode)
                q_per_kv = config.n_head // config.n_query_groups
                qs = torch.split(q, config.head_size * q_per_kv)
                ks = torch.split(k, config.head_size)
                vs = torch.split(v, config.head_size)
                cycled = [t for group in zip(qs, ks, vs) for t in group]
                qkv = torch.cat(cycled)

                qkv_ref = saver.store_early(
                    f"transformer.h.{l}.attn.attn.weight", qkv
                )
                # store early returns a reference to the actual memory stored in the disk
                # freeing up space from the RAM
                state_dict[f"transformer.h.{l}.attn.attn.weight"] = qkv_ref

                qkv_weights[l] = None
                del qkv_weights[l]

            # Now doing proj reshaping with the same principle of incremental QKV reshaping
            # Similarly doing the same check for the gate projection layers
            if (
                gate_up_proj is not None
                and isinstance(gate_up_proj, list)
                and None not in gate_up_proj
                and saver is not None
            ):
                # Storing early is only done for non-MoE case right now
                # TODO(Ishaan): Add early save support for MoE case

                gate_proj, up_proj = gate_up_proj
                gate_proj = load_param(
                    gate_proj,
                    f"layer {l} {e} gate_proj",
                    dtype,
                    verbose=debug_mode,
                )
                up_proj = load_param(
                    up_proj,
                    f"layer {l} {e} up_proj",
                    dtype,
                    verbose=debug_mode,
                )

                gate_up_proj = torch.stack(
                    (gate_proj, up_proj), dim=1
                ).reshape(2 * gate_proj.size(0), -1)

                gate_up_proj_ref = saver.store_early(
                    f"transformer.h.{l}.mlp.gate_up_proj.weight", gate_up_proj
                )
                state_dict[f"transformer.h.{l}.mlp.gate_up_proj.weight"] = (
                    gate_up_proj_ref
                )

                gate_up_proj_weights[l] = None

                del gate_up_proj_weights[l]

            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l=l, e=e)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)

        if to_name == "transformer.wte.weight":
            transformer_wte_weight = param.clone().detach()

        if saver is not None:
            # For the tensors that have a to mapping, we use the same store early principle
            # we store the reference and then we delete the loaded tensor from the RAM
            param_saved = saver.store_early(to_name, param)
            del param
            state_dict[to_name] = param_saved

        else:
            state_dict[to_name] = param

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight
            )
            del transformer_wte_weight
            state_dict["lm_head.weight"] = param_saved

    # convert separate gate proj and up proj into one tensor
    for i, proj in list(gate_up_proj_weights.items()):
        if isinstance(proj, list):
            # Non-MoE case
            gate_proj = proj[0]
            up_proj = proj[1]
            if gate_proj is None or up_proj is None:
                continue
            gate_proj = load_param(
                gate_proj, f"layer {i} gate_proj", dtype, verbose=debug_mode
            )
            up_proj = load_param(
                up_proj, f"layer {i} up_proj", dtype, verbose=debug_mode
            )
            gate_up_proj = torch.stack((gate_proj, up_proj), dim=1).reshape(
                2 * gate_proj.size(0), -1
            )
            state_dict[f"transformer.h.{i}.mlp.gate_up_proj.weight"] = (
                gate_up_proj
            )
            del gate_up_proj_weights[i]
        elif isinstance(proj, dict):
            # MoE case
            if len(list(proj.items())) != config.n_expert:
                # Not all experts are present
                continue

            # Check all experts are present first
            all_present = True
            for e in range(config.n_expert):
                if e not in proj:
                    all_present = False
                    break
                proj_e = proj[e]
                if proj_e[0] is None or proj_e[1] is None:
                    all_present = False
                    break
            if not all_present:
                continue

            # Pre-allocate output tensor after loading first expert to get shape
            first_gate = load_param(
                proj[0][0], f"layer {i} 0 gate_proj", dtype, verbose=debug_mode
            )
            first_up = load_param(
                proj[0][1], f"layer {i} 0 up_proj", dtype, verbose=debug_mode
            )
            intermediate_size, hidden_size = first_gate.shape

            # Pre-allocate combined tensor: [n_expert, 2*intermediate, hidden]
            gate_up_proj_combined = torch.empty(
                (config.n_expert, 2 * intermediate_size, hidden_size),
                dtype=first_gate.dtype,
            )

            # Fill first expert
            gate_up_proj_combined[0] = torch.stack(
                (first_gate, first_up), dim=1
            ).reshape(2 * intermediate_size, -1)
            del first_gate, first_up

            # Fill remaining experts directly into pre-allocated tensor
            for e in range(1, config.n_expert):
                gate_proj_e = load_param(
                    proj[e][0],
                    f"layer {i} {e} gate_proj",
                    dtype,
                    verbose=debug_mode,
                )
                up_proj_e = load_param(
                    proj[e][1],
                    f"layer {i} {e} up_proj",
                    dtype,
                    verbose=debug_mode,
                )
                gate_up_proj_combined[e] = torch.stack(
                    (gate_proj_e, up_proj_e), dim=1
                ).reshape(2 * intermediate_size, -1)
                del gate_proj_e, up_proj_e

            # Use incremental save if available
            if saver is not None:
                gate_up_proj_ref = saver.store_early(
                    f"transformer.h.{i}.mlp.experts.gate_up_proj",
                    gate_up_proj_combined,
                )
                state_dict[f"transformer.h.{i}.mlp.experts.gate_up_proj"] = (
                    gate_up_proj_ref
                )
            else:
                state_dict[f"transformer.h.{i}.mlp.experts.gate_up_proj"] = (
                    gate_up_proj_combined
                )
            del gate_up_proj_combined
            del gate_up_proj_weights[i]

    for i, proj in list(down_proj_weights.items()):
        assert isinstance(
            proj, dict
        ), "Down projection weights should be a dictionary"
        if len(list(proj.items())) != config.n_expert:
            # Not all experts are present
            continue

        # Check all experts are present first
        all_present = True
        for e in range(config.n_expert):
            if e not in proj:
                all_present = False
                break
            if proj[e][0] is None:
                all_present = False
                break
        if not all_present:
            continue

        # Pre-allocate after loading first expert to get shape
        first_down = load_param(
            proj[0][0], f"layer {i} 0 down_proj", dtype, verbose=debug_mode
        )
        hidden_size, intermediate_size = first_down.shape

        # Pre-allocate combined tensor: [n_expert, hidden, intermediate]
        down_proj_combined = torch.empty(
            (config.n_expert, hidden_size, intermediate_size),
            dtype=first_down.dtype,
        )
        down_proj_combined[0] = first_down
        del first_down

        # Fill remaining experts directly into pre-allocated tensor
        for e in range(1, config.n_expert):
            down_proj_e = load_param(
                proj[e][0],
                f"layer {i} {e} down_proj",
                dtype,
                verbose=debug_mode,
            )
            down_proj_combined[e] = down_proj_e
            del down_proj_e

        # Use incremental save if available
        if saver is not None:
            down_proj_ref = saver.store_early(
                f"transformer.h.{i}.mlp.experts.proj", down_proj_combined
            )
            state_dict[f"transformer.h.{i}.mlp.experts.proj"] = down_proj_ref
        else:
            state_dict[f"transformer.h.{i}.mlp.experts.proj"] = (
                down_proj_combined
            )
        del down_proj_combined
        del down_proj_weights[i]

    # convert separate q, k, v matrices into an interleaved qkv
    for i, (q, k, v) in list(qkv_weights.items()):
        if q is None or k is None or v is None:
            # split across different .bin files
            continue
        q = load_param(q, f"layer {i} q", dtype, verbose=debug_mode)
        k = load_param(k, f"layer {i} k", dtype, verbose=debug_mode)
        v = load_param(v, f"layer {i} v", dtype, verbose=debug_mode)
        q_per_kv = config.n_head // config.n_query_groups
        qs = torch.split(q, config.head_size * q_per_kv)
        ks = torch.split(k, config.head_size)
        vs = torch.split(v, config.head_size)
        cycled = [t for group in zip(qs, ks, vs) for t in group]
        qkv = torch.cat(cycled)

        state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv

        del qkv_weights[i]


def copy_weights_gemma_2(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.mlp.gate_proj.weight": None,
        "model.layers.{}.mlp.up_proj.weight": None,
        "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.post_attention_norm.weight",
        "model.layers.{}.pre_feedforward_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.layers.{}.post_feedforward_layernorm.weight": "transformer.h.{}.post_mlp_norm.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }

    transformer_wte_weight = None
    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, l_idx = layer_template(name, 2)
            qkv = qkv_weights.setdefault(l_idx, defaultdict(dict))
            gate_up_proj = gate_up_proj_weights.setdefault(l_idx, [None, None])
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param
            elif "gate_proj" in name:
                gate_up_proj[0] = param
            elif "up_proj" in name:
                gate_up_proj[1] = param
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l_idx)
        else:
            to_name = weight_map[name]

        if to_name == "transformer.wte.weight":
            transformer_wte_weight = param.clone().detach()

        param = load_param(param, name, dtype)
        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight
            )
            del transformer_wte_weight
            state_dict["lm_head.weight"] = param_saved

    # convert separate q, k, v matrices into an interleaved qkv
    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # split across different .bin files
                continue
            q = load_param(qkv["q_proj"], f"layer {i} q {weight_type}", dtype)
            k = load_param(qkv["k_proj"], f"layer {i} k {weight_type}", dtype)
            v = load_param(qkv["v_proj"], f"layer {i} v {weight_type}", dtype)
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]

    # convert separate gate proj and up proj into one tensor
    for i, (gate_proj, up_proj) in list(gate_up_proj_weights.items()):
        if gate_proj is None or up_proj is None:
            # split across different .bin files
            continue

        gate_proj = load_param(
            gate_proj, f"layer {i} gate_proj", dtype, verbose=debug_mode
        )
        up_proj = load_param(
            up_proj, f"layer {i} up_proj", dtype, verbose=debug_mode
        )

        # shape of gate_proj -> intermediate x hidden
        # shape of up_proj -> intermediate x hidden
        # after stacking -> intermediate x 2 x hidden
        # we are using stacking to interleave the rows of both tensors
        # so after stacking it's basically [gate_proj[0], up_proj[0], gate_proj[1], up_proj[1] ...]
        # and finally reshaping to 2*intermediate x hidden
        # we interleave so that during TP each GPU has access to the corresponding outputs of their
        # local gate and up projs so that they can apply swiglu locally.
        # for example let's say gate_up_proj = [gate_proj[0], up_proj[0], gate_proj[1], up_proj[1]]
        # and you have 2 GPUs
        # GPU 0 will get get [gate_proj[0], up_proj[0]] and GPU 1 will get [gate_proj[1], up_proj[1]]]
        # now they do not need to communicate with each other to apply swiglu.

        gate_up_proj = torch.stack((gate_proj, up_proj), dim=1).reshape(
            2 * gate_proj.size(0), -1
        )
        state_dict[f"transformer.h.{i}.mlp.gate_up_proj.weight"] = gate_up_proj
        del gate_up_proj_weights[i]


def copy_weights_phi(
    config: Config,
    qkv_weights: dict,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    if any(
        layer_name.startswith(("layers.", "transformer."))
        for layer_name in hf_weights
    ):
        raise ValueError(
            "You are using an outdated Phi checkpoint. Please reload it as described in 'tutorials/download_phi.md'"
        )

    # Phi uses NeoX style MLP
    # Phi-3 uses LLama style MLP
    # For phi-3 the gate and up-proj weights are already fused

    # maps from hf -> litgpt
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.input_layernorm.bias": "transformer.h.{}.norm_1.bias",
        "model.layers.{}.self_attn.q_proj.weight": None,
        "model.layers.{}.self_attn.q_proj.bias": None,
        "model.layers.{}.self_attn.k_proj.weight": None,
        "model.layers.{}.self_attn.k_proj.bias": None,
        "model.layers.{}.self_attn.v_proj.weight": None,
        "model.layers.{}.self_attn.v_proj.bias": None,
        "model.layers.{}.self_attn.dense.weight": "transformer.h.{}.attn.proj.weight",
        "model.layers.{}.self_attn.dense.bias": "transformer.h.{}.attn.proj.bias",
        "model.layers.{}.mlp.fc1.weight": "transformer.h.{}.mlp.fc.weight",
        "model.layers.{}.mlp.fc1.bias": "transformer.h.{}.mlp.fc.bias",
        "model.layers.{}.mlp.fc2.weight": "transformer.h.{}.mlp.proj.weight",
        "model.layers.{}.mlp.fc2.bias": "transformer.h.{}.mlp.proj.bias",
        "model.final_layernorm.weight": "transformer.ln_f.weight",
        "model.final_layernorm.bias": "transformer.ln_f.bias",
        "lm_head.weight": "lm_head.weight",
        "lm_head.bias": "lm_head.bias",
    }

    if config.name.startswith("Phi-3"):
        weight_map.update(
            {
                "model.layers.{}.self_attn.qkv_proj.weight": "transformer.h.{}.attn.attn.weight",
                "model.layers.{}.self_attn.o_proj.weight": "transformer.h.{}.attn.proj.weight",
                "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
                "model.layers.{}.mlp.down_proj.weight": "transformer.h.{}.mlp.proj.weight",
                "model.norm.weight": "transformer.ln_f.weight",
            }
        )

    for name, param in hf_weights.items():
        if name.startswith("model.layers."):
            from_name, l = layer_template(name, 2)
            qkv = qkv_weights.setdefault(l, defaultdict(dict))
            if "qkv_proj" in from_name:
                weight = load_param(param, f"layer {l} qkv", dtype)
                weight = qkv_reassemble(weight, config)
                to_name = weight_map[from_name].format(l)
                state_dict[to_name] = weight
                continue
            if any(w in from_name for w in ("q_proj", "k_proj", "v_proj")):
                weight_name, weight_type = from_name.split(".")[-2:]
                qkv[weight_type][weight_name] = param
            elif from_name.endswith("gate_up_proj.weight"):
                weight = load_param(param, f"layer {l} gate_up_proj", dtype)
                state_dict[f"transformer.h.{l}.mlp.gate_up_proj.weight"] = (
                    weight
                )
                continue
            to_name = weight_map[from_name]
            if to_name is None:
                continue
            to_name = to_name.format(l)
        else:
            to_name = weight_map[name]
        param = load_param(param, name, dtype, verbose=debug_mode)
        if saver is not None:
            param = saver.store_early(to_name, param)
        state_dict[to_name] = param

    for i in list(qkv_weights):
        for weight_type in list(qkv_weights[i]):
            qkv = qkv_weights[i][weight_type]
            if len(qkv) != 3:
                # split across different .bin files
                continue
            q = load_param(
                qkv["q_proj"],
                f"layer {i} q {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            k = load_param(
                qkv["k_proj"],
                f"layer {i} k {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            v = load_param(
                qkv["v_proj"],
                f"layer {i} v {weight_type}",
                dtype,
                verbose=debug_mode,
            )
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q, config.head_size * q_per_kv)
            ks = torch.split(k, config.head_size)
            vs = torch.split(v, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.{weight_type}"] = qkv
            del qkv_weights[i][weight_type]


def qkv_reassemble(
    param: Union[torch.Tensor, NotYetLoadedTensor], config: Config
) -> torch.Tensor:
    """Reassemble from a normal to an interleaved placement in a QKV matrix.
    [Q, Q, ..., K, K, ..., V, V, ...] --> [Q, K, V, Q, K, V, ...]
    """
    q, k, v = param.split(
        (
            config.n_head * config.head_size,
            config.n_query_groups * config.head_size,
            config.n_query_groups * config.head_size,
        )
    )
    qs = q.split(config.n_head // config.n_query_groups * config.head_size)
    ks = k.split(config.head_size)
    vs = v.split(config.head_size)
    interleaved = [t for group in zip(qs, ks, vs) for t in group]
    return torch.cat(interleaved)


def layer_template(layer_name: str, idx: int) -> Tuple[str, int]:
    split = layer_name.split(".")
    number = int(split[idx])
    split[idx] = "{}"
    from_name = ".".join(split)
    return from_name, number


def load_param(
    param: Union[torch.Tensor, NotYetLoadedTensor],
    name: str,
    dtype: Optional[torch.dtype],
    verbose=False,
) -> torch.Tensor:
    if hasattr(param, "_load_tensor"):
        # support tensors loaded via `lazy_load()`
        if verbose:
            print(f"Loading {name!r} into RAM")
        param = param._load_tensor()
    if (
        dtype is not None
        and type(dtype) is not NotYetLoadedTensor
        and dtype != param.dtype
    ):
        if verbose:
            print(f"Converting {name!r} from {param.dtype} to {dtype}")
        param = param.to(dtype)
    return param


def _load_shard(bin_file: Path) -> Dict[str, Any]:
    """Load a single shard file (safetensors or bin)."""
    if bin_file.suffix == ".safetensors":
        return load_safetensors(bin_file)
    return lazy_load(bin_file)


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False,
    profile: Optional[bool] = False,
) -> None:
    """
    Convert a Hugging Face Transformers checkpoint into a LitGPT compatible checkpoint.

    Arguments:
        checkpoint_dir: Where to save the downloaded files.
        model_name: The existing config name to load. This is useful to download alternative weights of existing
            architectures.
        dtype: The data type to convert the checkpoint files to. If not specified, the weights will remain in the
            dtype they are downloaded in.
        debug_mode: Prints the individual layers being loaded instead of a progress bar, which can be useful when
            developing and adding new models to LitGPT.
        profile: If True, collect and print memory/timing profiling information.
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    save_config(config, checkpoint_dir)

    # Initialize profiler
    profiler = ConversionProfiler(enabled=profile)

    # Initialize weight accumulators (used by some model types)
    qkv_weights = {}
    gate_up_proj_weights = {}
    down_proj_weights = {}

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, model_name)
    elif model_name.lower().startswith("gemma-2"):
        copy_fn = partial(
            copy_weights_gemma_2, config, qkv_weights, gate_up_proj_weights
        )
    elif model_name.lower().startswith("phi"):
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    elif model_name.lower().startswith(("qwen2.5", "qwq")):
        copy_fn = partial(
            copy_weights_qwen_2_5, config, qkv_weights, gate_up_proj_weights
        )
    elif model_name.lower().startswith(("qwen3")):
        copy_fn = partial(
            copy_weights_qwen_3,
            config,
            qkv_weights,
            gate_up_proj_weights,
            down_proj_weights,
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        copy_fn = partial(
            copy_weights_hf_llama,
            config,
            qkv_weights,
            gate_up_proj_weights,
            down_proj_weights,
        )
    # New models (Qwen)

    else:
        copy_fn = copy_weights_gpt_neox

    # initialize a new empty state dict to hold our new weights
    sd = (
        {}
    )  # This is the main state_dict that  is being updated and will get finally saved in the last line.

    # Load the json file containing weight mapping
    pytorch_bin_map_json_path = checkpoint_dir / "pytorch_model.bin.index.json"
    model_safetensor_map_json_path = (
        checkpoint_dir / "model.safetensors.index.json"
    )
    if (
        pytorch_bin_map_json_path.is_file()
    ):  # not all checkpoints have this file
        with open(pytorch_bin_map_json_path, encoding="utf-8") as json_map:
            bin_index = json.load(json_map)
        bin_files = {
            checkpoint_dir / bin for bin in bin_index["weight_map"].values()
        }
    elif model_safetensor_map_json_path.is_file():
        with open(
            model_safetensor_map_json_path, encoding="utf-8"
        ) as json_map:
            bin_index = json.load(json_map)
        bin_files = {
            checkpoint_dir / bin for bin in bin_index["weight_map"].values()
        }
    else:
        bin_files = set(checkpoint_dir.glob("*.bin")) | set(
            checkpoint_dir.glob("*.safetensors")
        )
        # some checkpoints serialize the training arguments
        bin_files = {f for f in bin_files if f.name != "training_args.bin"}
    if not bin_files:
        raise ValueError(
            f"Expected {str(checkpoint_dir)!r} to contain .bin files"
        )

    save_dir = checkpoint_dir / "yalis_checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    # with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
    with incremental_save(save_dir) as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end

        # Sort bin files for deterministic processing order
        sorted_bin_files = sorted(bin_files)

        profiler.snapshot(
            "Before processing shards",
            qkv_weights=qkv_weights,
            gate_up_proj_weights=gate_up_proj_weights,
            down_proj_weights=down_proj_weights,
            state_dict=sd,
        )

        num_shards = len(sorted_bin_files)

        if not debug_mode:
            # Phase 1: Load and process shards
            print(f"Phase 1: Loading {num_shards} shards...")
            with tqdm(
                total=num_shards,
                desc="Loading shards",
                bar_format="{desc}: {n}/{total}|{bar}| {elapsed}",
            ) as pbar:
                # Use prefetch pipeline: load next shard while processing current
                with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                    # Start loading first shard
                    next_future = prefetch_executor.submit(
                        _load_shard, sorted_bin_files[0]
                    )

                    for i, bin_file in enumerate(sorted_bin_files):
                        pbar.set_description(f"Loading: {bin_file.name}")

                        # Get current shard (already loading or loaded)
                        hf_weights = next_future.result()

                        # Start loading NEXT shard while we process current
                        if i + 1 < len(sorted_bin_files):
                            next_future = prefetch_executor.submit(
                                _load_shard, sorted_bin_files[i + 1]
                            )

                        # Process current shard
                        copy_fn(
                            sd,
                            hf_weights,
                            saver=saver,
                            dtype=dtype,
                            debug_mode=debug_mode,
                        )

                        # Free memory after processing each shard
                        del hf_weights
                        gc.collect()

                        # Update progress after shard is processed
                        pbar.update(1)

                        # Profile after each shard
                        profiler.snapshot(
                            f"After shard {i+1}/{num_shards}: {bin_file.name}",
                            qkv_weights=qkv_weights,
                            gate_up_proj_weights=gate_up_proj_weights,
                            down_proj_weights=down_proj_weights,
                            state_dict=sd,
                        )

            # Phase 2: Finalization message
            print("Phase 2: Finalizing weights...")
        else:
            # Debug mode: also use prefetch for consistency
            print(f"Phase 1: Loading {num_shards} shards (debug mode)...")
            with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                next_future = prefetch_executor.submit(
                    _load_shard, sorted_bin_files[0]
                )

                for i, bin_file in enumerate(sorted_bin_files):
                    print(
                        f"  [{i+1}/{num_shards}] Processing: {bin_file.name}"
                    )
                    hf_weights = next_future.result()

                    if i + 1 < num_shards:
                        next_future = prefetch_executor.submit(
                            _load_shard, sorted_bin_files[i + 1]
                        )

                    copy_fn(
                        sd,
                        hf_weights,
                        saver=saver,
                        dtype=dtype,
                        debug_mode=debug_mode,
                    )

                    del hf_weights
                    gc.collect()

                    # Profile after each shard (debug mode)
                    profiler.snapshot(
                        f"After shard {i+1}/{num_shards}: {bin_file.name}",
                        qkv_weights=qkv_weights,
                        gate_up_proj_weights=gate_up_proj_weights,
                        down_proj_weights=down_proj_weights,
                        state_dict=sd,
                    )

            print("Phase 2: Finalizing weights...")

        profiler.snapshot(
            "After all shards processed",
            qkv_weights=qkv_weights,
            gate_up_proj_weights=gate_up_proj_weights,
            down_proj_weights=down_proj_weights,
            state_dict=sd,
        )

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        saver.save(sd)

        profiler.snapshot(
            "After final save",
            qkv_weights=qkv_weights,
            gate_up_proj_weights=gate_up_proj_weights,
            down_proj_weights=down_proj_weights,
            state_dict=sd,
        )

    profiler.print_summary()
