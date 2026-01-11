# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import gc
import json
from collections import defaultdict
from functools import partial
import os
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Union, Any

from tqdm import tqdm
import torch
from lightning.fabric.utilities.load import (
    _NotYetLoadedTensor as NotYetLoadedTensor,
)
from safetensors.torch import load_file as load_safetensors


DEFAULT_SAFETENSOR_CHUNK_SIZE = int(
    os.environ.get("YALIS_SAFETENSOR_CHUNK_SIZE", "1")
)

if DEFAULT_SAFETENSOR_CHUNK_SIZE < 1:
    raise ValueError("YALIS_SAFETENSOR_CHUNK_SIZE must be at least 1")

from config import Config, find_multiple
from litgpt.utils import (
    extend_checkpoint_dir,
    lazy_load,
    save_config,
    # incremental_save,
)
from safetensors import safe_open
from safetensor_saver import incremental_save


def copy_weights_gpt_neox(
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

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

        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_falcon(
    model_name: str,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(1, len(hf_weights))

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
        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_qwen_3(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[Any, List[Optional[NotYetLoadedTensor]]],
    down_proj_weights: Dict[Any, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(
            1, len(hf_weights) + len(qkv_weights)
        )

    transformer_wte_weight = None

    for from_name, param in hf_weights.items():

        if "model.layers" in from_name:
            name_template, l = layer_template(from_name, 2)
            if "experts" in from_name:
                name_template, e = layer_template(name_template, 5)
            else:
                e = None
            # print (f"{name_template}, {e}")

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

            if any(w in from_name for w in ("down_proj")) and e is not None:
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

        if progress_per_file is not None:
            pbar.update(progress_per_file)

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

            if progress_per_file is not None:
                pbar.update(progress_per_file)

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

                gate_up_proj_combined = []
                for e in range(config.n_expert):

                    gate_proj_e = gate_proj[e]
                    up_proj_e = up_proj[e]
                    gate_proj_e = load_param(
                        gate_proj_e,
                        f"layer {i} gate_proj {e}",
                        dtype,
                        verbose=debug_mode,
                    )
                    up_proj_e = load_param(
                        up_proj_e,
                        f"layer {i} up_proj {e}",
                        dtype,
                        verbose=debug_mode,
                    )
                    gate_up_proj_e = torch.stack(
                        (gate_proj_e, up_proj_e), dim=1
                    ).reshape(2 * gate_proj_e.size(0), -1)

                    gate_up_proj_combined.append(gate_up_proj_e)

                gate_up_proj_combined = torch.stack(
                    gate_up_proj_combined, dim=0
                )
                state_dict[f"transformer.h.{i}.mlp.experts.gate_up_proj"] = (
                    gate_up_proj_combined
                )
                del gate_up_proj_combined
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
                del gate_up_proj_weights[i]

            if progress_per_file is not None:
                pbar.update(progress_per_file)

    for i in list(down_proj_weights):
        for weight_type in list(down_proj_weights[i]):
            down_proj = down_proj_weights[i][weight_type]["down_proj"]
            if len(down_proj) != config.n_expert:
                continue
            down_proj_combined = []
            for e in range(config.n_expert):
                down_proj_e = down_proj[e]
                down_proj_e = load_param(
                    down_proj_e,
                    f"layer {i} down_proj {e}",
                    dtype,
                    verbose=debug_mode,
                )
                down_proj_combined.append(down_proj_e)
            down_proj_combined = torch.stack(down_proj_combined, dim=0)
            print(
                f"Down projection for layer {i} combined: {down_proj_combined.shape}"
            )
            state_dict[f"transformer.h.{i}.mlp.experts.proj"] = (
                down_proj_combined
            )
            del down_proj_weights[i]

            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_qwen_2_5(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(
            1, len(hf_weights) + len(qkv_weights)
        )

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

        if progress_per_file is not None:
            pbar.update(progress_per_file)

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

            if progress_per_file is not None:
                pbar.update(progress_per_file)

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
            if progress_per_file is not None:
                pbar.update(progress_per_file)


def copy_weights_hf_llama(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    down_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(
            1, len(hf_weights) + len(qkv_weights)
        )

    transformer_wte_weight = None
    for name, param in hf_weights.items():
        # print (f"{name}")
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
                # print (f"{name} Gate up proj for layer {l} expert {e}: {gate_up_proj}")
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
                if progress_per_file is not None:
                    pbar.update(progress_per_file)

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

                if progress_per_file is not None:
                    pbar.update(progress_per_file)
            gc.collect()

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
            gc.collect()
            state_dict[to_name] = param_saved

        else:
            state_dict[to_name] = param

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight
            )
            del transformer_wte_weight
            gc.collect()
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
            gate_up_proj_combined = []
            all_present = True
            for e, proj_e in list(proj.items()):
                gate_proj_e = proj_e[0]
                up_proj_e = proj_e[1]
                if gate_proj_e is None or up_proj_e is None:
                    all_present = False
                    break
                gate_proj_e = load_param(
                    gate_proj_e,
                    f"layer {i} {e} gate_proj",
                    dtype,
                    verbose=debug_mode,
                )
                up_proj_e = load_param(
                    up_proj_e,
                    f"layer {i} {e} up_proj",
                    dtype,
                    verbose=debug_mode,
                )
                gate_up_proj_e = torch.stack(
                    (gate_proj_e, up_proj_e), dim=1
                ).reshape(2 * gate_proj_e.size(0), -1)
                gate_up_proj_combined.append(gate_up_proj_e)
            if not all_present:
                continue

            gate_up_proj_combined = torch.stack(gate_up_proj_combined, dim=0)
            state_dict[f"transformer.h.{i}.mlp.experts.gate_up_proj"] = (
                gate_up_proj_combined
            )
            assert (
                gate_up_proj_combined.shape[0] == config.n_expert
            ), "Gate up projection combined shape should be equal to the number of experts"
            # print (f"Gate up projection for layer {i} combined: {gate_up_proj_combined.shape}")
            del gate_up_proj_combined
            del gate_up_proj_weights[i]

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    for i, proj in list(down_proj_weights.items()):
        assert isinstance(
            proj, dict
        ), "Down projection weights should be a dictionary"
        if len(list(proj.items())) != config.n_expert:
            # Not all experts are present
            continue
        down_proj_combined = []
        all_present = True
        for e, proj_e in list(proj.items()):
            down_proj_e = proj_e[0]
            if down_proj_e is None:
                all_present = False
                break
            down_proj_e = load_param(
                down_proj_e,
                f"layer {i} {e} down_proj",
                dtype,
                verbose=debug_mode,
            )
            down_proj_combined.append(down_proj_e)

        if not all_present:
            continue
        down_proj_combined = torch.stack(down_proj_combined, dim=0)
        assert (
            down_proj_combined.shape[0] == config.n_expert
        ), "Down projection combined shape should be equal to the number of experts"
        state_dict[f"transformer.h.{i}.mlp.experts.proj"] = down_proj_combined
        # print (f"Down projection for layer {i} combined: {down_proj_combined.shape}")
        del down_proj_combined
        del down_proj_weights[i]

        if progress_per_file is not None:
            pbar.update(progress_per_file)

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
        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_gemma_2(
    config: Config,
    qkv_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    gate_up_proj_weights: Dict[int, List[Optional[NotYetLoadedTensor]]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(
            1, len(hf_weights) + len(qkv_weights)
        )

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

        if progress_per_file is not None:
            pbar.update(progress_per_file)

    if "lm_head.weight" not in state_dict:
        if transformer_wte_weight is not None:
            param_saved = saver.store_early(
                "lm_head.weight", transformer_wte_weight
            )
            del transformer_wte_weight
            gc.collect()
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
            if progress_per_file is not None:
                pbar.update(progress_per_file)

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

        if progress_per_file is not None:
            pbar.update(progress_per_file)


def copy_weights_phi(
    config: Config,
    qkv_weights: dict,
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
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

    if progress_per_file is not None:
        progress_per_file = progress_per_file / max(
            1, len(hf_weights) + len(qkv_weights)
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
        if progress_per_file is not None:
            pbar.update(progress_per_file)

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
            if progress_per_file is not None:
                pbar.update(progress_per_file)


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


def copy_weights_gpt_oss(
    config: Config,
    qkv_weights: Dict[int, Dict[str, Any]],
    moe_weights: Dict[int, Dict[str, Any]],
    state_dict: Dict[str, torch.Tensor],
    hf_weights: Dict[str, Union[torch.Tensor, NotYetLoadedTensor]],
    saver: Optional[incremental_save] = None,
    dtype: Optional[torch.dtype] = None,
    pbar: Optional[tqdm] = None,
    progress_per_file: Optional[float] = None,
    debug_mode: Optional[bool] = False,
) -> None:
    """Convert GPT-OSS HF checkpoint to yalis format.
    
    Handles:
    - Quantized MoE weights (blocks, scales, biases)
    - Attention weights with biases (q/k/v/o projections)
    - Sinks parameter
    - Router weights and biases
    - Layer norms (RMSNorm)
    """
    
    weight_map = {
        "model.embed_tokens.weight": "transformer.wte.weight",
        "model.layers.{}.input_layernorm.weight": "transformer.h.{}.norm_1.weight",
        "model.layers.{}.post_attention_layernorm.weight": "transformer.h.{}.norm_2.weight",
        "model.norm.weight": "transformer.ln_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    
    for name, param in hf_weights.items():
        if "model.layers" in name:
            from_name, number = layer_template(name, idx=2)
            
            # Handle attention weights (q, k, v projections - collect for deferred assembly)
            if "self_attn.q_proj.weight" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["q_proj"] = param
            elif "self_attn.q_proj.bias" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["q_proj.bias"] = param
            elif "self_attn.k_proj.weight" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["k_proj"] = param
            elif "self_attn.k_proj.bias" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["k_proj.bias"] = param
            elif "self_attn.v_proj.weight" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["v_proj"] = param
            elif "self_attn.v_proj.bias" in name:
                if number not in qkv_weights:
                    qkv_weights[number] = {}
                qkv_weights[number]["v_proj.bias"] = param
            
            # Handle attention output projection
            elif "self_attn.o_proj.weight" in name:
                to_name = f"transformer.h.{number}.attn.proj.weight"
                param = load_param(param, name, dtype, verbose=debug_mode)
                state_dict[to_name] = param
            elif "self_attn.o_proj.bias" in name:
                # Skip o_proj bias - model uses config.bias=False for proj
                if debug_mode:
                    print(f"Skipping {name} - model has no proj bias (config.bias=False)")
                pass
            
            # Handle sinks (reshape from (n_head,) to (n_head, 1, 1))
            elif "self_attn.sinks" in name:
                to_name = f"transformer.h.{number}.sinks"
                param = load_param(param, name, dtype, verbose=debug_mode)
                # Reshape from (n_head,) to (n_head, 1, 1) to match model expectation
                if param.dim() == 1:
                    param = param.view(-1, 1, 1)
                state_dict[to_name] = param
            
            # Handle layer norms
            elif "input_layernorm.weight" in name:
                to_name = f"transformer.h.{number}.norm_1.weight"
                param = load_param(param, name, dtype, verbose=debug_mode)
                state_dict[to_name] = param
            elif "post_attention_layernorm.weight" in name:
                to_name = f"transformer.h.{number}.norm_2.weight"
                param = load_param(param, name, dtype, verbose=debug_mode)
                state_dict[to_name] = param
            
            # Handle MoE router (maps to gate in GptOssMoE)
            elif "mlp.router.weight" in name:
                to_name = f"transformer.h.{number}.mlp.gate.weight"
                param = load_param(param, name, dtype, verbose=debug_mode)
                state_dict[to_name] = param
            elif "mlp.router.bias" in name:
                # GptOssMoE gate has no bias - skip this
                pass
            
            # Handle MoE expert weights (quantized format - collect for deferred dequantization)
            elif "mlp.experts.gate_up_proj_blocks" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["gate_up_proj_blocks"] = param
            elif "mlp.experts.gate_up_proj_scales" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["gate_up_proj_scales"] = param
            elif "mlp.experts.gate_up_proj_bias" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["gate_up_proj_bias"] = param
            elif "mlp.experts.down_proj_blocks" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["down_proj_blocks"] = param
            elif "mlp.experts.down_proj_scales" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["down_proj_scales"] = param
            elif "mlp.experts.down_proj_bias" in name:
                if number not in moe_weights:
                    moe_weights[number] = {}
                moe_weights[number]["down_proj_bias"] = param
            
        # Handle embeddings, norms, and lm_head via weight_map
        else:
            to_name = weight_map.get(name)
            if to_name is not None:
                param = load_param(param, name, dtype, verbose=debug_mode)
                
                # Pad embedding and lm_head to match padded_vocab_size
                if ("wte" in to_name or "lm_head" in to_name) and param.dim() >= 1:
                    vocab_size_checkpoint = param.shape[0]
                    padded_vocab_size = config.padded_vocab_size
                    if vocab_size_checkpoint < padded_vocab_size:
                        # Pad with zeros to match padded vocab size
                        pad_size = padded_vocab_size - vocab_size_checkpoint
                        padding = torch.zeros(
                            (pad_size,) + param.shape[1:],
                            dtype=param.dtype,
                            device=param.device
                        )
                        param = torch.cat([param, padding], dim=0)
                        if debug_mode:
                            print(f"Padded {to_name} from {vocab_size_checkpoint} to {padded_vocab_size}")
                
                state_dict[to_name] = param
        
        if pbar is not None and progress_per_file is not None:
            pbar.update(progress_per_file / len(hf_weights))
    
    # Deferred QKV assembly (interleave q, k, v for GQA)
    for i in list(qkv_weights.keys()):
        qkv = qkv_weights[i]
        if len(qkv) == 6:  # All q/k/v weights and biases present
            # Load weights
            q_weight = load_param(qkv["q_proj"], f"layer {i} q_proj", dtype, verbose=debug_mode)
            k_weight = load_param(qkv["k_proj"], f"layer {i} k_proj", dtype, verbose=debug_mode)
            v_weight = load_param(qkv["v_proj"], f"layer {i} v_proj", dtype, verbose=debug_mode)
            
            # Interleave for GQA
            q_per_kv = config.n_head // config.n_query_groups
            qs = torch.split(q_weight, config.head_size * q_per_kv)
            ks = torch.split(k_weight, config.head_size)
            vs = torch.split(v_weight, config.head_size)
            cycled = [t for group in zip(qs, ks, vs) for t in group]
            qkv_weight = torch.cat(cycled)
            state_dict[f"transformer.h.{i}.attn.attn.weight"] = qkv_weight
            
            # Load and interleave biases
            q_bias = load_param(qkv["q_proj.bias"], f"layer {i} q_proj.bias", dtype, verbose=debug_mode)
            k_bias = load_param(qkv["k_proj.bias"], f"layer {i} k_proj.bias", dtype, verbose=debug_mode)
            v_bias = load_param(qkv["v_proj.bias"], f"layer {i} v_proj.bias", dtype, verbose=debug_mode)
            
            qs_bias = torch.split(q_bias, config.head_size * q_per_kv)
            ks_bias = torch.split(k_bias, config.head_size)
            vs_bias = torch.split(v_bias, config.head_size)
            cycled_bias = [t for group in zip(qs_bias, ks_bias, vs_bias) for t in group]
            qkv_bias = torch.cat(cycled_bias)
            state_dict[f"transformer.h.{i}.attn.attn.bias"] = qkv_bias
            
            del qkv_weights[i]
            if progress_per_file is not None and pbar is not None:
                pbar.update(progress_per_file)
    
    # Deferred MoE weight dequantization and assembly (MXFP4 format)
    # FP4 lookup table for MXFP4 decoding
    FP4_VALUES = [
        +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    
    for i in list(moe_weights.keys()):
        moe = moe_weights[i]
        if len(moe) == 6:  # All gate_up and down weights present
            # Load quantized gate_up weights
            gate_up_blocks = load_param(moe["gate_up_proj_blocks"], f"layer {i} gate_up_proj_blocks", None, verbose=debug_mode)  # Keep as uint8
            gate_up_scales = load_param(moe["gate_up_proj_scales"], f"layer {i} gate_up_proj_scales", None, verbose=debug_mode)  # Keep as uint8
            gate_up_bias = load_param(moe["gate_up_proj_bias"], f"layer {i} gate_up_proj_bias", dtype, verbose=debug_mode)
            
            # Dequantize gate_up using MXFP4 decoding (following GPT-OSS reference)
            # Blocks shape: (n_experts, out_features, in_blocks, block_size=16)
            # Each uint8 byte contains TWO 4-bit FP4 values (nibbles)
            n_experts, out_features, in_blocks, block_size = gate_up_blocks.shape
            
            # Split into low and high nibbles, then interleave
            gate_up_blocks_lo = gate_up_blocks & 0x0F
            gate_up_blocks_hi = gate_up_blocks >> 4
            gate_up_blocks = torch.stack((gate_up_blocks_lo, gate_up_blocks_hi), dim=-1)
            gate_up_blocks = gate_up_blocks.view(*gate_up_blocks.shape[:-2], gate_up_blocks.shape[-2] * 2)
            
            # Convert scales: uint8 -> int32, subtract bias (127)
            gate_up_scales_adj = gate_up_scales.to(torch.int32) - 127
            
            # Create FP4 lookup table
            target_dtype = dtype if dtype is not None else torch.bfloat16
            fp4_lut = torch.tensor(FP4_VALUES, dtype=target_dtype, device=gate_up_blocks.device)
            
            # Decode: lookup FP4 values, then scale using ldexp (value * 2^scale)
            # Let broadcasting handle the scale expansion
            gate_up_weight = torch.ldexp(fp4_lut[gate_up_blocks.to(torch.int32)], gate_up_scales_adj.unsqueeze(-1))
            
            # Debug: Check if weights look reasonable
            if debug_mode or i == 0:
                print(f"Layer {i} gate_up - min: {gate_up_weight.min():.6f}, max: {gate_up_weight.max():.6f}, mean: {gate_up_weight.mean():.6f}, std: {gate_up_weight.std():.6f}")
            
            # Reshape to final dimensions: (n_experts, out_features, in_features)
            # After reshape we get (E, 5760, 2880) which is already correct!
            gate_up_weight = gate_up_weight.view(n_experts, out_features, -1)
            
            # Save with correct names for GptOssMoE (mlp1 = gate_up combined)
            state_dict[f"transformer.h.{i}.mlp.mlp1_weight"] = gate_up_weight
            state_dict[f"transformer.h.{i}.mlp.mlp1_bias"] = gate_up_bias
            
            # Load and dequantize down weights (same MXFP4 process)
            down_blocks = load_param(moe["down_proj_blocks"], f"layer {i} down_proj_blocks", None, verbose=debug_mode)
            down_scales = load_param(moe["down_proj_scales"], f"layer {i} down_proj_scales", None, verbose=debug_mode)
            down_bias = load_param(moe["down_proj_bias"], f"layer {i} down_proj_bias", dtype, verbose=debug_mode)
            
            n_experts_d, out_features_d, in_blocks_d, block_size_d = down_blocks.shape
            
            # Split nibbles
            down_blocks_lo = down_blocks & 0x0F
            down_blocks_hi = down_blocks >> 4
            down_blocks = torch.stack((down_blocks_lo, down_blocks_hi), dim=-1)
            down_blocks = down_blocks.view(*down_blocks.shape[:-2], down_blocks.shape[-2] * 2)
            
            # Convert scales
            down_scales_adj = down_scales.to(torch.int32) - 127
            
            # Decode
            down_decoded = fp4_lut[down_blocks.to(torch.int32)]
            down_weight = torch.ldexp(down_decoded, down_scales_adj.unsqueeze(-1))
            
            # Reshape to final dimensions: (n_experts, hidden_size, intermediate_size)
            in_features = in_blocks_d * block_size_d * 2  # blocks * 32
            down_weight = down_weight.view(n_experts_d, out_features_d, in_features)
            
            # Save with correct names for GptOssMoE (mlp2 = down projection)
            state_dict[f"transformer.h.{i}.mlp.mlp2_weight"] = down_weight
            state_dict[f"transformer.h.{i}.mlp.mlp2_bias"] = down_bias
            
            del moe_weights[i]
            if progress_per_file is not None and pbar is not None:
                pbar.update(progress_per_file)


def _chunked_safetensors(
    bin_file: Path, chunk_size: int = DEFAULT_SAFETENSOR_CHUNK_SIZE
):
    """Yield small dicts of tensors instead of loading the whole file at once."""
    with safe_open(bin_file, framework="pt") as reader:
        keys = list(reader.keys())
        total_keys = len(keys)
        for start in range(0, total_keys, chunk_size):
            end = min(start + chunk_size, total_keys)
            chunk = {name: reader.get_tensor(name) for name in keys[start:end]}
            yield chunk, len(chunk), total_keys


@torch.inference_mode()
def convert_hf_checkpoint(
    checkpoint_dir: Path,
    *,
    model_name: Optional[str] = None,
    dtype: Optional[str] = None,
    debug_mode: Optional[bool] = False,
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
    """
    checkpoint_dir = extend_checkpoint_dir(checkpoint_dir)
    pprint(locals())

    if model_name is None:
        model_name = checkpoint_dir.name
    if dtype is not None:
        dtype = getattr(torch, dtype)

    config = Config.from_name(model_name)
    
    # For GPT-OSS, load actual dimensions from HF config.json
    if model_name.lower().startswith("gpt-oss") or config.mlp_class_name == "GptOssMoE":
        hf_config_path = checkpoint_dir / "config.json"
        if hf_config_path.exists():
            import json
            with open(hf_config_path, "r") as f:
                hf_config = json.load(f)
            
            # Update config with actual HF values
            if "vocab_size" in hf_config:
                config.vocab_size = hf_config["vocab_size"]
                # Recompute padded vocab size
                config.padded_vocab_size = find_multiple(config.vocab_size, config.padding_multiple)
            
            if "hidden_size" in hf_config:
                config.n_embd = hf_config["hidden_size"]
            
            if "num_attention_heads" in hf_config:
                config.n_head = hf_config["num_attention_heads"]
            
            if "num_key_value_heads" in hf_config:
                config.n_query_groups = hf_config["num_key_value_heads"]
            
            if "num_hidden_layers" in hf_config:
                config.n_layer = hf_config["num_hidden_layers"]
            
            # GPT-OSS uses head_dim instead of computing from n_embd/n_head
            if "head_dim" in hf_config:
                config.head_size = hf_config["head_dim"]
            elif config.n_embd and config.n_head:
                # Fallback: compute from n_embd // n_head
                config.head_size = config.n_embd // config.n_head
            
            if debug_mode:
                print(f"Updated config from HF config.json:")
                print(f"  vocab_size: {config.vocab_size}")
                print(f"  n_embd: {config.n_embd}")
                print(f"  n_head: {config.n_head}")
                print(f"  n_query_groups: {config.n_query_groups}")
                print(f"  head_size: {config.head_size}")
    
    # Save config to main checkpoint directory
    save_config(config, checkpoint_dir)
    
    # Also save to yalis_checkpoints subdirectory where model will load from
    save_dir = checkpoint_dir / "yalis_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    save_config(config, save_dir)

    if "falcon" in model_name:
        copy_fn = partial(copy_weights_falcon, model_name)
    elif model_name.lower().startswith("gpt-oss") or config.mlp_class_name == "GptOssMoE":
        # GPT-OSS models with quantized MoE
        qkv_weights = {}
        moe_weights = {}
        copy_fn = partial(
            copy_weights_gpt_oss, config, qkv_weights, moe_weights
        )
    elif model_name.lower().startswith("gemma-2"):
        qkv_weights = {}
        gate_up_proj_weights = {}
        copy_fn = partial(
            copy_weights_gemma_2, config, qkv_weights, gate_up_proj_weights
        )
    elif model_name.lower().startswith("phi"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        copy_fn = partial(copy_weights_phi, config, qkv_weights)
    elif model_name.lower().startswith(("qwen2.5", "qwq")):
        qkv_weights = {}
        gate_up_proj_weights = {}
        copy_fn = partial(
            copy_weights_qwen_2_5, config, qkv_weights, gate_up_proj_weights
        )
    elif model_name.lower().startswith(("qwen3")):
        qkv_weights = {}
        gate_up_proj_weights = {}
        down_proj_weights = {}
        copy_fn = partial(
            copy_weights_qwen_3,
            config,
            qkv_weights,
            gate_up_proj_weights,
            down_proj_weights,
        )
    elif config.mlp_class_name in ("LLaMAMLP", "GemmaMLP", "LLaMAMoE"):
        # holder to reconstitute the split q, k, v
        qkv_weights = {}
        gate_up_proj_weights = {}
        down_proj_weights = {}
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

    # save_dir already created when saving config above (reuse the same path)

    # with incremental_save(checkpoint_dir / "lit_model.pth") as saver:
    with incremental_save(save_dir) as saver:
        # for checkpoints that split the QKV across several files, we need to keep all the bin files
        # open, so we use `ExitStack` to close them all together at the end

        if not debug_mode:
            # Using tqdm progress bar when not in debug mode

            total_size = max(
                1, sum(os.path.getsize(bin_file) for bin_file in bin_files)
            )
            total_progress = 100

            with tqdm(
                total=total_progress,
                desc="Initializing",
                bar_format="{desc}{percentage:3.0f}%|{bar}| {elapsed}<{remaining}, {rate_fmt}",
            ) as pbar:
                for bin_file in sorted(bin_files):
                    pbar.set_description(f"Loading weights: {bin_file.name}")
                    current_file_size = os.path.getsize(bin_file)
                    progress_per_file = (
                        current_file_size / total_size
                    ) * total_progress

                    if bin_file.suffix == ".safetensors":
                        for chunk, chunk_len, total_keys in _chunked_safetensors(
                            bin_file
                        ):
                            chunk_progress = (
                                progress_per_file * chunk_len / total_keys
                                if total_keys > 0 and progress_per_file is not None
                                else None
                            )
                            copy_fn(
                                sd,
                                chunk,
                                saver=saver,
                                dtype=dtype,
                                pbar=pbar,
                                progress_per_file=chunk_progress,
                                debug_mode=debug_mode,
                            )
                            del chunk
                    else:
                        hf_weights = load_safetensors(bin_file)
                        copy_fn(
                            sd,
                            hf_weights,
                            saver=saver,
                            dtype=dtype,
                            pbar=pbar,
                            progress_per_file=progress_per_file,
                            debug_mode=debug_mode,
                        )
                gc.collect()

                if pbar.n < total_progress:
                    pbar.update(total_progress - pbar.n)
                pbar.close()
        else:
            # Handling files without progress bar in debug mode
            for bin_file in sorted(bin_files):
                if bin_file.suffix == ".safetensors":
                    for chunk, _, _ in _chunked_safetensors(bin_file):
                        copy_fn(
                            sd,
                            chunk,
                            saver=saver,
                            dtype=dtype,
                            debug_mode=debug_mode,
                        )
                        del chunk
                else:
                    hf_weights = lazy_load(bin_file)
                    copy_fn(
                        sd,
                        hf_weights,
                        saver=saver,
                        dtype=dtype,
                        debug_mode=debug_mode,
                    )

        print(f"Saving converted checkpoint to {checkpoint_dir}")
        saver.save(sd)
