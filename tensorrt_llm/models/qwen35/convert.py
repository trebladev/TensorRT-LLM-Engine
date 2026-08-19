# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..convert_utils import iterate_shard_files, load_state_dict
from .config import Qwen35Config

if TYPE_CHECKING:
    import transformers


def _normalize_hf_name(name: str) -> str | None:
    if name.startswith("model.visual.") or name.startswith("visual."):
        return None
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith("language_model."):
        return "model." + name.removeprefix("language_model.")
    return name


def _to_bfloat16(param: torch.Tensor) -> torch.Tensor:
    return param.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()


def _convert_zero_centered_norm(param: torch.Tensor) -> torch.Tensor:
    return (
        (param.detach().to(device="cpu", dtype=torch.float32) + 1.0).to(torch.bfloat16).contiguous()
    )


def _split_query_and_gate(
    weight: torch.Tensor, config: Qwen35Config
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_rows = config.num_attention_heads * config.head_size * 2
    if weight.shape[0] != expected_rows:
        raise ValueError(f"Unexpected q_proj rows: expected {expected_rows}, got {weight.shape[0]}")
    weight = weight.reshape(config.num_attention_heads, 2, config.head_size, weight.shape[1])
    query = weight[:, 0].reshape(-1, weight.shape[-1])
    gate = weight[:, 1].reshape(-1, weight.shape[-1])
    return _to_bfloat16(query), _to_bfloat16(gate)


def _convert_parameter(
    name: str, param: torch.Tensor, config: Qwen35Config
) -> dict[str, torch.Tensor]:
    name = _normalize_hf_name(name)
    if name is None:
        return {}

    if name == "model.embed_tokens.weight":
        return {"transformer.vocab_embedding.weight": _to_bfloat16(param)}
    if name == "model.norm.weight":
        return {"transformer.ln_f.weight": _convert_zero_centered_norm(param)}
    if name == "lm_head.weight":
        return {"lm_head.weight": _to_bfloat16(param)}

    parts = name.split(".")
    if len(parts) < 5 or parts[:2] != ["model", "layers"]:
        return {}
    layer_idx = int(parts[2])
    suffix = ".".join(parts[3:])
    target_prefix = f"transformer.layers.{layer_idx}."

    common_names = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_layernorm.weight",
        "mlp.gate_proj.weight": "mlp.fc.weight",
        "mlp.up_proj.weight": "mlp.gate.weight",
        "mlp.down_proj.weight": "mlp.proj.weight",
    }
    if suffix in common_names:
        target = target_prefix + common_names[suffix]
        if suffix.endswith("layernorm.weight"):
            return {target: _convert_zero_centered_norm(param)}
        return {target: _to_bfloat16(param)}

    if suffix == "self_attn.q_proj.weight":
        query, gate = _split_query_and_gate(param, config)
        return {
            target_prefix + "attention.qkv.query.weight": query,
            target_prefix + "attention.qkv.gate.weight": gate,
        }

    full_attention_names = {
        "self_attn.k_proj.weight": "attention.qkv.key.weight",
        "self_attn.v_proj.weight": "attention.qkv.value.weight",
        "self_attn.o_proj.weight": "attention.dense.proj.weight",
    }
    if suffix in full_attention_names:
        return {target_prefix + full_attention_names[suffix]: _to_bfloat16(param)}
    if suffix == "self_attn.q_norm.weight":
        return {
            target_prefix + "attention.qkv.query_norm.weight": _convert_zero_centered_norm(param)
        }
    if suffix == "self_attn.k_norm.weight":
        return {target_prefix + "attention.qkv.key_norm.weight": _convert_zero_centered_norm(param)}

    linear_attention_names = {
        "linear_attn.in_proj_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight": "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_b.weight": "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight": "linear_attn.in_proj_a.weight",
        "linear_attn.out_proj.weight": "linear_attn.out_proj.weight",
        "linear_attn.norm.weight": "linear_attn.norm.weight",
    }
    if suffix in linear_attention_names:
        return {target_prefix + linear_attention_names[suffix]: _to_bfloat16(param)}
    if suffix == "linear_attn.conv1d.weight":
        weight = _to_bfloat16(param).unsqueeze(-1)
        bias = torch.zeros(param.shape[0], dtype=torch.bfloat16)
        return {
            target_prefix + "linear_attn.conv1d.weight": weight,
            target_prefix + "linear_attn.conv1d.bias": bias,
        }
    if suffix == "linear_attn.A_log":
        return {
            target_prefix + "linear_attn.A_log": param.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        }
    if suffix == "linear_attn.dt_bias":
        return {
            target_prefix + "linear_attn.dt_bias": param.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        }
    return {}


def convert_hf_qwen35(
    hf_model_or_state_dict: "transformers.PreTrainedModel" | Mapping[str, torch.Tensor],
    config: Qwen35Config,
) -> dict[str, torch.Tensor]:
    """Convert an in-memory dense Qwen3.5 checkpoint to TRT-LLM weights."""
    if isinstance(hf_model_or_state_dict, Mapping):
        state_dict = hf_model_or_state_dict
    else:
        state_dict = hf_model_or_state_dict.state_dict()

    weights: dict[str, torch.Tensor] = {}
    for name, param in state_dict.items():
        weights.update(_convert_parameter(name, param, config))
    if "lm_head.weight" not in weights and config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["transformer.vocab_embedding.weight"].clone()
    return weights


def load_weights_from_hf_checkpoint(
    model_dir: str | Path, config: Qwen35Config
) -> dict[str, torch.Tensor]:
    """Convert a local sharded Hugging Face Qwen3.5 checkpoint."""
    weights: dict[str, torch.Tensor] = {}
    for model_file in iterate_shard_files(str(model_dir), rank=0, progress_bar=False):
        state_dict = load_state_dict(model_file)
        for name, param in state_dict.items():
            weights.update(_convert_parameter(name, param, config))
        del state_dict
    if "lm_head.weight" not in weights and config.tie_word_embeddings:
        weights["lm_head.weight"] = weights["transformer.vocab_embedding.weight"].clone()
    return weights
