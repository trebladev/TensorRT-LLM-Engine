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

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from utils.llm_data import llm_models_root

from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.convert import convert_hf_qwen35

_MODEL_DIR_NAMES = ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B")
_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
_VISION_KEY = "model.visual.blocks.0.norm1.weight"
_REQUIRED_KEYS = (
    _EMBEDDING_KEY,
    "model.language_model.norm.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.mlp.gate_proj.weight",
    "model.language_model.layers.0.mlp.up_proj.weight",
    "model.language_model.layers.0.mlp.down_proj.weight",
    "model.language_model.layers.0.linear_attn.conv1d.weight",
    "model.language_model.layers.0.linear_attn.norm.weight",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.3.input_layernorm.weight",
    "model.language_model.layers.3.post_attention_layernorm.weight",
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "model.language_model.layers.3.self_attn.k_proj.weight",
    "model.language_model.layers.3.self_attn.v_proj.weight",
    "model.language_model.layers.3.self_attn.o_proj.weight",
    "model.language_model.layers.3.self_attn.q_norm.weight",
    "model.language_model.layers.3.self_attn.k_norm.weight",
    _VISION_KEY,
)


def test_qwen35_layer_types_use_linear_classification() -> None:
    config = Qwen35Config(
        architecture="Qwen3_5ForConditionalGeneration",
        dtype="bfloat16",
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        vocab_size=32_000,
        max_position_embeddings=1024,
        decoder_layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )

    assert config.layer_types == ["linear", "linear", "linear", "attention"]


@pytest.fixture(scope="module")
def qwen35_checkpoint_dir() -> Path:
    models_root = llm_models_root()
    if models_root is None:
        pytest.skip("LLM_MODELS_ROOT is not available")

    model_dir = next(
        (models_root / name for name in _MODEL_DIR_NAMES if (models_root / name).is_dir()),
        None,
    )
    if model_dir is None:
        pytest.skip(f"Qwen3.5-2B is not available under {models_root}")

    assert (model_dir / "config.json").is_file()
    assert (model_dir / "model.safetensors.index.json").is_file()
    return model_dir


def _load_checkpoint_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    missing_keys = sorted(set(_REQUIRED_KEYS) - set(weight_map))
    assert not missing_keys, f"Missing expected Qwen3.5 weights: {missing_keys}"

    keys_by_shard: dict[str, list[str]] = {}
    for key in _REQUIRED_KEYS:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    tensors = {}
    for shard, keys in keys_by_shard.items():
        shard_path = model_dir / shard
        assert shard_path.is_file(), f"Missing Qwen3.5 checkpoint shard: {shard_path}"
        with safe_open(shard_path, framework="pt", device="cpu") as checkpoint:
            for key in keys:
                if key == _EMBEDDING_KEY:
                    # The converter does not depend on vocab size. Two real rows validate
                    # embedding and tied-lm-head conversion without loading about 1 GiB.
                    tensors[key] = checkpoint.get_slice(key)[:2]
                else:
                    tensors[key] = checkpoint.get_tensor(key)
    return tensors


@pytest.fixture(scope="module")
def real_qwen35_conversion(
    qwen35_checkpoint_dir: Path,
) -> tuple[Qwen35Config, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    config = Qwen35Config.from_hugging_face(qwen35_checkpoint_dir)
    source = _load_checkpoint_tensors(qwen35_checkpoint_dir)
    converted = convert_hf_qwen35(source, config)
    return config, source, converted


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qwen35_checkpoint_source_dtypes(
    real_qwen35_conversion: tuple[Qwen35Config, dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    config, source, _ = real_qwen35_conversion

    assert config.dtype == "bfloat16"
    assert config.decoder_layer_types[0] == "linear_attention"
    assert config.decoder_layer_types[3] == "full_attention"
    assert config.layer_types[0] == "linear"
    assert config.layer_types[3] == "attention"

    source_fp32_keys = {
        "model.language_model.layers.0.linear_attn.norm.weight",
        "model.language_model.layers.0.linear_attn.A_log",
    }
    actual_fp32_keys = {key for key, tensor in source.items() if tensor.dtype == torch.float32}
    assert actual_fp32_keys == source_fp32_keys
    for key, tensor in source.items():
        if key not in source_fp32_keys:
            assert tensor.dtype == torch.bfloat16, f"Unexpected dtype for {key}: {tensor.dtype}"


def test_convert_real_qwen35_checkpoint_dtypes_and_values(
    real_qwen35_conversion: tuple[Qwen35Config, dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    config, source, converted = real_qwen35_conversion

    expected_keys = {
        "transformer.vocab_embedding.weight",
        "transformer.ln_f.weight",
        "lm_head.weight",
        "transformer.layers.0.input_layernorm.weight",
        "transformer.layers.0.post_layernorm.weight",
        "transformer.layers.0.mlp.fc.weight",
        "transformer.layers.0.mlp.gate.weight",
        "transformer.layers.0.mlp.proj.weight",
        "transformer.layers.0.linear_attn.in_proj_qkv.weight",
        "transformer.layers.0.linear_attn.conv1d.weight",
        "transformer.layers.0.linear_attn.conv1d.bias",
        "transformer.layers.0.linear_attn.norm.weight",
        "transformer.layers.0.linear_attn.A_log",
        "transformer.layers.0.linear_attn.dt_bias",
        "transformer.layers.3.input_layernorm.weight",
        "transformer.layers.3.post_layernorm.weight",
        "transformer.layers.3.attention.qkv.query.weight",
        "transformer.layers.3.attention.qkv.gate.weight",
        "transformer.layers.3.attention.qkv.key.weight",
        "transformer.layers.3.attention.qkv.value.weight",
        "transformer.layers.3.attention.dense.proj.weight",
        "transformer.layers.3.attention.qkv.query_norm.weight",
        "transformer.layers.3.attention.qkv.key_norm.weight",
    }
    assert set(converted) == expected_keys

    fp32_keys = {
        "transformer.layers.0.linear_attn.A_log",
        "transformer.layers.0.linear_attn.dt_bias",
    }
    for key, tensor in converted.items():
        expected_dtype = torch.float32 if key in fp32_keys else torch.bfloat16
        assert tensor.dtype == expected_dtype, f"Unexpected dtype for {key}: {tensor.dtype}"
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()

    zero_centered_norms = {
        "model.language_model.norm.weight": "transformer.ln_f.weight",
        "model.language_model.layers.0.input_layernorm.weight": (
            "transformer.layers.0.input_layernorm.weight"
        ),
        "model.language_model.layers.0.post_attention_layernorm.weight": (
            "transformer.layers.0.post_layernorm.weight"
        ),
        "model.language_model.layers.3.input_layernorm.weight": (
            "transformer.layers.3.input_layernorm.weight"
        ),
        "model.language_model.layers.3.post_attention_layernorm.weight": (
            "transformer.layers.3.post_layernorm.weight"
        ),
        "model.language_model.layers.3.self_attn.q_norm.weight": (
            "transformer.layers.3.attention.qkv.query_norm.weight"
        ),
        "model.language_model.layers.3.self_attn.k_norm.weight": (
            "transformer.layers.3.attention.qkv.key_norm.weight"
        ),
    }
    for source_key, target_key in zero_centered_norms.items():
        expected = (source[source_key].float() + 1.0).to(torch.bfloat16)
        _assert_exact(converted[target_key], expected)

    linear_norm_key = "model.language_model.layers.0.linear_attn.norm.weight"
    _assert_exact(
        converted["transformer.layers.0.linear_attn.norm.weight"],
        source[linear_norm_key].to(torch.bfloat16),
    )

    for suffix in ("A_log", "dt_bias"):
        source_key = f"model.language_model.layers.0.linear_attn.{suffix}"
        target_key = f"transformer.layers.0.linear_attn.{suffix}"
        _assert_exact(converted[target_key], source[source_key].float())

    conv_source = source["model.language_model.layers.0.linear_attn.conv1d.weight"]
    _assert_exact(
        converted["transformer.layers.0.linear_attn.conv1d.weight"],
        conv_source.unsqueeze(-1),
    )
    _assert_exact(
        converted["transformer.layers.0.linear_attn.conv1d.bias"],
        torch.zeros(conv_source.shape[0], dtype=torch.bfloat16),
    )

    q_proj = source["model.language_model.layers.3.self_attn.q_proj.weight"]
    q_proj = q_proj.reshape(config.num_attention_heads, 2, config.head_size, -1)
    expected_query = q_proj[:, 0].reshape(-1, q_proj.shape[-1]).contiguous()
    expected_gate = q_proj[:, 1].reshape(-1, q_proj.shape[-1]).contiguous()
    _assert_exact(converted["transformer.layers.3.attention.qkv.query.weight"], expected_query)
    _assert_exact(converted["transformer.layers.3.attention.qkv.gate.weight"], expected_gate)

    passthrough_weights = {
        "model.language_model.layers.0.mlp.gate_proj.weight": (
            "transformer.layers.0.mlp.fc.weight"
        ),
        "model.language_model.layers.0.mlp.up_proj.weight": (
            "transformer.layers.0.mlp.gate.weight"
        ),
        "model.language_model.layers.0.mlp.down_proj.weight": (
            "transformer.layers.0.mlp.proj.weight"
        ),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (
            "transformer.layers.0.linear_attn.in_proj_qkv.weight"
        ),
        "model.language_model.layers.3.self_attn.k_proj.weight": (
            "transformer.layers.3.attention.qkv.key.weight"
        ),
        "model.language_model.layers.3.self_attn.v_proj.weight": (
            "transformer.layers.3.attention.qkv.value.weight"
        ),
        "model.language_model.layers.3.self_attn.o_proj.weight": (
            "transformer.layers.3.attention.dense.proj.weight"
        ),
    }
    for source_key, target_key in passthrough_weights.items():
        _assert_exact(converted[target_key], source[source_key])

    embedding = source[_EMBEDDING_KEY]
    _assert_exact(converted["transformer.vocab_embedding.weight"], embedding)
    _assert_exact(converted["lm_head.weight"], embedding)
    assert (
        converted["lm_head.weight"].data_ptr()
        != converted["transformer.vocab_embedding.weight"].data_ptr()
    )
    assert _VISION_KEY not in converted
