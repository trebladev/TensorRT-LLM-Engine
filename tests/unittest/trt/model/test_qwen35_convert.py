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
import math
from pathlib import Path

import pytest
import tensorrt as trt
import torch
from safetensors import safe_open
from utils.llm_data import llm_models_root

from tensorrt_llm import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.convert import convert_hf_qwen35
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM, Qwen35VisionModel
from tensorrt_llm.network import net_guard

_MODEL_DIR_NAMES = ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B")
_EMBEDDING_KEY = "model.language_model.embed_tokens.weight"
_VISION_KEY = "model.visual.blocks.0.norm1.weight"
_REQUIRED_KEYS = (
    _EMBEDDING_KEY,
    "model.language_model.norm.weight",
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
    "model.language_model.layers.0.linear_attn.in_proj_z.weight",
    "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    "model.language_model.layers.0.linear_attn.in_proj_b.weight",
    "model.language_model.layers.0.linear_attn.out_proj.weight",
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
    assert not config.has_vision


def _small_qwen35_config(mapping: Mapping | None = None) -> Qwen35Config:
    return Qwen35Config(
        architecture="Qwen35ForCausalLM",
        dtype="bfloat16",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_size=8,
        vocab_size=128,
        max_position_embeddings=128,
        hidden_act="silu",
        norm_epsilon=1e-6,
        tie_word_embeddings=True,
        rotary_embedding_dim=8,
        mrope_section=[1, 1, 2],
        decoder_layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
        vision_depth=2,
        vision_hidden_size=16,
        vision_intermediate_size=32,
        vision_num_heads=4,
        vision_in_channels=3,
        vision_patch_size=2,
        vision_temporal_patch_size=2,
        vision_spatial_merge_size=2,
        vision_num_position_embeddings=16,
        vision_output_hidden_size=32,
        vision_hidden_act="gelu_pytorch_tanh",
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=122,
        vision_end_token_id=123,
        mapping=mapping,
    )


def test_qwen35_vision_parameter_names_and_conversion() -> None:
    config = _small_qwen35_config()
    vision_model = Qwen35VisionModel(config)
    target_parameters = {
        f"visual.{name}": parameter for name, parameter in vision_model.named_parameters()
    }
    source = {
        f"model.{name}": torch.arange(math.prod(parameter.shape), dtype=torch.float32).reshape(
            parameter.shape
        )
        for name, parameter in target_parameters.items()
    }

    converted = convert_hf_qwen35(source, config)
    assert set(converted) == set(target_parameters)
    for source_name, source_tensor in source.items():
        target_name = source_name.removeprefix("model.")
        expected = source_tensor.to(torch.bfloat16)
        _assert_exact(converted[target_name], expected)
        assert converted[target_name].device.type == "cpu"
        assert converted[target_name].is_contiguous()

    tp2_config = _small_qwen35_config(Mapping(world_size=2, rank=1, tp_size=2))
    tp2_converted = convert_hf_qwen35(source, tp2_config)
    for name in converted:
        _assert_exact(tp2_converted[name], converted[name])

    model = Qwen35ForCausalLM(config)
    assert isinstance(model.visual, Qwen35VisionModel)
    assert dict(model.named_parameters())["visual.patch_embed.proj.weight"].shape == (
        16,
        3,
        2,
        2,
        2,
    )


def test_qwen35_vision_graph_construction() -> None:
    model = Qwen35VisionModel(_small_qwen35_config())
    network = Builder().create_network()
    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        hidden_states, merged_hidden_states = model(
            Tensor(name="pixel_values", dtype=trt.bfloat16, shape=[8, 24]),
            Tensor(name="position_ids", dtype=trt.int32, shape=[4, 8]),
            Tensor(name="position_weights", dtype=trt.bfloat16, shape=[4, 8]),
            Tensor(name="rotary_cos", dtype=trt.bfloat16, shape=[8, 4]),
            Tensor(name="rotary_sin", dtype=trt.bfloat16, shape=[8, 4]),
            Tensor(
                name="vision_attention_mask",
                dtype=trt.bfloat16,
                shape=[1, 1, 8, 8],
            ),
        )

    assert hidden_states.shape == (8, 16)
    assert merged_hidden_states.shape == (2, 32)


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


def test_qwen35_checkpoint_vision_parameter_names(qwen35_checkpoint_dir: Path) -> None:
    config = Qwen35Config.from_hugging_face(qwen35_checkpoint_dir)
    vision_model = Qwen35VisionModel(config)
    model_names = {f"visual.{name}" for name, _ in vision_model.named_parameters()}
    index_path = qwen35_checkpoint_dir / "model.safetensors.index.json"
    checkpoint_names = {
        name.removeprefix("model.")
        for name in json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        if name.startswith("model.visual.")
    }

    assert checkpoint_names == model_names


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
    assert config.has_vision
    assert config.vision_depth == 24
    assert config.vision_hidden_size == 1024
    assert config.vision_intermediate_size == 4096
    assert config.vision_num_heads == 16
    assert config.vision_in_channels == 3
    assert config.vision_patch_size == 16
    assert config.vision_temporal_patch_size == 2
    assert config.vision_spatial_merge_size == 2
    assert config.vision_num_position_embeddings == 2304
    assert config.vision_output_hidden_size == config.hidden_size
    assert config.vision_hidden_act == "gelu_pytorch_tanh"
    assert config.vision_deepstack_visual_indexes == []
    assert config.image_token_id == 248056
    assert config.video_token_id == 248057
    assert config.vision_start_token_id == 248053
    assert config.vision_end_token_id == 248054

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
        "transformer.layers.0.linear_attn.in_proj_z.weight",
        "transformer.layers.0.linear_attn.in_proj_a.weight",
        "transformer.layers.0.linear_attn.in_proj_b.weight",
        "transformer.layers.0.linear_attn.out_proj.weight",
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
        "visual.blocks.0.norm1.weight",
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
        "model.language_model.layers.0.linear_attn.in_proj_z.weight": (
            "transformer.layers.0.linear_attn.in_proj_z.weight"
        ),
        "model.language_model.layers.0.linear_attn.in_proj_a.weight": (
            "transformer.layers.0.linear_attn.in_proj_a.weight"
        ),
        "model.language_model.layers.0.linear_attn.in_proj_b.weight": (
            "transformer.layers.0.linear_attn.in_proj_b.weight"
        ),
        "model.language_model.layers.0.linear_attn.out_proj.weight": (
            "transformer.layers.0.linear_attn.out_proj.weight"
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
        _VISION_KEY: "visual.blocks.0.norm1.weight",
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


def _merge_gdn_qkv_shards(
    shards: tuple[torch.Tensor, torch.Tensor], config: Qwen35Config
) -> torch.Tensor:
    local_key_rows = (
        config.linear_num_key_heads // config.mapping.tp_size * config.linear_key_head_dim
    )
    local_value_rows = (
        config.linear_num_value_heads // config.mapping.tp_size * config.linear_value_head_dim
    )
    sections = [
        torch.split(shard, [local_key_rows, local_key_rows, local_value_rows], dim=0)
        for shard in shards
    ]
    return torch.cat(
        [
            torch.cat([rank_sections[section_idx] for rank_sections in sections], dim=0)
            for section_idx in range(3)
        ],
        dim=0,
    ).contiguous()


def test_convert_real_qwen35_checkpoint_tp2_shards_reconstruct_tp1(
    qwen35_checkpoint_dir: Path,
    real_qwen35_conversion: tuple[Qwen35Config, dict[str, torch.Tensor], dict[str, torch.Tensor]],
) -> None:
    _, source, full_weights = real_qwen35_conversion
    tp_configs = tuple(
        Qwen35Config.from_hugging_face(
            qwen35_checkpoint_dir,
            mapping=Mapping(world_size=2, rank=rank, tp_size=2),
        )
        for rank in range(2)
    )
    rank_weights = tuple(convert_hf_qwen35(source, config) for config in tp_configs)

    for weights in rank_weights:
        assert set(weights) == set(full_weights)
        for name, tensor in weights.items():
            expected_dtype = (
                torch.float32 if name.endswith(("A_log", "dt_bias")) else torch.bfloat16
            )
            assert tensor.dtype == expected_dtype
            assert tensor.device.type == "cpu"
            assert tensor.is_contiguous()

    column_parallel_keys = {
        "lm_head.weight",
        "transformer.layers.0.mlp.fc.weight",
        "transformer.layers.0.mlp.gate.weight",
        "transformer.layers.0.linear_attn.in_proj_z.weight",
        "transformer.layers.0.linear_attn.in_proj_a.weight",
        "transformer.layers.0.linear_attn.in_proj_b.weight",
        "transformer.layers.0.linear_attn.A_log",
        "transformer.layers.0.linear_attn.dt_bias",
        "transformer.layers.3.attention.qkv.query.weight",
        "transformer.layers.3.attention.qkv.gate.weight",
        "transformer.layers.3.attention.qkv.key.weight",
        "transformer.layers.3.attention.qkv.value.weight",
    }
    for key in column_parallel_keys:
        reconstructed = torch.cat([weights[key] for weights in rank_weights], dim=0)
        _assert_exact(reconstructed, full_weights[key])

    row_parallel_keys = {
        "transformer.layers.0.mlp.proj.weight",
        "transformer.layers.0.linear_attn.out_proj.weight",
        "transformer.layers.3.attention.dense.proj.weight",
    }
    for key in row_parallel_keys:
        reconstructed = torch.cat([weights[key] for weights in rank_weights], dim=1)
        _assert_exact(reconstructed, full_weights[key])

    gdn_semantic_keys = {
        "transformer.layers.0.linear_attn.in_proj_qkv.weight",
        "transformer.layers.0.linear_attn.conv1d.weight",
        "transformer.layers.0.linear_attn.conv1d.bias",
    }
    for key in gdn_semantic_keys:
        reconstructed = _merge_gdn_qkv_shards(
            (rank_weights[0][key], rank_weights[1][key]), tp_configs[0]
        )
        _assert_exact(reconstructed, full_weights[key])

    sharded_keys = column_parallel_keys | row_parallel_keys | gdn_semantic_keys
    for key in set(full_weights) - sharded_keys:
        _assert_exact(rank_weights[0][key], full_weights[key])
        _assert_exact(rank_weights[1][key], full_weights[key])
