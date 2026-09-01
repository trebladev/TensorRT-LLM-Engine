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

import gc
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import tensorrt as trt
import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5VisionModel as HuggingFaceQwen35VisionModel,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionRotaryEmbedding
from utils.llm_data import llm_models_root
from utils.util import run_session

from tensorrt_llm import Builder
from tensorrt_llm.functional import Tensor
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.convert import convert_hf_qwen35
from tensorrt_llm.models.qwen35.model import Qwen35VisionModel
from tensorrt_llm.models.qwen35.vision_utils import prepare_qwen35_vision_position_inputs
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import Session

_MODEL_DIR_NAMES = ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B")
_GRID_THW = torch.tensor([[1, 2, 2], [1, 2, 2]], dtype=torch.int64)
_HIDDEN_COSINE_THRESHOLD = 0.995
_HIDDEN_MEAN_ABSOLUTE_ERROR = 0.25
_POOLED_COSINE_THRESHOLD = 0.9985


@dataclass
class _Qwen35VisionArtifacts:
    config: Qwen35Config
    hf_model: HuggingFaceQwen35VisionModel
    session: Session


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
    return model_dir


def _load_hf_vision_state_dict(
    model_dir: Path, parameter_names: set[str]
) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    keys_by_shard: dict[str, list[tuple[str, str]]] = {}
    for parameter_name in parameter_names:
        checkpoint_name = f"model.visual.{parameter_name}"
        keys_by_shard.setdefault(weight_map[checkpoint_name], []).append(
            (checkpoint_name, parameter_name)
        )

    state_dict = {}
    for shard, keys in keys_by_shard.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as checkpoint:
            for checkpoint_name, parameter_name in keys:
                state_dict[parameter_name] = checkpoint.get_tensor(checkpoint_name)
    return state_dict


def _build_vision_session(model: Qwen35VisionModel, config: Qwen35Config) -> Session:
    total_tokens = int((_GRID_THW[:, 0] * _GRID_THW[:, 1] * _GRID_THW[:, 2]).sum())
    patch_dim = (
        config.vision_in_channels * config.vision_temporal_patch_size * config.vision_patch_size**2
    )
    vision_head_dim = config.vision_hidden_size // config.vision_num_heads

    builder = Builder()
    builder_config = builder.create_builder_config(
        name="qwen35_vision",
        precision="bfloat16",
        strongly_typed=True,
    )
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.trt_builder_config.builder_optimization_level = 0
    builder_config.trt_builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30)
    network = builder.create_network()
    network.plugin_config.gemm_plugin = "bfloat16"

    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        hidden_states, pooled_output = model(
            Tensor(
                name="pixel_values",
                dtype=trt.bfloat16,
                shape=[total_tokens, patch_dim],
            ),
            Tensor(
                name="position_ids",
                dtype=trt.int32,
                shape=[4, total_tokens],
            ),
            Tensor(
                name="position_weights",
                dtype=trt.bfloat16,
                shape=[4, total_tokens],
            ),
            Tensor(
                name="rotary_cos",
                dtype=trt.bfloat16,
                shape=[total_tokens, vision_head_dim],
            ),
            Tensor(
                name="rotary_sin",
                dtype=trt.bfloat16,
                shape=[total_tokens, vision_head_dim],
            ),
            Tensor(
                name="vision_attention_mask",
                dtype=trt.bfloat16,
                shape=[1, 1, total_tokens, total_tokens],
            ),
        )
        hidden_states.mark_output("hidden_states", "bfloat16")
        pooled_output.mark_output("pooled_output", "bfloat16")

    engine = builder.build_engine(network, builder_config)
    assert engine is not None
    return Session.from_serialized_engine(engine)


@pytest.fixture(scope="module")
def qwen35_vision_artifacts(
    qwen35_checkpoint_dir: Path,
) -> _Qwen35VisionArtifacts:
    if not torch.cuda.is_available():
        pytest.skip("Qwen3.5 vision equivalence test requires CUDA")

    hf_root_config = AutoConfig.from_pretrained(qwen35_checkpoint_dir, local_files_only=True)
    hf_root_config.vision_config._attn_implementation = "eager"
    hf_model = HuggingFaceQwen35VisionModel(hf_root_config.vision_config).eval()
    state_dict = _load_hf_vision_state_dict(qwen35_checkpoint_dir, set(hf_model.state_dict()))
    hf_model.load_state_dict(state_dict, strict=True)
    hf_model = hf_model.to(device="cuda", dtype=torch.bfloat16)

    config = Qwen35Config.from_hugging_face(qwen35_checkpoint_dir)
    trt_model = Qwen35VisionModel(config)
    converted = convert_hf_qwen35(
        {f"model.visual.{name}": value for name, value in state_dict.items()},
        config,
    )
    for name, parameter in trt_model.named_parameters():
        parameter.value = converted[f"visual.{name}"]
    session = _build_vision_session(trt_model, config)

    del trt_model
    del converted
    del state_dict
    gc.collect()

    artifacts = _Qwen35VisionArtifacts(config, hf_model, session)
    yield artifacts

    del artifacts
    del session
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()


def _cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
    )


def _hf_rotary_frequencies_fp32(
    artifacts: _Qwen35VisionArtifacts, grid_thw: torch.Tensor
) -> torch.Tensor:
    rotary_dim = artifacts.config.vision_hidden_size // artifacts.config.vision_num_heads // 2
    original_rotary_embedding = artifacts.hf_model.rotary_pos_emb
    rotary_device = artifacts.hf_model.pos_embed.weight.device
    artifacts.hf_model.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(rotary_dim).to(rotary_device)
    try:
        with torch.inference_mode():
            rotary_frequencies = artifacts.hf_model.rot_pos_emb(grid_thw)
    finally:
        artifacts.hf_model.rotary_pos_emb = original_rotary_embedding
    return torch.cat((rotary_frequencies, rotary_frequencies), dim=-1)


def test_qwen35_vision_position_inputs_match_hugging_face(
    qwen35_vision_artifacts: _Qwen35VisionArtifacts,
) -> None:
    artifacts = qwen35_vision_artifacts
    metadata = prepare_qwen35_vision_position_inputs(
        _GRID_THW,
        artifacts.config,
        dtype=torch.bfloat16,
        device="cuda",
    )

    with torch.inference_mode():
        actual_position_embeddings = (
            artifacts.hf_model.pos_embed(metadata.position_ids)
            * metadata.position_weights.unsqueeze(-1)
        ).sum(dim=0)
        expected_position_embeddings = artifacts.hf_model.fast_pos_embed_interpolate(_GRID_THW)
    rotary_frequencies = _hf_rotary_frequencies_fp32(artifacts, _GRID_THW)

    torch.testing.assert_close(
        actual_position_embeddings,
        expected_position_embeddings,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        metadata.rotary_cos,
        rotary_frequencies.cos().to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        metadata.rotary_sin,
        rotary_frequencies.sin().to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    assert torch.count_nonzero(metadata.attention_mask[..., :4, :4]) == 0
    assert torch.count_nonzero(metadata.attention_mask[..., 4:, 4:]) == 0
    assert torch.all(metadata.attention_mask[..., :4, 4:] < 0)
    assert torch.all(metadata.attention_mask[..., 4:, :4] < 0)


def test_qwen35_vision_large_grid_rotary_inputs_match_hugging_face(
    qwen35_vision_artifacts: _Qwen35VisionArtifacts,
) -> None:
    artifacts = qwen35_vision_artifacts
    grid_thw = torch.tensor([[1, 22, 34]], dtype=torch.int64)
    metadata = prepare_qwen35_vision_position_inputs(
        grid_thw,
        artifacts.config,
        dtype=torch.bfloat16,
        device="cuda",
    )

    rotary_frequencies = _hf_rotary_frequencies_fp32(artifacts, grid_thw)

    torch.testing.assert_close(
        metadata.rotary_cos,
        rotary_frequencies.cos().to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        metadata.rotary_sin,
        rotary_frequencies.sin().to(torch.bfloat16),
        rtol=0,
        atol=0,
    )


def test_qwen35_vision_session_matches_hugging_face(
    qwen35_vision_artifacts: _Qwen35VisionArtifacts,
) -> None:
    artifacts = qwen35_vision_artifacts
    total_tokens = int((_GRID_THW[:, 0] * _GRID_THW[:, 1] * _GRID_THW[:, 2]).sum())
    patch_dim = (
        artifacts.config.vision_in_channels
        * artifacts.config.vision_temporal_patch_size
        * artifacts.config.vision_patch_size**2
    )
    torch.manual_seed(1234)
    pixel_values = torch.randn((total_tokens, patch_dim), device="cuda", dtype=torch.bfloat16)
    metadata = prepare_qwen35_vision_position_inputs(
        _GRID_THW,
        artifacts.config,
        dtype=torch.bfloat16,
        device="cuda",
    )

    with torch.inference_mode():
        reference = artifacts.hf_model(pixel_values, _GRID_THW)
    outputs = run_session(
        artifacts.session,
        {
            "pixel_values": pixel_values,
            "position_ids": metadata.position_ids,
            "position_weights": metadata.position_weights,
            "rotary_cos": metadata.rotary_cos,
            "rotary_sin": metadata.rotary_sin,
            "vision_attention_mask": metadata.attention_mask,
        },
    )
    actual_hidden_states = outputs["hidden_states"].float()
    expected_hidden_states = reference.last_hidden_state.float()
    actual_pooled_output = outputs["pooled_output"].float()
    expected_pooled_output = reference.pooler_output.float()

    hidden_cosine = _cosine_similarity(actual_hidden_states, expected_hidden_states)
    hidden_mean_absolute_error = float(
        (actual_hidden_states - expected_hidden_states).abs().mean().item()
    )
    assert hidden_cosine >= _HIDDEN_COSINE_THRESHOLD
    assert hidden_mean_absolute_error <= _HIDDEN_MEAN_ABSOLUTE_ERROR

    pooled_cosine = _cosine_similarity(actual_pooled_output, expected_pooled_output)
    assert pooled_cosine >= _POOLED_COSINE_THRESHOLD
    torch.testing.assert_close(
        actual_pooled_output,
        expected_pooled_output,
        rtol=0.1,
        atol=0.06,
    )
