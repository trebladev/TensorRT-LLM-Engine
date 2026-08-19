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
from pathlib import Path

import pytest
import tensorrt as trt
import torch
from transformers import Qwen3_5ForConditionalGeneration
from utils.llm_data import llm_models_root
from utils.util import run_session

from tensorrt_llm import Builder
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.models import MODEL_MAP, Qwen35ForCausalLM
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import Session

_MODEL_DIR_NAMES = ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B")
_PREFILL_CASES = (
    (17,),
    (33, 65),
    (1, 64, 127, 256),
    (511, 1024),
    (8193,),
)
_MAX_BATCH_SIZE = max(len(request_lengths) for request_lengths in _PREFILL_CASES)
_MAX_INPUT_LENGTH = max(max(request_lengths) for request_lengths in _PREFILL_CASES)
_MAX_SEQUENCE_LENGTH = _MAX_INPUT_LENGTH
_MAX_NUM_TOKENS = max(sum(request_lengths) for request_lengths in _PREFILL_CASES)
_OPT_BATCH_SIZE = 2
_OPT_NUM_TOKENS = 512


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


def _make_input_ids(request_lengths: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(request_lengths)
    max_input_length = max(request_lengths)
    input_ids = torch.zeros((batch_size, max_input_length), dtype=torch.int64, device="cuda")
    attention_mask = torch.zeros_like(input_ids)
    for request_idx, input_length in enumerate(request_lengths):
        input_ids[request_idx, :input_length] = (
            torch.arange(input_length, device="cuda") + request_idx * 997
        ) % 32_000 + 1
        attention_mask[request_idx, :input_length] = 1
    return input_ids, attention_mask


def _reference_logits(
    model: Qwen3_5ForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    request_lengths: tuple[int, ...],
) -> torch.Tensor:
    with torch.inference_mode():
        outputs = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        request_indices = torch.arange(len(request_lengths), device="cuda")
        last_token_indices = torch.tensor(request_lengths, device="cuda") - 1
        last_hidden_states = outputs.last_hidden_state[request_indices, last_token_indices]
        logits = model.lm_head(last_hidden_states)
    return logits.float().cpu()


@pytest.fixture(scope="module")
def qwen35_references(qwen35_checkpoint_dir: Path) -> dict[tuple[int, ...], torch.Tensor]:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen35_checkpoint_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).cuda()
    model.eval()
    references = {}
    for request_lengths in _PREFILL_CASES:
        input_ids, attention_mask = _make_input_ids(request_lengths)
        references[request_lengths] = _reference_logits(
            model, input_ids, attention_mask, request_lengths
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return references


def _build_qwen35_session(model_dir: Path) -> tuple[Session, Qwen35Config]:
    model = Qwen35ForCausalLM.from_hugging_face(model_dir, dtype="bfloat16")
    config = model.config

    builder = Builder()
    builder_config = builder.create_builder_config(
        name="qwen35",
        precision="bfloat16",
        strongly_typed=True,
    )
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    network.plugin_config.gpt_attention_plugin = "bfloat16"
    network.plugin_config.gemm_plugin = "bfloat16"
    network.plugin_config.mamba_conv1d_plugin = "bfloat16"
    network.plugin_config.remove_input_padding = True
    network.plugin_config.paged_kv_cache = False
    network.plugin_config.paged_state = False

    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        inputs = model.prepare_inputs(
            max_batch_size=_MAX_BATCH_SIZE,
            max_input_len=_MAX_INPUT_LENGTH,
            max_seq_len=_MAX_SEQUENCE_LENGTH,
            max_num_tokens=_MAX_NUM_TOKENS,
            opt_batch_size=_OPT_BATCH_SIZE,
            opt_num_tokens=_OPT_NUM_TOKENS,
            use_cache=True,
        )
        model(**inputs)

    engine = builder.build_engine(network, builder_config)
    assert engine is not None
    del model
    del inputs
    del network
    del builder_config
    del builder
    gc.collect()
    torch.cuda.empty_cache()
    return Session.from_serialized_engine(engine), config


def _engine_input_names(session: Session) -> set[str]:
    engine = session.engine
    return {
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.INPUT
    }


def _mrope_cache(config: Qwen35Config) -> torch.Tensor:
    _, rotary_cos_sin = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
        num_pos=config.max_position_embeddings,
        dim=config.rotary_embedding_dim,
        theta=config.rotary_base,
    )
    return torch.from_numpy(rotary_cos_sin).cuda()


@pytest.fixture(scope="module")
def qwen35_session(
    qwen35_checkpoint_dir: Path,
    qwen35_references: dict[tuple[int, ...], torch.Tensor],
) -> tuple[Session, Qwen35Config, torch.Tensor]:
    del qwen35_references
    session, config = _build_qwen35_session(qwen35_checkpoint_dir)
    return session, config, _mrope_cache(config)


def _prefill_inputs(
    session: Session,
    config: Qwen35Config,
    padded_input_ids: torch.Tensor,
    request_lengths: tuple[int, ...],
    mrope_cache: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch_size = len(request_lengths)
    context_lengths = torch.tensor(request_lengths, dtype=torch.int32, device="cuda")
    host_context_lengths = context_lengths.cpu()
    last_token_ids = torch.cumsum(context_lengths, dim=0, dtype=torch.int32)
    packed_input_ids = torch.cat(
        [padded_input_ids[idx, :input_length] for idx, input_length in enumerate(request_lengths)]
    ).to(dtype=torch.int32)
    position_ids = torch.cat(
        [
            torch.arange(input_length, dtype=torch.int32, device="cuda")
            for input_length in request_lengths
        ]
    )
    cu_seqlens = torch.cat([torch.zeros(1, dtype=torch.int32, device="cuda"), last_token_ids])
    attention_layer_ids = [
        layer_idx
        for layer_idx, layer_type in enumerate(config.decoder_layer_types)
        if layer_type == "full_attention"
    ]
    recurrent_layer_ids = [
        layer_idx
        for layer_idx, layer_type in enumerate(config.decoder_layer_types)
        if layer_type == "linear_attention"
    ]
    candidates = {
        "input_ids": packed_input_ids,
        "position_ids": position_ids,
        "context_lengths": context_lengths,
        "last_token_ids": last_token_ids,
        "sequence_length": context_lengths.clone(),
        "cache_indirection": torch.zeros(
            (batch_size, 1, _MAX_SEQUENCE_LENGTH), dtype=torch.int32, device="cuda"
        ),
        "host_request_types": torch.zeros(batch_size, dtype=torch.int32),
        "host_context_lengths": host_context_lengths,
        "host_past_key_value_lengths": torch.zeros(batch_size, dtype=torch.int32),
        "host_max_attention_window_sizes": torch.full(
            (len(attention_layer_ids),),
            _MAX_SEQUENCE_LENGTH,
            dtype=torch.int32,
        ),
        "host_sink_token_length": torch.zeros(1, dtype=torch.int32),
        "host_runtime_perf_knobs": torch.full((16,), -1, dtype=torch.int64),
        "host_context_progress": torch.zeros(1, dtype=torch.int64),
        "mrope_rotary_cos_sin": mrope_cache.expand(batch_size, -1).contiguous(),
        "mrope_position_deltas": torch.zeros((batch_size, 1), dtype=torch.int32, device="cuda"),
        "gated_delta_cu_seqlens": cu_seqlens,
        "state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
        "host_has_initial_state": torch.zeros(batch_size, dtype=torch.int8),
    }

    kv_shape = (
        batch_size,
        2,
        config.num_key_value_heads,
        _MAX_SEQUENCE_LENGTH,
        config.head_size,
    )
    for attention_idx in range(len(attention_layer_ids)):
        candidates[f"past_key_value_{attention_idx}"] = torch.zeros(
            kv_shape, dtype=torch.bfloat16, device="cuda"
        )

    conv_dim = (
        2 * config.linear_num_key_heads * config.linear_key_head_dim
        + config.linear_num_value_heads * config.linear_value_head_dim
    )
    conv_shape = (
        batch_size,
        config.linear_conv_kernel_dim - 1,
        conv_dim,
    )
    recurrent_shape = (
        batch_size,
        config.linear_num_value_heads,
        config.linear_value_head_dim,
        config.linear_key_head_dim,
    )
    for layer_idx in recurrent_layer_ids:
        candidates[f"past_conv_state_{layer_idx}"] = torch.zeros(
            conv_shape, dtype=torch.bfloat16, device="cuda"
        )
        candidates[f"past_recurrent_state_{layer_idx}"] = torch.zeros(
            recurrent_shape, dtype=torch.float32, device="cuda"
        )

    input_names = _engine_input_names(session)
    required_qwen35_inputs = {
        "gated_delta_cu_seqlens",
        "state_slot_mapping",
        "host_has_initial_state",
        "mrope_rotary_cos_sin",
        "mrope_position_deltas",
    }
    assert required_qwen35_inputs <= input_names
    missing_inputs = input_names - candidates.keys()
    assert not missing_inputs, f"Missing Qwen3.5 engine inputs: {sorted(missing_inputs)}"
    return {name: candidates[name] for name in input_names}


def test_qwen35_is_registered() -> None:
    assert MODEL_MAP["Qwen35ForCausalLM"] is Qwen35ForCausalLM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3.5 engine test requires CUDA")
@pytest.mark.parametrize(
    "request_lengths",
    [
        pytest.param((17,), id="batch1-len17"),
        pytest.param((33, 65), id="batch2-ragged-65"),
        pytest.param((1, 64, 127, 256), id="batch4-ragged-256"),
        pytest.param((511, 1024), id="batch2-ragged-1024"),
        pytest.param((8193,), id="batch1-len8193"),
    ],
)
def test_qwen35_prefill_session_matches_hugging_face(
    request_lengths: tuple[int, ...],
    qwen35_references: dict[tuple[int, ...], torch.Tensor],
    qwen35_session: tuple[Session, Qwen35Config, torch.Tensor],
) -> None:
    padded_input_ids, _ = _make_input_ids(request_lengths)
    reference = qwen35_references[request_lengths]
    session, config, mrope_cache = qwen35_session
    inputs = _prefill_inputs(session, config, padded_input_ids, request_lengths, mrope_cache)

    outputs = run_session(session, inputs)
    actual = outputs["logits"].float().cpu()

    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, atol=0.35, rtol=0.05)
    assert torch.equal(actual.argmax(dim=-1), reference.argmax(dim=-1))
