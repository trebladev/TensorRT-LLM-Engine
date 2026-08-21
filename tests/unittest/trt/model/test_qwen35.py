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
from dataclasses import dataclass
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
from tensorrt_llm.models.qwen35.convert import load_weights_from_hf_checkpoint
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import Session

_MODEL_DIR_NAMES = ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B")
_TEST_SOURCE_LAYER_INDICES = (0, 3)
_NUM_TEST_LAYERS = len(_TEST_SOURCE_LAYER_INDICES)
_TEST_VOCAB_SIZE = 8_192
_DECODE_INPUT_LENGTH = 17
_DECODE_STEPS = 4
_PREFILL_CASES = (
    (17,),
    (33, 65),
    (1, 64, 127, 256),
    (511, 1024),
    (8193,),
)
_MAX_BATCH_SIZE = max(len(request_lengths) for request_lengths in _PREFILL_CASES)
_MAX_INPUT_LENGTH = max(max(request_lengths) for request_lengths in _PREFILL_CASES)
_MAX_SEQUENCE_LENGTH = _MAX_INPUT_LENGTH + 8
_MAX_NUM_TOKENS = max(sum(request_lengths) for request_lengths in _PREFILL_CASES)
_OPT_BATCH_SIZE = 2
_OPT_NUM_TOKENS = 512
_MAX_PREFILL_TOP_LOGIT_GAP = 0.25


@dataclass
class _Qwen35ReferenceData:
    prefill_logits: dict[tuple[int, ...], torch.Tensor]
    greedy_input_tokens: list[int]
    greedy_logits: list[torch.Tensor]


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
        ) % (_TEST_VOCAB_SIZE - 1) + 1
        attention_mask[request_idx, :input_length] = 1
    return input_ids, attention_mask


def _truncate_hf_model(model: Qwen3_5ForConditionalGeneration) -> None:
    text_model = model.model.language_model
    source_layers = list(text_model.layers)
    text_model.layers = torch.nn.ModuleList(
        [source_layers[layer_idx] for layer_idx in _TEST_SOURCE_LAYER_INDICES]
    )
    model.config.text_config.num_hidden_layers = _NUM_TEST_LAYERS
    source_layer_types = list(model.config.text_config.layer_types)
    model.config.text_config.layer_types = [
        source_layer_types[layer_idx] for layer_idx in _TEST_SOURCE_LAYER_INDICES
    ]


def _truncate_trt_config(config: Qwen35Config) -> None:
    config.vocab_size = _TEST_VOCAB_SIZE
    config.num_hidden_layers = _NUM_TEST_LAYERS
    config.decoder_layer_types = [
        config.decoder_layer_types[layer_idx] for layer_idx in _TEST_SOURCE_LAYER_INDICES
    ]
    config.layer_types = [config.layer_types[layer_idx] for layer_idx in _TEST_SOURCE_LAYER_INDICES]


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
        logits = model.lm_head(last_hidden_states)[:, :_TEST_VOCAB_SIZE]
    return logits.float().cpu()


@pytest.fixture(scope="module")
def qwen35_references(qwen35_checkpoint_dir: Path) -> _Qwen35ReferenceData:
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        qwen35_checkpoint_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _truncate_hf_model(model)
    model.cuda()
    model.eval()
    references = {}
    for request_lengths in _PREFILL_CASES:
        input_ids, attention_mask = _make_input_ids(request_lengths)
        references[request_lengths] = _reference_logits(
            model, input_ids, attention_mask, request_lengths
        )

    greedy_input_ids, greedy_attention_mask = _make_input_ids((_DECODE_INPUT_LENGTH,))
    greedy_logits = [references[(_DECODE_INPUT_LENGTH,)]]
    greedy_input_tokens = []
    for _ in range(_DECODE_STEPS):
        next_token = int(greedy_logits[-1].argmax(dim=-1).item())
        greedy_input_tokens.append(next_token)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.int64, device="cuda")
        greedy_input_ids = torch.cat([greedy_input_ids, next_token_tensor], dim=1)
        greedy_attention_mask = torch.ones_like(greedy_input_ids)
        greedy_logits.append(
            _reference_logits(
                model,
                greedy_input_ids,
                greedy_attention_mask,
                (greedy_input_ids.shape[1],),
            )
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return _Qwen35ReferenceData(references, greedy_input_tokens, greedy_logits)


def _build_qwen35_session(model_dir: Path) -> tuple[Session, Qwen35Config]:
    config = Qwen35Config.from_hugging_face(model_dir, dtype="bfloat16")
    _truncate_trt_config(config)
    model = Qwen35ForCausalLM(config)
    weights = load_weights_from_hf_checkpoint(model_dir, config)
    required_weights = {name for name, _ in model.named_parameters()}
    selected_weights = {
        name: value
        for name, value in weights.items()
        if name in required_weights and not name.startswith("transformer.layers.")
    }
    for target_layer_idx, source_layer_idx in enumerate(_TEST_SOURCE_LAYER_INDICES):
        source_prefix = f"transformer.layers.{source_layer_idx}."
        target_prefix = f"transformer.layers.{target_layer_idx}."
        for name, value in weights.items():
            if name.startswith(source_prefix):
                target_name = target_prefix + name.removeprefix(source_prefix)
                if target_name in required_weights:
                    selected_weights[target_name] = value
    for name in ("transformer.vocab_embedding.weight", "lm_head.weight"):
        selected_weights[name] = selected_weights[name][:_TEST_VOCAB_SIZE].contiguous()
    model.load(selected_weights)
    del weights

    builder = Builder()
    builder_config = builder.create_builder_config(
        name="qwen35",
        precision="bfloat16",
        strongly_typed=True,
    )
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.trt_builder_config.builder_optimization_level = 0
    builder_config.trt_builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 16 << 30)
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
    qwen35_references: _Qwen35ReferenceData,
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
        "host_has_initial_state": torch.zeros(batch_size, dtype=torch.int32),
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
    required_qwen35_inputs = {"mrope_rotary_cos_sin", "mrope_position_deltas"}
    if recurrent_layer_ids:
        required_qwen35_inputs.update(
            {"gated_delta_cu_seqlens", "state_slot_mapping", "host_has_initial_state"}
        )
    assert required_qwen35_inputs <= input_names
    missing_inputs = input_names - candidates.keys()
    assert not missing_inputs, f"Missing Qwen3.5 engine inputs: {sorted(missing_inputs)}"
    return {name: candidates[name] for name in input_names}


def _generation_inputs(
    session: Session,
    config: Qwen35Config,
    input_ids: torch.Tensor,
    request_lengths: tuple[int, ...],
    step: int,
    mrope_cache: torch.Tensor,
    previous_outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    batch_size = len(request_lengths)
    context_lengths = torch.tensor(request_lengths, dtype=torch.int32, device="cuda")
    past_lengths = context_lengths + step - 1
    sequence_lengths = past_lengths + 1
    last_token_ids = torch.arange(1, batch_size + 1, dtype=torch.int32, device="cuda")
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
        "input_ids": input_ids.to(dtype=torch.int32),
        "position_ids": past_lengths,
        "context_lengths": context_lengths,
        "last_token_ids": last_token_ids,
        "sequence_length": sequence_lengths,
        "cache_indirection": torch.zeros(
            (batch_size, 1, _MAX_SEQUENCE_LENGTH), dtype=torch.int32, device="cuda"
        ),
        "host_request_types": torch.ones(batch_size, dtype=torch.int32),
        "host_context_lengths": context_lengths.cpu(),
        "host_past_key_value_lengths": past_lengths.cpu(),
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
        "gated_delta_cu_seqlens": torch.arange(batch_size + 1, dtype=torch.int32, device="cuda"),
        "state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
        "host_has_initial_state": torch.ones(batch_size, dtype=torch.int32),
    }
    for attention_idx, layer_idx in enumerate(attention_layer_ids):
        candidates[f"past_key_value_{attention_idx}"] = previous_outputs[
            f"present_key_value_{layer_idx}"
        ]
    for layer_idx in recurrent_layer_ids:
        candidates[f"past_conv_state_{layer_idx}"] = previous_outputs[
            f"present_conv_state_{layer_idx}"
        ]
        candidates[f"past_recurrent_state_{layer_idx}"] = previous_outputs[
            f"present_recurrent_state_{layer_idx}"
        ]

    input_names = _engine_input_names(session)
    missing_inputs = input_names - candidates.keys()
    assert not missing_inputs, f"Missing Qwen3.5 generation inputs: {sorted(missing_inputs)}"
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
    qwen35_references: _Qwen35ReferenceData,
    qwen35_session: tuple[Session, Qwen35Config, torch.Tensor],
) -> None:
    padded_input_ids, _ = _make_input_ids(request_lengths)
    reference = qwen35_references.prefill_logits[request_lengths]
    session, config, mrope_cache = qwen35_session
    inputs = _prefill_inputs(session, config, padded_input_ids, request_lengths, mrope_cache)

    outputs = run_session(session, inputs)
    actual = outputs["logits"].float().cpu()

    assert actual.shape == reference.shape
    torch.testing.assert_close(actual, reference, atol=0.35, rtol=0.05)
    actual_top_tokens = actual.argmax(dim=-1, keepdim=True)
    reference_top_tokens = reference.argmax(dim=-1, keepdim=True)
    actual_reference_token_gap = actual.max(dim=-1).values - actual.gather(
        dim=-1, index=reference_top_tokens
    ).squeeze(-1)
    reference_actual_token_gap = reference.max(dim=-1).values - reference.gather(
        dim=-1, index=actual_top_tokens
    ).squeeze(-1)
    assert torch.all(actual_reference_token_gap <= _MAX_PREFILL_TOP_LOGIT_GAP)
    assert torch.all(reference_actual_token_gap <= _MAX_PREFILL_TOP_LOGIT_GAP)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen3.5 engine test requires CUDA")
def test_qwen35_session_prefill_decode_matches_hugging_face(
    qwen35_references: _Qwen35ReferenceData,
    qwen35_session: tuple[Session, Qwen35Config, torch.Tensor],
) -> None:
    request_lengths = (_DECODE_INPUT_LENGTH,)
    input_ids, _ = _make_input_ids(request_lengths)
    session, config, mrope_cache = qwen35_session
    inputs = _prefill_inputs(session, config, input_ids, request_lengths, mrope_cache)
    previous_outputs = run_session(session, inputs)

    for step, input_token in enumerate(qwen35_references.greedy_input_tokens, start=1):
        generation_inputs = _generation_inputs(
            session,
            config,
            torch.tensor([input_token], dtype=torch.int32, device="cuda"),
            request_lengths,
            step,
            mrope_cache,
            previous_outputs,
        )
        previous_outputs = run_session(session, generation_inputs)
        actual = previous_outputs["logits"].float().cpu()
        reference = qwen35_references.greedy_logits[step]
        absolute_error = (actual - reference).abs()

        assert absolute_error.mean().item() < 0.75
        assert torch.quantile(absolute_error, 0.99).item() < 1.75
        assert absolute_error.max().item() < 3.0
        assert torch.equal(actual.argmax(dim=-1), reference.argmax(dim=-1))
