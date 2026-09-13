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

from examples.models.core.qwen3_5.target_verification import Qwen35VerificationSession
from tensorrt_llm import Builder
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.mapping import Mapping
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


def _build_qwen35_session(
    model_dir: Path,
    *,
    max_draft_len: int = 0,
    max_input_len: int = _MAX_INPUT_LENGTH,
    max_num_tokens: int = _MAX_NUM_TOKENS,
    paged_state: bool = False,
) -> tuple[Session, Qwen35Config]:
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
    network.plugin_config.paged_state = paged_state

    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        inputs = model.prepare_inputs(
            max_batch_size=_MAX_BATCH_SIZE,
            max_input_len=max_input_len,
            max_seq_len=_MAX_SEQUENCE_LENGTH,
            max_num_tokens=max_num_tokens,
            opt_batch_size=_OPT_BATCH_SIZE,
            opt_num_tokens=_OPT_NUM_TOKENS,
            use_cache=True,
            max_draft_len=max_draft_len,
            speculative_decoding_draft_tokens_external=max_draft_len > 0,
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
        "source_state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
        "target_state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
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

    if "spec_decoding_use" in _engine_input_names(session):
        candidates.update(_verification_attention_inputs(batch_size, 1))
    input_names = _engine_input_names(session)
    required_qwen35_inputs = {"mrope_rotary_cos_sin", "mrope_position_deltas"}
    if recurrent_layer_ids:
        required_qwen35_inputs.update(
            {
                "gated_delta_cu_seqlens",
                "source_state_slot_mapping",
                "target_state_slot_mapping",
                "host_has_initial_state",
            }
        )
    assert required_qwen35_inputs <= input_names
    missing_inputs = input_names - candidates.keys()
    assert not missing_inputs, f"Missing Qwen3.5 engine inputs: {sorted(missing_inputs)}"
    return {name: candidates[name] for name in input_names}


def _verification_attention_inputs(batch_size: int, num_tokens: int) -> dict[str, torch.Tensor]:
    return {
        "spec_decoding_use": torch.tensor([int(num_tokens > 1)], dtype=torch.int32),
        "spec_decoding_generation_lengths": torch.full(
            (batch_size,), num_tokens, dtype=torch.int32, device="cuda"
        ),
        "spec_decoding_position_offsets": torch.arange(num_tokens, dtype=torch.int32, device="cuda")
        .expand(batch_size, -1)
        .contiguous(),
        "spec_decoding_packed_mask": torch.tensor(
            [(1 << (idx + 1)) - 1 for idx in range(num_tokens)], dtype=torch.int32, device="cuda"
        )
        .repeat(batch_size)
        .view(-1, 1),
    }


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
        "source_state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
        "target_state_slot_mapping": torch.arange(batch_size, dtype=torch.int32, device="cuda"),
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

    if "spec_decoding_use" in _engine_input_names(session):
        candidates.update(_verification_attention_inputs(batch_size, 1))
    input_names = _engine_input_names(session)
    missing_inputs = input_names - candidates.keys()
    assert not missing_inputs, f"Missing Qwen3.5 generation inputs: {sorted(missing_inputs)}"
    return {name: candidates[name] for name in input_names}


def test_qwen35_is_registered() -> None:
    assert MODEL_MAP["Qwen35ForCausalLM"] is Qwen35ForCausalLM


def test_qwen35_tp2_model_uses_local_attention_and_gdn_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = Qwen35Config(
        architecture="Qwen35ForCausalLM",
        dtype="bfloat16",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_size=64,
        vocab_size=32_000,
        max_position_embeddings=1024,
        rotary_embedding_dim=64,
        mrope_section=[11, 11, 10],
        decoder_layer_types=["linear_attention", "full_attention"],
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        mapping=Mapping(world_size=2, rank=0, tp_size=2),
    )
    model = Qwen35ForCausalLM(config)

    linear_attention = model.transformer.layers[0].linear_attn
    full_attention = model.transformer.layers[1].attention

    assert full_attention.qkv.num_attention_heads == 2
    assert full_attention.qkv.num_key_value_heads == 1
    assert full_attention.qkv.query.out_features == 128
    assert full_attention.qkv.gate.out_features == 128
    assert full_attention.qkv.key.out_features == 64
    assert full_attention.qkv.value.out_features == 64
    assert full_attention.dense.proj.in_features == 128

    assert linear_attention.global_num_q_heads == 4
    assert linear_attention.global_num_v_heads == 4
    assert linear_attention.num_q_heads == 2
    assert linear_attention.num_v_heads == 2
    assert linear_attention.global_key_dim == 128
    assert linear_attention.global_value_dim == 128
    assert linear_attention.global_conv_dim == 384
    assert linear_attention.key_dim == 64
    assert linear_attention.value_dim == 64
    assert linear_attention.conv_dim == 192
    assert linear_attention.in_proj_qkv.out_features == 192
    assert linear_attention.in_proj_z.out_features == 64
    assert linear_attention.in_proj_a.out_features == 2
    assert linear_attention.in_proj_b.out_features == 2
    assert linear_attention.conv1d.d_inner == 192
    assert linear_attention.conv1d.weight.shape == (192, 1, 4, 1)
    assert linear_attention.dt_bias.shape == (2,)
    assert linear_attention.A_log.shape == (2,)
    assert linear_attention.gated_delta_rule.num_q_heads == 2
    assert linear_attention.gated_delta_rule.num_v_heads == 2
    assert linear_attention.out_proj.in_features == 64

    recurrent_state_bytes = 2 * 32 * 32 * 4
    conv_state_bytes = 3 * 192 * 2
    assert linear_attention.state_slot_stride_bytes == (recurrent_state_bytes + conv_state_bytes)

    class _FakePluginConfig:
        paged_state = False

    class _FakeNetwork:
        plugin_config = _FakePluginConfig()

    class _FakeTensor:
        def __init__(self, **kwargs) -> None:
            self.name = kwargs["name"]
            self.shape = kwargs["shape"]
            self.dim_range = kwargs["dim_range"]

    monkeypatch.setattr("tensorrt_llm.models.qwen35.model.default_net", _FakeNetwork)
    monkeypatch.setattr("tensorrt_llm.models.qwen35.model.Tensor", _FakeTensor)
    recurrent_inputs = model._prepare_recurrent_inputs(
        num_profiles=1,
        batch_range=[[1, 2, 4]],
    )

    conv_state = recurrent_inputs["conv_states"][0]
    assert conv_state.shape == [-1, 3, -1]
    assert conv_state.dim_range["batch_size"] == [[1, 2, 4]]
    assert conv_state.dim_range["kernel_size"] == [3]
    assert conv_state.dim_range["conv_dim"] == [192]
    recurrent_state = recurrent_inputs["recurrent_states"][0]
    assert recurrent_state.shape == [-1, -1, -1, -1]
    assert recurrent_state.dim_range["state_slots"] == [[1, 2, 4]]
    assert recurrent_state.dim_range["value_heads"] == [2]
    assert recurrent_state.dim_range["value_head_dim"] == [32]
    assert recurrent_state.dim_range["key_head_dim"] == [32]
    assert recurrent_inputs["source_state_slot_mapping"].name == "source_state_slot_mapping"
    assert recurrent_inputs["target_state_slot_mapping"].name == "target_state_slot_mapping"


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


@pytest.fixture(scope="module")
def qwen35_verification_session(qwen35_checkpoint_dir: Path):
    session, config = _build_qwen35_session(
        qwen35_checkpoint_dir, max_draft_len=1, max_input_len=128, max_num_tokens=512
    )
    return session, config, _mrope_cache(config)


@pytest.mark.parametrize("lengths", [(17,), (63,), (17, 31)])
@pytest.mark.parametrize("matching_draft", [True, False])
def test_qwen35_external_draft_verification(
    qwen35_verification_session: tuple[Session, Qwen35Config, torch.Tensor],
    lengths: tuple[int, ...],
    matching_draft: bool,
) -> None:
    """Two-token verification matches sequential decode, including tentative state.

    The caller retains the prefix state. Acceptance and cache promotion are
    intentionally outside this engine-level test.
    """
    session, config, mrope_cache = qwen35_verification_session
    batch_size = len(lengths)
    input_ids, _ = _make_input_ids(lengths)
    prefix = run_session(session, _prefill_inputs(session, config, input_ids, lengths, mrope_cache))
    token = prefix["logits"].argmax(dim=-1).to(torch.int32)
    first = run_session(
        session, _generation_inputs(session, config, token, lengths, 1, mrope_cache, prefix)
    )
    draft = first["logits"].argmax(dim=-1).to(torch.int32)
    if not matching_draft:
        draft = (draft + 1) % config.vocab_size
    second = run_session(
        session, _generation_inputs(session, config, draft, lengths, 2, mrope_cache, first)
    )

    verification_inputs = _generation_inputs(
        session, config, token, lengths, 1, mrope_cache, prefix
    )
    verification_inputs.update(_verification_attention_inputs(batch_size, 2))
    verification_inputs.update(
        {
            "input_ids": torch.stack([token, draft], dim=1).flatten(),
            "last_token_ids": torch.arange(1, 2 * batch_size + 1, dtype=torch.int32, device="cuda"),
            "sequence_length": torch.tensor(lengths, dtype=torch.int32, device="cuda") + 2,
            "gated_delta_cu_seqlens": torch.arange(
                0, 2 * batch_size + 1, 2, dtype=torch.int32, device="cuda"
            ),
        }
    )
    verified = run_session(session, verification_inputs)
    expected_logits = torch.stack([first["logits"], second["logits"]], dim=1).flatten(0, 1)
    assert verified["logits"].shape == (2 * batch_size, config.vocab_size)
    torch.testing.assert_close(verified["logits"], expected_logits, atol=0.15, rtol=0.02)
    assert torch.equal(verified["logits"].argmax(dim=-1), expected_logits.argmax(dim=-1))
    for name in second:
        if name != "logits":
            # BF16 projections and chunk/decode reductions use different
            # accumulation orders; KV differences can slightly exceed 0.02.
            atol = 0.05 if name.startswith("present_key_value_") else 0.02
            torch.testing.assert_close(verified[name], second[name], atol=atol, rtol=0.02, msg=name)

    # A subsequent ordinary decode must also consume the tentative state correctly.
    next_token = second["logits"].argmax(dim=-1).to(torch.int32)
    continuations = [
        run_session(
            session, _generation_inputs(session, config, next_token, lengths, 3, mrope_cache, state)
        )["logits"]
        for state in (second, verified)
    ]
    torch.testing.assert_close(*continuations, atol=0.15, rtol=0.02)


@pytest.mark.parametrize(
    "draft_length,external,tp_size,use_cache,variable_length,error",
    [
        (0, True, 1, True, False, "one external draft token"),
        (1, False, 1, True, False, "one external draft token"),
        (2, True, 1, True, False, "one external draft token"),
        (1, True, 2, True, False, "TP=1"),
        (1, True, 1, False, False, "use_cache=true"),
        (0, False, 1, True, True, "Variable generation lengths require K=1"),
    ],
)
def test_qwen35_verification_rejects_unsupported_settings(
    draft_length: int,
    external: bool,
    tp_size: int,
    use_cache: bool,
    variable_length: bool,
    error: str,
) -> None:
    config = Qwen35Config(
        architecture="Qwen35ForCausalLM",
        dtype="bfloat16",
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        rotary_embedding_dim=64,
        mrope_section=[11, 11, 10],
        decoder_layer_types=["linear_attention", "full_attention"],
        mapping=Mapping(world_size=tp_size, rank=0, tp_size=tp_size),
    )
    model = Qwen35ForCausalLM(config)
    with pytest.raises(ValueError, match=error):
        model.prepare_inputs(
            max_batch_size=1,
            max_input_len=16,
            max_seq_len=32,
            max_num_tokens=16,
            use_cache=use_cache,
            max_draft_len=draft_length,
            speculative_decoding_draft_tokens_external=external,
            spec_decoding_is_generation_length_variable=variable_length,
        )


@pytest.fixture(scope="module")
def qwen35_snapshot_verification_session(qwen35_checkpoint_dir: Path):
    return _build_qwen35_session(
        qwen35_checkpoint_dir,
        max_draft_len=1,
        max_input_len=128,
        max_num_tokens=512,
        paged_state=True,
    )


@pytest.mark.parametrize("acceptance", ["accept", "reject", "alternate", "mixed"])
def test_qwen35_verification_commits_accepted_prefix(
    qwen35_snapshot_verification_session: tuple[Session, Qwen35Config],
    qwen35_verification_session: tuple[Session, Qwen35Config, torch.Tensor],
    acceptance: str,
) -> None:
    """Full snapshots must sustain repeated acceptance/rejection and slot reuse."""
    session, config = qwen35_snapshot_verification_session
    baseline_session, baseline_config, mrope_cache = qwen35_verification_session
    lengths = (61, 63)
    batch = len(lengths)
    runner = Qwen35VerificationSession(session, config, batch, _MAX_SEQUENCE_LENGTH)
    padded_ids, _ = _make_input_ids(lengths)
    prompts = [padded_ids[idx, :length].tolist() for idx, length in enumerate(lengths)]
    pending = runner.prefill(prompts)
    baseline = run_session(
        baseline_session,
        _prefill_inputs(baseline_session, baseline_config, padded_ids, lengths, mrope_cache),
    )
    assert torch.equal(pending, baseline["logits"].argmax(-1).int())
    processed = torch.tensor(lengths, dtype=torch.int32)

    for iteration in range(8):
        pending = baseline["logits"].argmax(-1).int()
        first = run_session(
            baseline_session,
            _generation_inputs(
                baseline_session,
                baseline_config,
                pending,
                tuple(processed.tolist()),
                1,
                mrope_cache,
                baseline,
            ),
        )
        want_accept = torch.tensor(
            [
                acceptance == "accept"
                or (acceptance == "alternate" and iteration % 2 == 0)
                or (acceptance == "mixed" and (iteration + idx) % 2 == 0)
                for idx in range(batch)
            ],
            device="cuda",
        )
        draft = (first["logits"].argmax(-1).int() + (~want_accept).int()) % config.vocab_size
        second = run_session(
            baseline_session,
            _generation_inputs(
                baseline_session,
                baseline_config,
                draft,
                tuple((processed + 1).tolist()),
                1,
                mrope_cache,
                first,
            ),
        )
        old_slots = runner._state.slots.clone()
        old_records = {
            idx: records.index_select(0, old_slots.cuda().long()).clone()
            for idx, records in runner._records.items()
        }
        result = runner.step(draft)
        assert torch.equal(result.accepted_draft, want_accept)
        expected_tokens = torch.stack(
            [
                first["logits"].argmax(-1).int(),
                torch.where(want_accept, second["logits"].argmax(-1).int(), -1),
            ],
            dim=1,
        )
        assert torch.equal(result.tokens, expected_tokens)
        for idx, records in runner._records.items():
            torch.testing.assert_close(
                records.index_select(0, old_slots.cuda().long()), old_records[idx], atol=0, rtol=0
            )
        processed += 1 + want_accept.cpu().int()
        assert torch.equal(runner.past_lengths, processed)
        baseline = {
            name: torch.where(
                want_accept.reshape((batch,) + (1,) * (value.ndim - 1)), second[name], value
            )
            for name, value in first.items()
        }
        assert torch.equal(runner.current_tokens, baseline["logits"].argmax(-1).int())
        for name, value in runner.recurrent_states().items():
            torch.testing.assert_close(value, baseline[name], atol=0.03, rtol=0.03, msg=name)
        for local_idx, layer_idx in enumerate(runner._attention_ids):
            kv = runner._state.kv[f"past_key_value_{local_idx}"]
            for idx, length in enumerate(processed.tolist()):
                torch.testing.assert_close(
                    kv[idx, :, :, :length],
                    baseline[f"present_key_value_{layer_idx}"][idx, :, :, :length],
                    atol=0.08,
                    rtol=0.03,
                )
                # Stale/rejected rows must remain invisible to subsequent calls.
                kv[idx, :, :, length:] = 1000

    # Reuse the same state slots for fresh requests after an explicit reset.
    runner.reset()
    with pytest.raises(RuntimeError, match="prefill"):
        runner.decode()
    fresh_baseline = run_session(
        baseline_session,
        _prefill_inputs(baseline_session, baseline_config, padded_ids, lengths, mrope_cache),
    )
    assert torch.equal(runner.prefill(prompts), fresh_baseline["logits"].argmax(-1).int())


def test_qwen35_verification_failed_step_preserves_committed_state(
    qwen35_snapshot_verification_session: tuple[Session, Qwen35Config],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, config = qwen35_snapshot_verification_session
    runner = Qwen35VerificationSession(session, config, 1, _MAX_SEQUENCE_LENGTH)
    runner.prefill([[1, 2, 3]])
    state = runner._state
    snapshots = runner.recurrent_states()
    original_run = session.run

    def fail_after_enqueue(*args, **kwargs):
        assert original_run(*args, **kwargs)
        torch.cuda.synchronize()
        return False

    with monkeypatch.context() as patch:
        patch.setattr(session, "run", fail_after_enqueue)
        with pytest.raises(RuntimeError, match="not committed"):
            runner.step(torch.tensor([0], dtype=torch.int32))
    assert runner._state is state
    for name, value in runner.recurrent_states().items():
        torch.testing.assert_close(value, snapshots[name], atol=0, rtol=0)
    runner.step(torch.tensor([0], dtype=torch.int32))
    assert runner.past_lengths.item() in (4, 5)


def test_qwen35_verification_limits_do_not_commit(
    qwen35_snapshot_verification_session: tuple[Session, Qwen35Config],
) -> None:
    session, config = qwen35_snapshot_verification_session
    runner = Qwen35VerificationSession(session, config, 1, max_seq_len=4)
    runner.prefill([[1, 2, 3]])
    state = runner._state
    for invalid in (
        torch.tensor([-1]),
        torch.tensor([config.vocab_size]),
        torch.tensor([2**32 + 1]),
    ):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            runner.step(invalid)
        assert runner._state is state
    with pytest.raises(ValueError, match="integer draft token"):
        runner.step(torch.tensor([1.0]))
    with pytest.raises(ValueError, match="single-token decode"):
        runner.step(torch.tensor([0]))
    assert runner._state is state
    runner.decode()
    assert runner.past_lengths.item() == 4
    state = runner._state
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        runner.decode()
    with pytest.raises(RuntimeError, match="reset"):
        runner.prefill([[1]])
    assert runner._state is state
