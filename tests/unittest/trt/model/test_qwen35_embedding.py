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

from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model as HuggingFaceQwen35Model
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding as HuggingFaceQwen35TextRotaryEmbedding,
)

from tensorrt_llm import Builder
from tensorrt_llm.layers import PromptTuningEmbedding
from tensorrt_llm.models.modeling_utils import set_prompt_tuning
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM, Qwen35Model
from tensorrt_llm.models.qwen35.vision_utils import (
    prepare_qwen35_executor_prompt_inputs,
    prepare_qwen35_mrope_inputs,
    prepare_qwen35_multimodal_cache_input,
    prepare_qwen35_prompt_tuning_inputs,
)
from tensorrt_llm.module import Module, ModuleList
from tensorrt_llm.network import net_guard


def _small_qwen35_config(*, with_vision: bool = False) -> Qwen35Config:
    vision_kwargs = {}
    if with_vision:
        vision_kwargs = {
            "vision_depth": 1,
            "vision_hidden_size": 16,
            "vision_intermediate_size": 32,
            "vision_num_heads": 4,
            "vision_in_channels": 3,
            "vision_patch_size": 2,
            "vision_temporal_patch_size": 2,
            "vision_spatial_merge_size": 2,
            "vision_num_position_embeddings": 16,
            "vision_output_hidden_size": 64,
            "vision_hidden_act": "gelu_pytorch_tanh",
        }
    return Qwen35Config(
        architecture="Qwen35ForCausalLM",
        dtype="bfloat16",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_size=16,
        vocab_size=128,
        max_position_embeddings=128,
        position_embedding_type="mrope",
        hidden_act="silu",
        norm_epsilon=1e-6,
        tie_word_embeddings=True,
        rotary_embedding_dim=16,
        mrope_section=[3, 3, 2],
        decoder_layer_types=["full_attention"],
        image_token_id=120,
        video_token_id=121,
        vision_start_token_id=119,
        **vision_kwargs,
    )


class _RecordingEmbedding(Module):
    def __init__(self) -> None:
        super().__init__()
        self.args: tuple[object, ...] | None = None

    def forward(self, *args: object) -> object:
        self.args = args
        return args[0]

    __call__ = forward


class _Identity(Module):
    def forward(self, value: object) -> object:
        return value

    __call__ = forward


class _EmptyKvCacheParams:
    def fill_none_tensor_list(self, num_layers: int) -> None:
        assert num_layers == 0


def _embedding_only_model() -> tuple[Qwen35Model, _RecordingEmbedding]:
    model = Qwen35Model(_small_qwen35_config())
    embedding = _RecordingEmbedding()
    model.vocab_embedding = embedding
    model.layers = ModuleList([])
    model.ln_f = _Identity()
    return model, embedding


def test_qwen35_gather_context_logits_keeps_recurrent_last_token_ids() -> None:
    model = Qwen35ForCausalLM(_small_qwen35_config())
    network = Builder().create_network()
    network.plugin_config.gpt_attention_plugin = "bfloat16"
    network.plugin_config.gemm_plugin = "bfloat16"
    network.plugin_config.mamba_conv1d_plugin = "bfloat16"
    network.plugin_config.remove_input_padding = True

    with net_guard(network):
        inputs = model.prepare_inputs(
            max_batch_size=2,
            max_input_len=16,
            max_seq_len=24,
            max_num_tokens=32,
            use_cache=True,
            prompt_embedding_table_size=8,
            gather_context_logits=True,
        )

    assert inputs["last_token_ids"] is not None
    assert inputs["last_token_ids"].name == "last_token_ids"


def _run_embedding_only_model(
    model: Qwen35Model,
    input_ids: object,
    prompt_embedding_table: object | None = None,
    prompt_tasks: object | None = None,
    prompt_vocab_size: object | None = None,
) -> object:
    hidden_states, present_kvs, present_convs, present_recurrent_states = model.forward(
        input_ids=input_ids,
        use_cache=True,
        attention_mask=None,
        kv_cache_params=_EmptyKvCacheParams(),
        attention_params=None,
        mrope_params=None,
        conv_states=[],
        recurrent_states=[],
        host_request_types=None,
        last_token_ids=None,
        host_context_lengths=None,
        cu_seqlens=None,
        source_state_slot_mapping=None,
        target_state_slot_mapping=None,
        host_has_initial_state=None,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=prompt_tasks,
        prompt_vocab_size=prompt_vocab_size,
    )
    assert present_kvs == []
    assert present_convs == []
    assert present_recurrent_states == []
    return hidden_states


def test_qwen35_prompt_tuning_replaces_vocab_embedding() -> None:
    model = Qwen35ForCausalLM(_small_qwen35_config())

    set_prompt_tuning(model)

    assert isinstance(model.transformer.vocab_embedding, PromptTuningEmbedding)


def test_qwen35_prefill_passes_visual_prompt_inputs_to_embedding() -> None:
    model, embedding = _embedding_only_model()
    input_ids = object()
    prompt_embedding_table = object()
    prompt_tasks = object()
    prompt_vocab_size = object()

    hidden_states = _run_embedding_only_model(
        model,
        input_ids,
        prompt_embedding_table,
        prompt_tasks,
        prompt_vocab_size,
    )

    assert hidden_states is input_ids
    assert embedding.args == (
        input_ids,
        prompt_embedding_table,
        prompt_tasks,
        prompt_vocab_size,
    )


def test_qwen35_without_prompt_tuning_uses_plain_embedding() -> None:
    model, embedding = _embedding_only_model()
    input_ids = object()

    hidden_states = _run_embedding_only_model(model, input_ids)

    assert hidden_states is input_ids
    assert embedding.args == (input_ids,)


def test_qwen35_prepare_inputs_exposes_prompt_tuning_bindings() -> None:
    model = Qwen35ForCausalLM(_small_qwen35_config())
    network = Builder().create_network()
    network.plugin_config.to_legacy_setting()
    network.plugin_config.gpt_attention_plugin = "bfloat16"
    network.plugin_config.gemm_plugin = "bfloat16"
    network.plugin_config.mamba_conv1d_plugin = "bfloat16"
    network.plugin_config.remove_input_padding = True
    network.plugin_config.paged_kv_cache = True

    with net_guard(network):
        inputs = model.prepare_inputs(
            max_batch_size=2,
            max_input_len=16,
            max_seq_len=24,
            max_num_tokens=32,
            use_cache=True,
            opt_batch_size=1,
            opt_num_tokens=16,
            prompt_embedding_table_size=8,
        )

    assert inputs["prompt_embedding_table"].name == "prompt_embedding_table"
    assert inputs["prompt_tasks"].name == "tasks"
    assert inputs["prompt_vocab_size"].name == "prompt_vocab_size"


def test_prepare_qwen35_prompt_tuning_inputs_maps_image_and_video_features() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [1, config.image_token_id, config.image_token_id, 2, config.video_token_id],
            [config.video_token_id, 3, config.image_token_id, 4, 5],
        ],
        dtype=torch.int64,
    )
    image_features = torch.arange(3 * config.hidden_size, dtype=torch.float32).reshape(
        3, config.hidden_size
    )
    video_features = (
        torch.arange(2 * config.hidden_size, dtype=torch.float32).reshape(2, config.hidden_size)
        + 1_000
    )

    prompt_inputs = prepare_qwen35_prompt_tuning_inputs(
        input_ids,
        config,
        image_features=image_features,
        video_features=video_features,
    )

    expected_input_ids = torch.tensor(
        [
            [1, 128, 129, 2, 131],
            [132, 3, 130, 4, 5],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(prompt_inputs.input_ids, expected_input_ids)
    expected_table = torch.cat((image_features, video_features)).to(torch.bfloat16)
    torch.testing.assert_close(prompt_inputs.prompt_embedding_table, expected_table)
    torch.testing.assert_close(
        prompt_inputs.prompt_tasks,
        torch.zeros_like(expected_input_ids),
    )
    torch.testing.assert_close(
        prompt_inputs.prompt_vocab_size,
        torch.tensor([5], dtype=torch.int32),
    )

    visual_mask = (input_ids == config.image_token_id) | (input_ids == config.video_token_id)
    table_indices = prompt_inputs.input_ids[visual_mask] - config.vocab_size
    selected_features = prompt_inputs.prompt_embedding_table[table_indices]
    expected_sequence = torch.stack(
        (
            expected_table[0],
            expected_table[1],
            expected_table[3],
            expected_table[4],
            expected_table[2],
        )
    )
    torch.testing.assert_close(selected_features, expected_sequence)


def test_prepare_qwen35_prompt_tuning_inputs_rejects_feature_count_mismatch() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor([config.image_token_id, config.image_token_id])
    image_features = torch.zeros((1, config.hidden_size))

    with pytest.raises(ValueError, match="image features and tokens do not match"):
        prepare_qwen35_prompt_tuning_inputs(
            input_ids,
            config,
            image_features=image_features,
        )


def test_prepare_qwen35_prompt_tuning_inputs_builds_text_only_dummy_table() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)

    prompt_inputs = prepare_qwen35_prompt_tuning_inputs(input_ids, config)

    torch.testing.assert_close(prompt_inputs.input_ids, input_ids.to(torch.int32))
    assert prompt_inputs.prompt_embedding_table.shape == (1, config.hidden_size)
    assert prompt_inputs.prompt_embedding_table.dtype == torch.bfloat16
    assert torch.count_nonzero(prompt_inputs.prompt_embedding_table) == 0
    assert torch.count_nonzero(prompt_inputs.prompt_tasks) == 0
    torch.testing.assert_close(
        prompt_inputs.prompt_vocab_size,
        torch.tensor([0], dtype=torch.int32),
    )


def test_prepare_qwen35_executor_prompt_inputs_partitions_requests() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [1, config.image_token_id, config.image_token_id, 2, config.video_token_id, 0],
            [config.video_token_id, 3, config.image_token_id, 4, 5, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 0],
        ]
    )
    image_features = torch.arange(3 * config.hidden_size, dtype=torch.float32).reshape(
        3, config.hidden_size
    )
    video_features = (
        torch.arange(2 * config.hidden_size, dtype=torch.float32).reshape(2, config.hidden_size)
        + 1_000
    )

    executor_inputs = prepare_qwen35_executor_prompt_inputs(
        input_ids,
        attention_mask,
        config,
        image_features=image_features,
        video_features=video_features,
    )

    expected_input_ids = [
        torch.tensor([1, 128, 129, 2, 130], dtype=torch.int32),
        torch.tensor([129, 3, 128, 4, 5], dtype=torch.int32),
    ]
    assert len(executor_inputs.batch_input_ids) == len(expected_input_ids)
    for actual, expected in zip(executor_inputs.batch_input_ids, expected_input_ids):
        torch.testing.assert_close(actual, expected)

    expected_table = torch.zeros((2, 3, config.hidden_size), dtype=torch.bfloat16)
    expected_table[0] = torch.cat((image_features[:2], video_features[:1])).to(torch.bfloat16)
    expected_table[1, :2] = torch.cat((image_features[2:], video_features[1:])).to(torch.bfloat16)
    torch.testing.assert_close(executor_inputs.prompt_table, expected_table)
    assert executor_inputs.prompt_tasks == "0,1"


def test_prepare_qwen35_executor_prompt_inputs_supports_text_only_request() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3],
            [config.video_token_id, 4, config.image_token_id, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0],
        ]
    )
    image_features = torch.full((1, config.hidden_size), 11.0)
    video_features = torch.full((1, config.hidden_size), 22.0)

    executor_inputs = prepare_qwen35_executor_prompt_inputs(
        input_ids,
        attention_mask,
        config,
        image_features=image_features,
        video_features=video_features,
    )

    torch.testing.assert_close(
        executor_inputs.batch_input_ids[0],
        torch.tensor([1, 2, 3], dtype=torch.int32),
    )
    torch.testing.assert_close(
        executor_inputs.batch_input_ids[1],
        torch.tensor([129, 4, 128], dtype=torch.int32),
    )
    assert executor_inputs.prompt_table.shape == (2, 2, config.hidden_size)
    assert torch.count_nonzero(executor_inputs.prompt_table[0]) == 0
    torch.testing.assert_close(
        executor_inputs.prompt_table[1],
        torch.cat((image_features, video_features)).to(torch.bfloat16),
    )


def test_prepare_qwen35_executor_prompt_inputs_rejects_unused_features() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor([[1, config.image_token_id, 2]])
    attention_mask = torch.ones_like(input_ids)
    image_features = torch.zeros((2, config.hidden_size))

    with pytest.raises(ValueError, match="image features and tokens do not match"):
        prepare_qwen35_executor_prompt_inputs(
            input_ids,
            attention_mask,
            config,
            image_features=image_features,
        )


def test_prepare_qwen35_multimodal_cache_input_image_span() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [[0, 7, config.vision_start_token_id, config.image_token_id, config.image_token_id, 8]]
    )
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1]])
    content_hash = [1, 2, 3, 4, 5, 6, 7, 8]

    multimodal_input = prepare_qwen35_multimodal_cache_input(
        input_ids,
        attention_mask,
        config,
        "image",
        content_hash,
    )

    assert multimodal_input.multimodal_hashes == [content_hash]
    assert multimodal_input.multimodal_positions == [1]
    assert multimodal_input.multimodal_lengths == [3]


def test_prepare_qwen35_multimodal_cache_input_duplicates_video_hash() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [
                7,
                config.vision_start_token_id,
                config.video_token_id,
                config.video_token_id,
                8,
                config.vision_start_token_id,
                config.video_token_id,
                9,
            ]
        ]
    )
    attention_mask = torch.ones_like(input_ids)
    content_hash = [8, 7, 6, 5, 4, 3, 2, 1]

    multimodal_input = prepare_qwen35_multimodal_cache_input(
        input_ids,
        attention_mask,
        config,
        "video",
        content_hash,
    )

    assert multimodal_input.multimodal_hashes == [content_hash, content_hash]
    assert multimodal_input.multimodal_positions == [1, 5]
    assert multimodal_input.multimodal_lengths == [3, 2]


def test_prepare_qwen35_multimodal_cache_input_maps_per_span_hashes() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [
                config.vision_start_token_id,
                config.video_token_id,
                7,
                config.vision_start_token_id,
                config.video_token_id,
            ]
        ]
    )
    content_hashes = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1],
    ]

    multimodal_input = prepare_qwen35_multimodal_cache_input(
        input_ids,
        torch.ones_like(input_ids),
        config,
        "video",
        content_hashes=content_hashes,
    )

    assert multimodal_input.multimodal_hashes == content_hashes
    assert multimodal_input.multimodal_positions == [0, 3]
    assert multimodal_input.multimodal_lengths == [2, 2]


def test_prepare_qwen35_multimodal_cache_input_rejects_unmapped_tokens() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor([[7, config.image_token_id, 8]])

    with pytest.raises(ValueError, match="Could not map every"):
        prepare_qwen35_multimodal_cache_input(
            input_ids,
            torch.ones_like(input_ids),
            config,
            "image",
            [1, 2, 3, 4, 5, 6, 7, 8],
        )


class _HuggingFaceMropeReference:
    get_vision_position_ids = HuggingFaceQwen35Model.get_vision_position_ids

    def __init__(self, spatial_merge_size: int) -> None:
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=spatial_merge_size)
        )


def _reference_rotary_cos_sin(
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: Qwen35Config,
) -> torch.Tensor:
    inv_freq = 1.0 / (
        config.rotary_base
        ** (
            torch.arange(0, config.rotary_embedding_dim, 2, dtype=torch.float32)
            / config.rotary_embedding_dim
        )
    )
    result = torch.zeros(
        (
            input_ids_batch_size := position_ids.shape[1],
            config.max_position_embeddings,
            config.rotary_embedding_dim // 2,
            2,
        ),
        dtype=torch.float32,
    )
    result[..., 0] = 1
    for batch_index in range(input_ids_batch_size):
        request_positions = position_ids[:, batch_index, attention_mask[batch_index].bool()]
        frequencies = request_positions.to(torch.float32).unsqueeze(-1) * inv_freq
        frequencies = frequencies.unsqueeze(1)
        interleaved = HuggingFaceQwen35TextRotaryEmbedding.apply_interleaved_mrope(
            None,
            frequencies,
            config.mrope_section,
        ).squeeze(0)
        request_rotary = torch.stack((interleaved.cos(), interleaved.sin()), dim=-1)
        result[batch_index, : request_rotary.shape[0]] = request_rotary
    return result.reshape(input_ids_batch_size, -1)


def test_prepare_qwen35_mrope_inputs_matches_hugging_face() -> None:
    config = _small_qwen35_config(with_vision=True)
    input_ids = torch.tensor(
        [
            [10, 11, 120, 120, 120, 120, 12, 0, 0],
            [0, 0, 20, 121, 21, 121, 22, 0, 0],
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    mm_token_type_ids = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 2, 0, 2, 0, 0, 0],
        ],
        dtype=torch.int32,
    )
    image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    video_grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64)

    actual = prepare_qwen35_mrope_inputs(
        input_ids,
        config,
        attention_mask=attention_mask,
        mm_token_type_ids=mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )

    reference = _HuggingFaceMropeReference(config.vision_spatial_merge_size)
    expected_position_ids, expected_deltas = HuggingFaceQwen35Model.get_rope_index(
        reference,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        attention_mask=attention_mask,
    )
    torch.testing.assert_close(actual.position_ids, expected_position_ids)
    torch.testing.assert_close(
        actual.mrope_position_deltas,
        expected_deltas.to(torch.int32),
    )
    torch.testing.assert_close(
        actual.mrope_rotary_cos_sin,
        _reference_rotary_cos_sin(expected_position_ids, attention_mask, config),
    )


def test_prepare_qwen35_mrope_inputs_supports_text_only_padding() -> None:
    config = _small_qwen35_config()
    input_ids = torch.tensor(
        [
            [0, 0, 10, 11, 12],
            [20, 21, 0, 0, 0],
        ],
        dtype=torch.int64,
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0],
        ]
    )

    actual = prepare_qwen35_mrope_inputs(
        input_ids,
        config,
        attention_mask=attention_mask,
    )

    expected_position_ids = torch.zeros((3, 2, 5), dtype=torch.int64)
    expected_position_ids[:, 0, 2:] = torch.arange(3).view(1, -1)
    expected_position_ids[:, 1, :2] = torch.arange(2).view(1, -1)
    torch.testing.assert_close(actual.position_ids, expected_position_ids)
    torch.testing.assert_close(
        actual.mrope_position_deltas,
        torch.zeros((2, 1), dtype=torch.int32),
    )
    torch.testing.assert_close(
        actual.mrope_rotary_cos_sin,
        _reference_rotary_cos_sin(expected_position_ids, attention_mask, config),
    )


def test_prepare_qwen35_mrope_inputs_rejects_grid_token_mismatch() -> None:
    config = _small_qwen35_config(with_vision=True)
    input_ids = torch.tensor([[10, 120, 120, 120, 11]])
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.tensor([[0, 1, 1, 1, 0]], dtype=torch.int32)

    with pytest.raises(ValueError, match="image token group has length 3, expected 4"):
        prepare_qwen35_mrope_inputs(
            input_ids,
            config,
            attention_mask=attention_mask,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
