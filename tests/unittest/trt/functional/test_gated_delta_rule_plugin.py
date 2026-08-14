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

import numpy as np
import pytest
import tensorrt as trt
import torch

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm.plugin import TRT_LLM_PLUGIN_NAMESPACE

HEAD_K_DIM = 128
HEAD_V_DIM = 128
CHUNK_SIZE = 64
BATCH_SIZES = (1, 2, 4, 8)
QWEN3_5_CONFIGS = (
    pytest.param(16, 16, id="qwen3.5-2b"),
    pytest.param(16, 32, id="qwen3.5-4b"),
    pytest.param(16, 32, id="qwen3.5-9b"),
    pytest.param(16, 48, id="qwen3.5-27b"),
)


def _make_plugin_field(
    name: str,
    value: int,
    dtype: type[np.integer],
    field_type: trt.PluginFieldType,
) -> trt.PluginField:
    return trt.PluginField(name, np.array([value], dtype=dtype), field_type)


def _add_gated_delta_rule_plugin(
    inputs: list[Tensor],
    num_q_heads: int,
    num_v_heads: int,
) -> trt.IPluginV3Layer:
    creator = trt.get_plugin_registry().get_creator("GatedDeltaRule", "1", TRT_LLM_PLUGIN_NAMESPACE)
    assert creator is not None

    fields = trt.PluginFieldCollection(
        [
            _make_plugin_field("num_q_heads", num_q_heads, np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field("num_v_heads", num_v_heads, np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field("head_k_dim", HEAD_K_DIM, np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field("head_v_dim", HEAD_V_DIM, np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field("chunk_size", CHUNK_SIZE, np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field("type_id", int(trt.bfloat16), np.int32, trt.PluginFieldType.INT32),
            _make_plugin_field(
                "state_type_id", int(trt.float32), np.int32, trt.PluginFieldType.INT32
            ),
            _make_plugin_field("remove_input_padding", 0, np.int8, trt.PluginFieldType.INT8),
            _make_plugin_field("paged_state", 0, np.int8, trt.PluginFieldType.INT8),
            _make_plugin_field("use_qk_l2norm", 1, np.int8, trt.PluginFieldType.INT8),
        ]
    )
    plugin = creator.create_plugin("gated_delta_rule", fields, trt.TensorRTPhase.BUILD)
    assert plugin is not None

    layer = tensorrt_llm.default_trtnet().add_plugin_v3(
        [tensor.trt_tensor for tensor in inputs], [], plugin
    )
    assert layer is not None
    return layer


def _gated_delta_rule_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation_dtype = value.dtype
    query = query.squeeze(1).float()
    key = key.squeeze(1).float()
    value = value.squeeze(1).float()
    log_decay = log_decay.squeeze(1).float()
    beta = beta.squeeze(1).float()

    query = query / (torch.linalg.vector_norm(query, dim=-1, keepdim=True) + 1e-6)
    key = key / (torch.linalg.vector_norm(key, dim=-1, keepdim=True) + 1e-6)

    heads_ratio = value.shape[1] // query.shape[1]
    query = query.repeat_interleave(heads_ratio, dim=1)
    key = key.repeat_interleave(heads_ratio, dim=1)

    initial_state_mask = has_initial_state[:, None, None, None].bool().to(state.device)
    state = torch.where(initial_state_mask, state.float(), torch.zeros_like(state))
    state = state * torch.exp(log_decay[..., None, None])
    value_residual = value - torch.einsum("bhvk,bhk->bhv", state, key)
    value_residual = value_residual * beta[..., None]
    state = state + torch.einsum("bhv,bhk->bhvk", value_residual, key)

    scale = HEAD_K_DIM**-0.5
    output = torch.einsum("bhvk,bhk->bhv", state, query * scale)
    return output[:, None].to(activation_dtype), state


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("num_q_heads,num_v_heads", QWEN3_5_CONFIGS)
def test_gated_delta_rule_decode(
    num_q_heads: int,
    num_v_heads: int,
    batch_size: int,
) -> None:
    torch.manual_seed(1234)
    device = "cuda"

    query = torch.randn(batch_size, 1, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(batch_size, 1, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16)
    log_decay = -0.1 * torch.rand(batch_size, 1, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(batch_size, 1, num_v_heads, device=device))
    state = 0.05 * torch.randn(
        batch_size,
        num_v_heads,
        HEAD_V_DIM,
        HEAD_K_DIM,
        device=device,
        dtype=torch.float32,
    )
    initial_state = state.clone()

    host_request_types = torch.ones(batch_size, dtype=torch.int32)
    cu_seqlens = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    state_slot_mapping = torch.arange(batch_size, device=device, dtype=torch.int32)
    host_has_initial_state = torch.ones(batch_size, dtype=torch.int8)

    builder = tensorrt_llm.Builder()
    network = builder.create_network()
    with tensorrt_llm.net_guard(network):
        query_tensor = Tensor("query", trt.bfloat16, tuple(query.shape))
        key_tensor = Tensor("key", trt.bfloat16, tuple(key.shape))
        value_tensor = Tensor("value", trt.bfloat16, tuple(value.shape))
        log_decay_tensor = Tensor("log_decay", trt.float32, tuple(log_decay.shape))
        beta_tensor = Tensor("beta", trt.float32, tuple(beta.shape))
        state_tensor = Tensor("state", trt.float32, tuple(state.shape))
        host_request_types_tensor = Tensor(
            "host_request_types",
            trt.int32,
            tuple(host_request_types.shape),
            location=trt.TensorLocation.HOST,
        )
        cu_seqlens_tensor = Tensor("cu_seqlens", trt.int32, tuple(cu_seqlens.shape))
        state_slot_mapping_tensor = Tensor(
            "state_slot_mapping", trt.int32, tuple(state_slot_mapping.shape)
        )
        host_has_initial_state_tensor = Tensor(
            "host_has_initial_state",
            trt.int8,
            tuple(host_has_initial_state.shape),
            location=trt.TensorLocation.HOST,
        )

        layer = _add_gated_delta_rule_plugin(
            [
                query_tensor,
                key_tensor,
                value_tensor,
                log_decay_tensor,
                beta_tensor,
                state_tensor,
                host_request_types_tensor,
                cu_seqlens_tensor,
                state_slot_mapping_tensor,
                host_has_initial_state_tensor,
            ],
            num_q_heads,
            num_v_heads,
        )
        output_tensor = layer.get_output(0)
        output_tensor.name = "output"
        network.trt_network.mark_output(output_tensor)
        final_state_tensor = layer.get_output(1)
        final_state_tensor.name = "final_state"
        network.trt_network.mark_output(final_state_tensor)

    builder_config = builder.create_builder_config(precision="bfloat16")
    engine = builder.build_engine(network, builder_config)
    assert engine is not None
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)

    output = torch.empty_like(value)
    stream = torch.cuda.current_stream()
    ok = session.run(
        inputs={
            "query": query,
            "key": key,
            "value": value,
            "log_decay": log_decay,
            "beta": beta,
            "state": state,
            "host_request_types": host_request_types,
            "cu_seqlens": cu_seqlens,
            "state_slot_mapping": state_slot_mapping,
            "host_has_initial_state": host_has_initial_state,
        },
        outputs={"output": output, "final_state": state},
        stream=stream.cuda_stream,
    )
    assert ok
    stream.synchronize()

    output_ref, final_state_ref = _gated_delta_rule_decode_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        host_has_initial_state,
    )
    torch.testing.assert_close(output.float(), output_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state, final_state_ref, atol=2e-3, rtol=2e-3)
