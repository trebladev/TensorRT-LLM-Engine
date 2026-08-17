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
from tensorrt_llm._torch.modules.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from tensorrt_llm._torch.modules.fla.chunk_o import chunk_fwd_o
from tensorrt_llm._torch.modules.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from tensorrt_llm._torch.modules.fla.cumsum import chunk_local_cumsum
from tensorrt_llm._torch.modules.fla.l2norm import l2norm_fwd
from tensorrt_llm._torch.modules.fla.solve_tril import solve_tril
from tensorrt_llm._torch.modules.fla.wy_fast import recompute_w_u_fwd
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
SHORT_PREFILL_SEQUENCE_LENGTHS = (17, 64, 81)
LONG_PREFILL_SEQUENCE_LENGTHS = (32, 96, 127, 4000, 8193)


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
    remove_input_padding: bool = False,
    paged_state: bool = False,
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
            _make_plugin_field(
                "remove_input_padding", int(remove_input_padding), np.int8, trt.PluginFieldType.INT8
            ),
            _make_plugin_field(
                "paged_state", int(paged_state), np.int8, trt.PluginFieldType.INT8
            ),
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


def _build_gated_delta_rule_session(
    input_shapes: dict[str, tuple[int, ...]],
    num_q_heads: int,
    num_v_heads: int,
    *,
    paged_state: bool,
    remove_input_padding: bool,
    optimization_profiles: tuple[
        dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]], ...
    ] = (),
) -> tensorrt_llm.runtime.Session:
    builder = tensorrt_llm.Builder()
    network = builder.create_network()
    with tensorrt_llm.net_guard(network):
        query_tensor = Tensor("query", trt.bfloat16, input_shapes["query"])
        key_tensor = Tensor("key", trt.bfloat16, input_shapes["key"])
        value_tensor = Tensor("value", trt.bfloat16, input_shapes["value"])
        log_decay_tensor = Tensor("log_decay", trt.float32, input_shapes["log_decay"])
        beta_tensor = Tensor("beta", trt.float32, input_shapes["beta"])
        state_tensor = Tensor(
            "state",
            trt.int64 if paged_state else trt.float32,
            input_shapes["state"],
            location=trt.TensorLocation.HOST if paged_state else trt.TensorLocation.DEVICE,
        )
        host_request_types_tensor = Tensor(
            "host_request_types",
            trt.int32,
            input_shapes["host_request_types"],
            location=trt.TensorLocation.HOST,
        )
        cu_seqlens_tensor = Tensor("cu_seqlens", trt.int32, input_shapes["cu_seqlens"])
        state_slot_mapping_tensor = Tensor(
            "state_slot_mapping", trt.int32, input_shapes["state_slot_mapping"]
        )
        host_has_initial_state_tensor = Tensor(
            "host_has_initial_state",
            trt.int8,
            input_shapes["host_has_initial_state"],
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
            remove_input_padding=remove_input_padding,
            paged_state=paged_state,
        )
        output_tensor = layer.get_output(0)
        output_tensor.name = "output"
        network.trt_network.mark_output(output_tensor)
        final_state_tensor = layer.get_output(1)
        final_state_tensor.name = "final_state"
        network.trt_network.mark_output(final_state_tensor)

    builder_config = builder.create_builder_config(precision="bfloat16")
    for shape_ranges in optimization_profiles:
        profile = builder.trt_builder.create_optimization_profile()
        for name, (min_shape, opt_shape, max_shape) in shape_ranges.items():
            profile.set_shape(name, min_shape, opt_shape, max_shape)
        builder_config.trt_builder_config.add_optimization_profile(profile)
    engine = builder.build_engine(network, builder_config)
    assert engine is not None
    return tensorrt_llm.runtime.Session.from_serialized_engine(engine)


def _run_gated_delta_rule_session(
    session: tensorrt_llm.runtime.Session,
    inputs: dict[str, torch.Tensor],
    context: trt.IExecutionContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = context or session.context
    session.set_shapes(inputs, context=context)
    output = torch.empty_like(inputs["value"])
    final_state = torch.empty(
        tuple(context.get_tensor_shape("final_state")),
        dtype=torch.float32,
        device="cuda",
    )
    stream = torch.cuda.current_stream()
    ok = session.run(
        inputs=inputs,
        outputs={"output": output, "final_state": final_state},
        stream=stream.cuda_stream,
        context=context,
    )
    assert ok
    stream.synchronize()
    return output, final_state


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


def _gated_delta_rule_paged_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state_pool: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    slot_indices = state_slot_mapping.to(torch.long)
    request_states = state_pool.index_select(0, slot_indices)
    output, updated_request_states = _gated_delta_rule_decode_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        request_states,
        has_initial_state,
    )
    updated_state_pool = state_pool.clone()
    updated_state_pool.index_copy_(0, slot_indices, updated_request_states)
    return output, updated_state_pool


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


def _gated_delta_rule_prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    activation_dtype = value.dtype
    query = query.squeeze(0).float()
    key = key.squeeze(0).float()
    value_fp32 = value.squeeze(0).float()
    log_decay = log_decay.squeeze(0).float()
    beta = beta.squeeze(0).float()

    query = query / torch.sqrt(torch.sum(query * query, dim=-1, keepdim=True) + 1e-6)
    key = key / torch.sqrt(torch.sum(key * key, dim=-1, keepdim=True) + 1e-6)
    heads_ratio = value.shape[2] // query.shape[1]
    query = query.repeat_interleave(heads_ratio, dim=1)
    key = key.repeat_interleave(heads_ratio, dim=1)

    output = torch.empty_like(value_fp32)
    final_state = state.clone().float()
    sequence_offsets = cu_seqlens.cpu().tolist()
    state_slots = state_slot_mapping.cpu().tolist()
    initial_state_flags = has_initial_state.cpu().tolist()
    scale = HEAD_K_DIM**-0.5
    for request_idx, (begin, end) in enumerate(zip(sequence_offsets[:-1], sequence_offsets[1:])):
        state_slot = state_slots[request_idx]
        recurrent_state = (
            state[state_slot].clone().float()
            if initial_state_flags[request_idx]
            else torch.zeros_like(state[state_slot], dtype=torch.float32)
        )
        for token_idx in range(begin, end):
            recurrent_state *= torch.exp(log_decay[token_idx])[:, None, None]
            value_residual = value_fp32[token_idx] - torch.einsum(
                "hvk,hk->hv", recurrent_state, key[token_idx]
            )
            value_residual *= beta[token_idx, :, None]
            recurrent_state += torch.einsum("hv,hk->hvk", value_residual, key[token_idx])
            output[token_idx] = torch.einsum(
                "hvk,hk->hv", recurrent_state, query[token_idx] * scale
            )
        final_state[state_slot] = recurrent_state
    return output.unsqueeze(0).to(activation_dtype), final_state


@torch.no_grad()
def _gated_delta_rule_prefill_chunk_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    log_decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    has_initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = l2norm_fwd(query)
    key = l2norm_fwd(key)
    cumulative_decay = chunk_local_cumsum(log_decay, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens)
    transition = chunk_scaled_dot_kkt_fwd(
        k=key,
        beta=beta,
        g_cumsum=cumulative_decay,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
        output_dtype=torch.float32,
    )
    transition = solve_tril(
        A=transition,
        cu_seqlens=cu_seqlens,
        output_dtype=key.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=key,
        v=value,
        beta=beta,
        A=transition,
        g_cumsum=cumulative_decay,
        cu_seqlens=cu_seqlens,
    )

    final_state = state.clone()
    initial_state_mask = has_initial_state.to(device=state.device, dtype=torch.bool)
    missing_state_slots = state_slot_mapping[~initial_state_mask]
    final_state.index_fill_(0, missing_state_slots.to(torch.long), 0)
    chunk_state, updated_value, _ = chunk_gated_delta_rule_fwd_h(
        k=key,
        w=w,
        u=u,
        g=cumulative_decay,
        initial_state=final_state,
        initial_state_indices=state_slot_mapping,
        output_final_state=True,
        inplace_indexed_state_update=True,
        chunk_size=CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
    )
    output = chunk_fwd_o(
        q=query,
        k=key,
        v=updated_value,
        h=chunk_state,
        g=cumulative_decay,
        scale=HEAD_K_DIM**-0.5,
        cu_seqlens=cu_seqlens,
        chunk_size=CHUNK_SIZE,
    )
    return output, final_state


@pytest.mark.parametrize(
    "num_q_heads,num_v_heads,sequence_lengths,use_chunk_reference",
    (
        pytest.param(16, 16, SHORT_PREFILL_SEQUENCE_LENGTHS, False, id="qwen3.5-2b-short"),
        pytest.param(16, 32, SHORT_PREFILL_SEQUENCE_LENGTHS, False, id="qwen3.5-9b-short"),
        pytest.param(16, 48, SHORT_PREFILL_SEQUENCE_LENGTHS, False, id="qwen3.5-27b-short"),
        pytest.param(16, 16, (32, 33), False, id="qwen3.5-2b-batch2"),
        pytest.param(16, 16, (1, 32, 63, 64), False, id="qwen3.5-2b-batch4"),
        pytest.param(
            16,
            16,
            (1, 17, 31, 32, 33, 63, 64, 65),
            False,
            id="qwen3.5-2b-batch8",
        ),
        pytest.param(16, 16, LONG_PREFILL_SEQUENCE_LENGTHS, True, id="qwen3.5-2b-long"),
    ),
)
def test_gated_delta_rule_packed_prefill(
    num_q_heads: int,
    num_v_heads: int,
    sequence_lengths: tuple[int, ...],
    use_chunk_reference: bool,
) -> None:
    torch.manual_seed(1234)
    device = "cuda"
    total_tokens = sum(sequence_lengths)
    num_requests = len(sequence_lengths)

    query = torch.randn(
        1, total_tokens, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn(
        1, total_tokens, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16
    )
    log_decay = -0.1 * torch.rand(1, total_tokens, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total_tokens, num_v_heads, device=device))
    state = 0.02 * torch.randn(
        num_requests,
        num_v_heads,
        HEAD_V_DIM,
        HEAD_K_DIM,
        device=device,
        dtype=torch.float32,
    )
    initial_state = state.clone()

    host_request_types = torch.zeros(num_requests, dtype=torch.int32)
    cu_seqlens = torch.tensor(
        [0, *np.cumsum(sequence_lengths).tolist()], device=device, dtype=torch.int32
    )
    state_slot_mapping = torch.arange(num_requests, device=device, dtype=torch.int32)
    host_has_initial_state = torch.tensor(
        [request_idx % 2 == 0 for request_idx in range(num_requests)], dtype=torch.int8
    )

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
            remove_input_padding=True,
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
    final_state = torch.empty_like(state)
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
        outputs={"output": output, "final_state": final_state},
        stream=stream.cuda_stream,
    )
    assert ok
    stream.synchronize()

    reference = (
        _gated_delta_rule_prefill_chunk_reference
        if use_chunk_reference
        else _gated_delta_rule_prefill_reference
    )
    output_ref, final_state_ref = reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state,
        cu_seqlens,
        state_slot_mapping,
        host_has_initial_state,
    )
    output_atol = 8e-2 if use_chunk_reference else 2e-2
    torch.testing.assert_close(output.float(), output_ref.float(), atol=output_atol, rtol=2e-2)
    state_atol = 1e-1 if use_chunk_reference else 1e-2
    torch.testing.assert_close(final_state, final_state_ref, atol=state_atol, rtol=1e-2)



def test_gated_delta_rule_paged_state_prefill_decode_continuity() -> None:
    torch.manual_seed(1234)
    device = "cuda"
    num_q_heads = 16
    num_v_heads = 16
    num_requests = 2
    sequence_lengths = (17, 33)
    total_tokens = sum(sequence_lengths)

    state_pool = 0.02 * torch.randn(
        4,
        num_v_heads,
        HEAD_V_DIM,
        HEAD_K_DIM,
        device=device,
        dtype=torch.float32,
    )
    expected_state_pool = state_pool.clone()
    state_pointer = torch.tensor([state_pool.data_ptr()], dtype=torch.int64)
    state_slot_mapping = torch.arange(num_requests, device=device, dtype=torch.int32)
    host_has_initial_state = torch.tensor([0, 1], dtype=torch.int8)
    host_request_types = torch.zeros(num_requests, dtype=torch.int32)
    cu_seqlens = torch.tensor(
        [0, *np.cumsum(sequence_lengths).tolist()], device=device, dtype=torch.int32
    )
    query = torch.randn(
        1, total_tokens, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn(
        1, total_tokens, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16
    )
    log_decay = -0.1 * torch.rand(1, total_tokens, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total_tokens, num_v_heads, device=device))
    prefill_inputs = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "beta": beta,
        "state": state_pointer,
        "host_request_types": host_request_types,
        "cu_seqlens": cu_seqlens,
        "state_slot_mapping": state_slot_mapping,
        "host_has_initial_state": host_has_initial_state,
    }
    prefill_session = _build_gated_delta_rule_session(
        {name: tuple(tensor.shape) for name, tensor in prefill_inputs.items()},
        num_q_heads,
        num_v_heads,
        paged_state=True,
        remove_input_padding=True,
    )
    output, final_state = _run_gated_delta_rule_session(prefill_session, prefill_inputs)
    output_ref, expected_state_pool = _gated_delta_rule_prefill_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        expected_state_pool,
        cu_seqlens,
        state_slot_mapping,
        host_has_initial_state,
    )
    torch.testing.assert_close(output.float(), output_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state_pool, expected_state_pool, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        final_state,
        expected_state_pool.index_select(0, state_slot_mapping.to(torch.long)),
        atol=1e-2,
        rtol=1e-2,
    )

    host_request_types = torch.ones(num_requests, dtype=torch.int32)
    expected_state_pool = state_pool.clone()
    host_has_initial_state = torch.ones(num_requests, dtype=torch.int8)
    cu_seqlens = torch.arange(num_requests + 1, device=device, dtype=torch.int32)
    decode_shapes = {
        "query": (num_requests, 1, num_q_heads, HEAD_K_DIM),
        "key": (num_requests, 1, num_q_heads, HEAD_K_DIM),
        "value": (num_requests, 1, num_v_heads, HEAD_V_DIM),
        "log_decay": (num_requests, 1, num_v_heads),
        "beta": (num_requests, 1, num_v_heads),
        "state": (1,),
        "host_request_types": (num_requests,),
        "cu_seqlens": (num_requests + 1,),
        "state_slot_mapping": (num_requests,),
        "host_has_initial_state": (num_requests,),
    }
    decode_session = _build_gated_delta_rule_session(
        decode_shapes,
        num_q_heads,
        num_v_heads,
        paged_state=True,
        remove_input_padding=True,
    )
    for _ in range(3):
        query = torch.randn(
            num_requests, 1, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
        )
        key = torch.randn_like(query)
        value = torch.randn(
            num_requests,
            1,
            num_v_heads,
            HEAD_V_DIM,
            device=device,
            dtype=torch.bfloat16,
        )
        log_decay = -0.1 * torch.rand(num_requests, 1, num_v_heads, device=device)
        beta = torch.sigmoid(torch.randn(num_requests, 1, num_v_heads, device=device))
        decode_inputs = {
            "query": query,
            "key": key,
            "value": value,
            "log_decay": log_decay,
            "beta": beta,
            "state": state_pointer,
            "host_request_types": host_request_types,
            "cu_seqlens": cu_seqlens,
            "state_slot_mapping": state_slot_mapping,
            "host_has_initial_state": host_has_initial_state,
        }
        output, _ = _run_gated_delta_rule_session(decode_session, decode_inputs)
        output_ref, expected_state_pool = _gated_delta_rule_paged_decode_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            expected_state_pool,
            state_slot_mapping,
            host_has_initial_state,
        )
        torch.testing.assert_close(output.float(), output_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(state_pool, expected_state_pool, atol=2e-3, rtol=2e-3)


def test_gated_delta_rule_paged_state_non_contiguous_slot_mapping() -> None:
    torch.manual_seed(5678)
    device = "cuda"
    num_q_heads = 16
    num_v_heads = 16
    sequence_lengths = (17, 33, 65)
    num_requests = len(sequence_lengths)
    total_tokens = sum(sequence_lengths)

    state_pool = 0.02 * torch.randn(
        8,
        num_v_heads,
        HEAD_V_DIM,
        HEAD_K_DIM,
        device=device,
        dtype=torch.float32,
    )
    initial_state_pool = state_pool.clone()
    state_pointer = torch.tensor([state_pool.data_ptr()], dtype=torch.int64)
    state_slot_mapping = torch.tensor([6, 2, 5], device=device, dtype=torch.int32)
    host_has_initial_state = torch.tensor([1, 0, 1], dtype=torch.int8)
    host_request_types = torch.zeros(num_requests, dtype=torch.int32)
    cu_seqlens = torch.tensor(
        [0, *np.cumsum(sequence_lengths).tolist()], device=device, dtype=torch.int32
    )
    query = torch.randn(
        1, total_tokens, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
    )
    key = torch.randn_like(query)
    value = torch.randn(
        1, total_tokens, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16
    )
    log_decay = -0.1 * torch.rand(1, total_tokens, num_v_heads, device=device)
    beta = torch.sigmoid(torch.randn(1, total_tokens, num_v_heads, device=device))
    inputs = {
        "query": query,
        "key": key,
        "value": value,
        "log_decay": log_decay,
        "beta": beta,
        "state": state_pointer,
        "host_request_types": host_request_types,
        "cu_seqlens": cu_seqlens,
        "state_slot_mapping": state_slot_mapping,
        "host_has_initial_state": host_has_initial_state,
    }
    session = _build_gated_delta_rule_session(
        {name: tuple(tensor.shape) for name, tensor in inputs.items()},
        num_q_heads,
        num_v_heads,
        paged_state=True,
        remove_input_padding=True,
    )
    output, final_state = _run_gated_delta_rule_session(session, inputs)
    output_ref, expected_state_pool = _gated_delta_rule_prefill_reference(
        query,
        key,
        value,
        log_decay,
        beta,
        initial_state_pool,
        cu_seqlens,
        state_slot_mapping,
        host_has_initial_state,
    )

    torch.testing.assert_close(output.float(), output_ref.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(state_pool, expected_state_pool, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        final_state,
        expected_state_pool.index_select(0, state_slot_mapping.to(torch.long)),
        atol=1e-2,
        rtol=1e-2,
    )
    unused_slots = torch.tensor([0, 1, 3, 4, 7], device=device, dtype=torch.long)
    torch.testing.assert_close(
        state_pool.index_select(0, unused_slots),
        initial_state_pool.index_select(0, unused_slots),
        atol=0,
        rtol=0,
    )


def test_gated_delta_rule_dynamic_context_and_generation_profiles() -> None:
    torch.manual_seed(9012)
    device = "cuda"
    num_q_heads = 16
    num_v_heads = 16

    input_shapes = {
        "query": (-1, -1, num_q_heads, HEAD_K_DIM),
        "key": (-1, -1, num_q_heads, HEAD_K_DIM),
        "value": (-1, -1, num_v_heads, HEAD_V_DIM),
        "log_decay": (-1, -1, num_v_heads),
        "beta": (-1, -1, num_v_heads),
        "state": (-1, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
        "host_request_types": (-1,),
        "cu_seqlens": (-1,),
        "state_slot_mapping": (-1,),
        "host_has_initial_state": (-1,),
    }
    context_profile = {
        "query": (
            (1, 1, num_q_heads, HEAD_K_DIM),
            (1, 65, num_q_heads, HEAD_K_DIM),
            (1, 8193, num_q_heads, HEAD_K_DIM),
        ),
        "key": (
            (1, 1, num_q_heads, HEAD_K_DIM),
            (1, 65, num_q_heads, HEAD_K_DIM),
            (1, 8193, num_q_heads, HEAD_K_DIM),
        ),
        "value": (
            (1, 1, num_v_heads, HEAD_V_DIM),
            (1, 65, num_v_heads, HEAD_V_DIM),
            (1, 8193, num_v_heads, HEAD_V_DIM),
        ),
        "log_decay": ((1, 1, num_v_heads), (1, 65, num_v_heads), (1, 8193, num_v_heads)),
        "beta": ((1, 1, num_v_heads), (1, 65, num_v_heads), (1, 8193, num_v_heads)),
        "state": (
            (1, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
            (2, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
            (3, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
        ),
        "host_request_types": ((1,), (2,), (3,)),
        "cu_seqlens": ((2,), (3,), (4,)),
        "state_slot_mapping": ((1,), (2,), (3,)),
        "host_has_initial_state": ((1,), (2,), (3,)),
    }
    generation_profile = {
        "query": (
            (1, 1, num_q_heads, HEAD_K_DIM),
            (4, 1, num_q_heads, HEAD_K_DIM),
            (8, 1, num_q_heads, HEAD_K_DIM),
        ),
        "key": (
            (1, 1, num_q_heads, HEAD_K_DIM),
            (4, 1, num_q_heads, HEAD_K_DIM),
            (8, 1, num_q_heads, HEAD_K_DIM),
        ),
        "value": (
            (1, 1, num_v_heads, HEAD_V_DIM),
            (4, 1, num_v_heads, HEAD_V_DIM),
            (8, 1, num_v_heads, HEAD_V_DIM),
        ),
        "log_decay": ((1, 1, num_v_heads), (4, 1, num_v_heads), (8, 1, num_v_heads)),
        "beta": ((1, 1, num_v_heads), (4, 1, num_v_heads), (8, 1, num_v_heads)),
        "state": (
            (1, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
            (4, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
            (8, num_v_heads, HEAD_V_DIM, HEAD_K_DIM),
        ),
        "host_request_types": ((1,), (4,), (8,)),
        "cu_seqlens": ((2,), (5,), (9,)),
        "state_slot_mapping": ((1,), (4,), (8,)),
        "host_has_initial_state": ((1,), (4,), (8,)),
    }
    session = _build_gated_delta_rule_session(
        input_shapes,
        num_q_heads,
        num_v_heads,
        paged_state=False,
        remove_input_padding=True,
        optimization_profiles=(context_profile, generation_profile),
    )
    assert session.engine.num_optimization_profiles == 2

    for sequence_lengths in ((17,), (1, 64, 64), (8193,)):
        num_requests = len(sequence_lengths)
        total_tokens = sum(sequence_lengths)
        query = torch.randn(
            1, total_tokens, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
        )
        key = torch.randn_like(query)
        value = torch.randn(
            1, total_tokens, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16
        )
        log_decay = -0.1 * torch.rand(1, total_tokens, num_v_heads, device=device)
        beta = torch.sigmoid(torch.randn(1, total_tokens, num_v_heads, device=device))
        state = 0.02 * torch.randn(
            num_requests,
            num_v_heads,
            HEAD_V_DIM,
            HEAD_K_DIM,
            device=device,
            dtype=torch.float32,
        )
        initial_state = state.clone()
        host_request_types = torch.zeros(num_requests, dtype=torch.int32)
        cu_seqlens = torch.tensor(
            [0, *np.cumsum(sequence_lengths).tolist()], device=device, dtype=torch.int32
        )
        state_slot_mapping = torch.arange(num_requests, device=device, dtype=torch.int32)
        host_has_initial_state = torch.tensor(
            [request_idx % 2 == 0 for request_idx in range(num_requests)], dtype=torch.int8
        )
        inputs = {
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
        }
        output, final_state = _run_gated_delta_rule_session(session, inputs)
        output_ref, final_state_ref = _gated_delta_rule_prefill_reference(
            query,
            key,
            value,
            log_decay,
            beta,
            initial_state,
            cu_seqlens,
            state_slot_mapping,
            host_has_initial_state,
        )
        torch.testing.assert_close(output.float(), output_ref.float(), atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(final_state, final_state_ref, atol=1e-2, rtol=1e-2)

    generation_context = session.engine.create_execution_context()
    assert generation_context is not None
    stream = torch.cuda.current_stream()
    assert generation_context.set_optimization_profile_async(1, stream.cuda_stream)
    stream.synchronize()
    for batch_size in (1, 8):
        query = torch.randn(
            batch_size, 1, num_q_heads, HEAD_K_DIM, device=device, dtype=torch.bfloat16
        )
        key = torch.randn_like(query)
        value = torch.randn(
            batch_size, 1, num_v_heads, HEAD_V_DIM, device=device, dtype=torch.bfloat16
        )
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
        inputs = {
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
        }
        output, final_state = _run_gated_delta_rule_session(
            session, inputs, context=generation_context
        )
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
        torch.testing.assert_close(final_state, final_state_ref, atol=2e-3, rtol=2e-3)
