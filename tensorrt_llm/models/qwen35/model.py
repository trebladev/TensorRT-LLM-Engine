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

from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (
    Tensor,
    cast,
    concat,
    exp,
    gather_last_token_logits,
    shape,
    sigmoid,
    softplus,
    split,
)
from ...layers import (
    Attention,
    AttentionMaskType,
    ColumnLinear,
    Embedding,
    GatedDeltaRule,
    GatedMLP,
    KeyValueCacheParams,
    RmsNorm,
    RmsNormGate,
    RowLinear,
)
from ...layers.ssm import MambaConv1d
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ..generation_mixin import GenerationMixin
from ..modeling_utils import PretrainedModel, QuantConfig, get_kv_cache_type_from_legacy
from .config import Qwen35Config
from .convert import convert_hf_qwen35, load_weights_from_hf_checkpoint

if TYPE_CHECKING:
    import transformers


_BF16_ELEMENT_BYTES = 2
_FP32_ELEMENT_BYTES = 4


class _AttentionGate:
    def __init__(self) -> None:
        self.value: Tensor | None = None


class _Qwen35QkvProjection(Module):
    def __init__(self, config: Qwen35Config, gate: _AttentionGate) -> None:
        super().__init__()
        attention_size = config.num_attention_heads * config.head_size
        kv_size = config.num_key_value_heads * config.head_size
        linear_kwargs = {
            "bias": config.attention_bias,
            "dtype": config.dtype,
            "tp_group": config.mapping.tp_group,
            "tp_size": config.mapping.tp_size,
            "gather_output": False,
        }
        self.query = ColumnLinear(config.hidden_size, attention_size, **linear_kwargs)
        self.gate = ColumnLinear(config.hidden_size, attention_size, **linear_kwargs)
        self.key = ColumnLinear(config.hidden_size, kv_size, **linear_kwargs)
        self.value = ColumnLinear(config.hidden_size, kv_size, **linear_kwargs)
        self.query_norm = RmsNorm(config.head_size, eps=config.norm_epsilon, dtype=config.dtype)
        self.key_norm = RmsNorm(config.head_size, eps=config.norm_epsilon, dtype=config.dtype)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_size = config.head_size
        self._gate = gate

    def _reshape_heads(self, tensor: Tensor, num_heads: int) -> tuple[Tensor, Tensor]:
        if tensor.ndim() == 2:
            base_shape = shape(tensor, 0)
        else:
            base_shape = concat([shape(tensor, 0), shape(tensor, 1)])
        tensor = tensor.view(concat([base_shape, num_heads, self.head_size]))
        return tensor, base_shape

    def forward(self, hidden_states: Tensor, lora_runtime_params=None) -> Tensor:
        if lora_runtime_params is not None:
            raise ValueError("Qwen3.5 does not support LoRA in the initial implementation")
        query = self.query(hidden_states)
        self._gate.value = self.gate(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query, base_shape = self._reshape_heads(query, self.num_attention_heads)
        key, _ = self._reshape_heads(key, self.num_key_value_heads)
        value, _ = self._reshape_heads(value, self.num_key_value_heads)
        query = self.query_norm(query)
        key = self.key_norm(key)
        qkv = concat([query, key, value], dim=query.ndim() - 2)
        total_size = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_size
        return qkv.view(concat([base_shape, total_size]))


class _Qwen35OutputProjection(Module):
    def __init__(self, config: Qwen35Config, gate: _AttentionGate) -> None:
        super().__init__()
        attention_size = config.num_attention_heads * config.head_size
        self.proj = RowLinear(
            attention_size,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
        )
        self._gate = gate

    def forward(self, context: Tensor, lora_runtime_params=None, all_reduce_params=None) -> Tensor:
        if lora_runtime_params is not None:
            raise ValueError("Qwen3.5 does not support LoRA in the initial implementation")
        if self._gate.value is None:
            raise RuntimeError(
                "The Qwen3.5 attention gate was not produced before the output projection"
            )
        context = context * sigmoid(self._gate.value)
        return self.proj(context, all_reduce_params=all_reduce_params)


class Qwen35Attention(Attention):
    """Qwen3.5 full attention with per-head Q/K norm and an output gate."""

    def __init__(self, config: Qwen35Config, local_layer_idx: int) -> None:
        super().__init__(
            local_layer_idx=local_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attention_bias,
            dense_bias=config.attention_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            rotary_embedding_percentage=config.rotary_embedding_percentage,
            tp_rank=config.mapping.tp_rank,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            cp_rank=config.mapping.cp_rank,
            cp_group=config.mapping.cp_group,
            cp_size=config.mapping.cp_size,
            quant_mode=config.quant_mode,
            enable_qkv=False,
        )
        gate = _AttentionGate()
        self.qkv = _Qwen35QkvProjection(config, gate)
        self.dense = _Qwen35OutputProjection(config, gate)


class Qwen35LinearAttention(Module):
    """Qwen3.5 gated-delta linear attention block."""

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.num_q_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_q_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        linear_kwargs = {
            "bias": False,
            "dtype": config.dtype,
            "tp_group": config.mapping.tp_group,
            "tp_size": config.mapping.tp_size,
            "gather_output": False,
        }
        self.in_proj_qkv = ColumnLinear(config.hidden_size, self.conv_dim, **linear_kwargs)
        self.in_proj_z = ColumnLinear(config.hidden_size, self.value_dim, **linear_kwargs)
        self.in_proj_b = ColumnLinear(config.hidden_size, self.num_v_heads, **linear_kwargs)
        self.in_proj_a = ColumnLinear(config.hidden_size, self.num_v_heads, **linear_kwargs)
        gated_delta_state_bytes = (
            self.num_v_heads * self.head_v_dim * self.head_k_dim * _FP32_ELEMENT_BYTES
        )
        conv_history = config.linear_conv_kernel_dim - 1
        conv_state_bytes = conv_history * self.conv_dim * _BF16_ELEMENT_BYTES
        self.state_slot_stride_bytes = gated_delta_state_bytes + conv_state_bytes
        self.conv_state_channel_stride_bytes = conv_history * _BF16_ELEMENT_BYTES
        self.conv_state_history_stride_bytes = _BF16_ELEMENT_BYTES
        self.conv1d = MambaConv1d(
            self.conv_dim,
            config.linear_conv_kernel_dim,
            dtype=config.dtype,
            apply_silu=True,
            state_slot_stride_bytes=self.state_slot_stride_bytes,
            state_channel_stride_bytes=self.conv_state_channel_stride_bytes,
            state_history_stride_bytes=self.conv_state_history_stride_bytes,
        )
        self.dt_bias = Parameter(shape=(self.num_v_heads,), dtype="float32")
        self.A_log = Parameter(shape=(self.num_v_heads,), dtype="float32")
        self.gated_delta_rule = GatedDeltaRule(
            num_q_heads=self.num_q_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            chunk_size=config.gated_delta_chunk_size,
            dtype=config.dtype,
            state_dtype=config.state_dtype,
            state_slot_stride_bytes=self.state_slot_stride_bytes,
            remove_input_padding=True,
            use_qk_l2norm=True,
        )
        self.norm = RmsNormGate(self.head_v_dim, eps=config.norm_epsilon, dtype=config.dtype)
        self.out_proj = RowLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
        )

    def forward(
        self,
        hidden_states: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        host_request_types: Tensor,
        last_token_ids: Tensor,
        host_context_lengths: Tensor,
        cu_seqlens: Tensor,
        state_slot_mapping: Tensor,
        host_has_initial_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        paged_state = default_net().plugin_config.paged_state
        self.conv1d.state_slot_stride_bytes = self.state_slot_stride_bytes if paged_state else 0
        self.conv1d.state_channel_stride_bytes = (
            self.conv_state_channel_stride_bytes if paged_state else 0
        )
        self.conv1d.state_history_stride_bytes = (
            self.conv_state_history_stride_bytes if paged_state else 0
        )
        self.gated_delta_rule.state_slot_stride_bytes = (
            self.state_slot_stride_bytes if paged_state else 0
        )

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv, present_conv_state = self.conv1d(
            mixed_qkv,
            conv_state,
            host_request_types,
            last_token_ids,
            host_context_lengths=host_context_lengths,
            slot_mapping=state_slot_mapping,
            host_has_initial_state=host_has_initial_state,
        )
        query, key, value = split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        num_tokens = shape(hidden_states, 0)
        query = query.view(concat([1, num_tokens, self.num_q_heads, self.head_k_dim]))
        key = key.view(concat([1, num_tokens, self.num_q_heads, self.head_k_dim]))
        value = value.view(concat([1, num_tokens, self.num_v_heads, self.head_v_dim]))

        gate = self.in_proj_z(hidden_states)
        beta = sigmoid(cast(self.in_proj_b(hidden_states), "float32"))
        decay_input = cast(self.in_proj_a(hidden_states), "float32") + self.dt_bias.value
        log_decay = exp(self.A_log.value) * -1.0 * softplus(decay_input, beta=1.0, threshold=20.0)
        beta = beta.view(concat([1, num_tokens, self.num_v_heads]))
        log_decay = log_decay.view(concat([1, num_tokens, self.num_v_heads]))

        output, present_recurrent_state = self.gated_delta_rule(
            query,
            key,
            value,
            log_decay,
            beta,
            recurrent_state,
            host_request_types,
            cu_seqlens,
            state_slot_mapping,
            host_has_initial_state,
        )
        output = output.view(concat([num_tokens * self.num_v_heads, self.head_v_dim]))
        gate = gate.view(concat([num_tokens * self.num_v_heads, self.head_v_dim]))
        output = self.norm(output, gate)
        output = output.view(concat([num_tokens, self.value_dim]))
        return self.out_proj(output), present_conv_state, present_recurrent_state


class Qwen35DecoderLayer(Module):
    def __init__(self, config: Qwen35Config, layer_idx: int, attention_layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.decoder_layer_types[layer_idx]
        self.input_layernorm = RmsNorm(
            config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.post_layernorm = RmsNorm(
            config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            dtype=config.dtype,
            bias=False,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
        )
        if self.layer_type == "full_attention":
            self.attention = Qwen35Attention(config, attention_layer_idx)
        else:
            self.linear_attn = Qwen35LinearAttention(config)

    def forward(
        self,
        hidden_states: Tensor,
        use_cache: bool,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        mrope_params=None,
        conv_state=None,
        recurrent_state=None,
        host_request_types=None,
        last_token_ids=None,
        host_context_lengths=None,
        cu_seqlens=None,
        state_slot_mapping=None,
        host_has_initial_state=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "full_attention":
            token_mixer_output = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
                mrope_params=mrope_params,
            )
            if use_cache:
                token_mixer_output, present_kv = token_mixer_output
            else:
                present_kv = None
            present_conv = None
            present_recurrent = None
        else:
            token_mixer_output, present_conv, present_recurrent = self.linear_attn(
                hidden_states,
                conv_state,
                recurrent_state,
                host_request_types,
                last_token_ids,
                host_context_lengths,
                cu_seqlens,
                state_slot_mapping,
                host_has_initial_state,
            )
            present_kv = None

        hidden_states = residual + token_mixer_output
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, present_kv, present_conv, present_recurrent


class Qwen35Model(Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.vocab_embedding = Embedding(config.vocab_size, config.hidden_size, dtype=config.dtype)
        attention_layer_idx = 0
        layers = []
        for layer_idx, layer_type in enumerate(config.decoder_layer_types):
            layers.append(Qwen35DecoderLayer(config, layer_idx, attention_layer_idx))
            if layer_type == "full_attention":
                attention_layer_idx += 1
        self.layers = ModuleList(layers)
        self.ln_f = RmsNorm(config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype)

    def forward(
        self,
        input_ids: Tensor,
        use_cache: bool,
        attention_mask,
        kv_cache_params,
        attention_params,
        mrope_params,
        conv_states,
        recurrent_states,
        host_request_types,
        last_token_ids,
        host_context_lengths,
        cu_seqlens,
        state_slot_mapping,
        host_has_initial_state,
    ):
        hidden_states = self.vocab_embedding(input_ids)
        kv_cache_params.fill_none_tensor_list(
            sum(layer.layer_type == "full_attention" for layer in self.layers)
        )
        present_kvs = []
        present_convs = []
        present_recurrent_states = []
        attention_idx = 0
        recurrent_idx = 0
        for layer in self.layers:
            layer_kv_cache_params = None
            conv_state = None
            recurrent_state = None
            if layer.layer_type == "full_attention":
                layer_kv_cache_params = KeyValueCacheParams(
                    past_key_value=[kv_cache_params.past_key_value[attention_idx]],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection,
                )
                attention_idx += 1
            else:
                conv_state = conv_states[recurrent_idx]
                recurrent_state = recurrent_states[recurrent_idx]
                recurrent_idx += 1

            hidden_states, present_kv, present_conv, present_recurrent = layer(
                hidden_states,
                use_cache,
                attention_mask,
                layer_kv_cache_params,
                attention_params,
                mrope_params,
                conv_state,
                recurrent_state,
                host_request_types,
                last_token_ids,
                host_context_lengths,
                cu_seqlens,
                state_slot_mapping,
                host_has_initial_state,
            )
            if present_kv is not None:
                present_kvs.append(present_kv)
            if present_conv is not None:
                present_convs.append(present_conv)
            if present_recurrent is not None:
                present_recurrent_states.append(present_recurrent)
        return self.ln_f(hidden_states), present_kvs, present_convs, present_recurrent_states


class Qwen35ForCausalLM(PretrainedModel):
    config_class = Qwen35Config

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__(config)
        self.dtype = str_dtype_to_trt(config.dtype)
        self._logits_dtype = str_dtype_to_trt(config.logits_dtype)
        self.gather_context_logits = False
        self.attention_layer_ids = [
            idx
            for idx, layer_type in enumerate(config.decoder_layer_types)
            if layer_type == "full_attention"
        ]
        self.recurrent_layer_ids = [
            idx
            for idx, layer_type in enumerate(config.decoder_layer_types)
            if layer_type == "linear_attention"
        ]
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type
        self.transformer = Qwen35Model(config)
        self.lm_head = ColumnLinear(
            config.hidden_size,
            pad_vocab_size(config.vocab_size, config.mapping.tp_size),
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=True,
        last_token_ids=None,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        mrope_params=None,
        conv_states=None,
        recurrent_states=None,
        host_request_types=None,
        cu_seqlens=None,
        state_slot_mapping=None,
        host_has_initial_state=None,
    ):
        del position_ids
        attention_params = Attention.fill_attention_params(self, attention_params)
        hidden_states, present_kvs, present_convs, present_recurrent_states = self.transformer(
            input_ids,
            use_cache,
            attention_mask,
            kv_cache_params,
            attention_params,
            mrope_params,
            conv_states,
            recurrent_states,
            host_request_types,
            last_token_ids,
            attention_params.host_context_lengths,
            cu_seqlens,
            state_slot_mapping,
            host_has_initial_state,
        )
        if not self.gather_context_logits:
            hidden_states = gather_last_token_logits(
                hidden_states, last_token_ids, default_net().plugin_config.remove_input_padding
            )
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", self._logits_dtype)

        if use_cache and not default_net().plugin_config.paged_kv_cache:
            for layer_idx, present in zip(self.attention_layer_ids, present_kvs):
                present.mark_output(f"present_key_value_{layer_idx}", self.config.kv_dtype)
        if not default_net().plugin_config.paged_state:
            for layer_idx, present in zip(self.recurrent_layer_ids, present_convs):
                present.mark_output(f"present_conv_state_{layer_idx}", self.dtype)
            for layer_idx, present in zip(self.recurrent_layer_ids, present_recurrent_states):
                present.mark_output(f"present_recurrent_state_{layer_idx}", self.config.state_dtype)
        return lm_logits, present_kvs, present_convs, present_recurrent_states

    def _prepare_recurrent_inputs(
        self, num_profiles: int, batch_range: list[list[int]]
    ) -> dict[str, object]:
        paged_state = default_net().plugin_config.paged_state
        one_dim_range = OrderedDict([("buffer_count", [1] * num_profiles)])
        conv_dim_range = OrderedDict(
            [
                ("batch_size", batch_range),
                ("kernel_size", [self.config.linear_conv_kernel_dim - 1] * num_profiles),
                (
                    "conv_dim",
                    [
                        2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
                        + self.config.linear_num_value_heads * self.config.linear_value_head_dim
                    ]
                    * num_profiles,
                ),
            ]
        )
        recurrent_dim_range = OrderedDict(
            [
                ("state_slots", batch_range),
                ("value_heads", [self.config.linear_num_value_heads] * num_profiles),
                ("value_head_dim", [self.config.linear_value_head_dim] * num_profiles),
                ("key_head_dim", [self.config.linear_key_head_dim] * num_profiles),
            ]
        )
        conv_states = []
        recurrent_states = []
        for layer_idx in self.recurrent_layer_ids:
            if paged_state:
                conv_state = Tensor(
                    name=f"conv_state_ptr_{layer_idx}",
                    dtype=trt.int64,
                    shape=[1],
                    dim_range=one_dim_range,
                    location=trt.TensorLocation.HOST,
                )
                recurrent_state = Tensor(
                    name=f"recurrent_state_ptr_{layer_idx}",
                    dtype=trt.int64,
                    shape=[1],
                    dim_range=one_dim_range,
                    location=trt.TensorLocation.HOST,
                )
            else:
                conv_state = Tensor(
                    name=f"past_conv_state_{layer_idx}",
                    dtype=self.dtype,
                    shape=[-1, self.config.linear_conv_kernel_dim - 1, -1],
                    dim_range=conv_dim_range,
                )
                recurrent_state = Tensor(
                    name=f"past_recurrent_state_{layer_idx}",
                    dtype=str_dtype_to_trt(self.config.state_dtype),
                    shape=[-1, -1, -1, -1],
                    dim_range=recurrent_dim_range,
                )
            conv_states.append(conv_state)
            recurrent_states.append(recurrent_state)

        batch_dim_range = OrderedDict([("batch_size", batch_range)])
        cu_seqlens_range = [[value + 1 for value in profile] for profile in batch_range]
        return {
            "conv_states": conv_states,
            "recurrent_states": recurrent_states,
            "cu_seqlens": Tensor(
                name="gated_delta_cu_seqlens",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size_plus_one", cu_seqlens_range)]),
            ),
            "state_slot_mapping": Tensor(
                name="state_slot_mapping", dtype=trt.int32, shape=[-1], dim_range=batch_dim_range
            ),
            "host_has_initial_state": Tensor(
                name="host_has_initial_state",
                dtype=trt.int32,
                shape=[-1],
                dim_range=batch_dim_range,
                location=trt.TensorLocation.HOST,
            ),
        }

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_seq_len,
        max_num_tokens,
        use_cache,
        max_beam_width: int = 1,
        opt_num_tokens: int | None = None,
        prompt_embedding_table_size: int = 0,
        position_encoding_2d: bool = False,
        max_draft_len: int = 0,
        speculative_decoding_draft_tokens_external: bool = False,
        spec_decoding_is_generation_length_variable: bool = False,
        gather_context_logits: bool = False,
        lora_target_modules: list[str] | None = None,
        opt_batch_size: int = 0,
        mrope_rotary_cos_sin_size: int | None = None,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"Unsupported Qwen3.5 prepare_inputs arguments: {sorted(kwargs)}")
        if max_beam_width != 1:
            raise ValueError("The initial Qwen3.5 implementation does not support beam search")
        if max_draft_len != 0 or speculative_decoding_draft_tokens_external:
            raise ValueError(
                "The initial Qwen3.5 implementation does not support speculative decoding"
            )
        if spec_decoding_is_generation_length_variable:
            raise ValueError(
                "The initial Qwen3.5 implementation does not support variable generation lengths"
            )
        if prompt_embedding_table_size != 0 or position_encoding_2d or lora_target_modules:
            raise ValueError(
                "The initial Qwen3.5 implementation does not support prompt tuning or LoRA"
            )
        if not default_net().plugin_config.remove_input_padding:
            raise ValueError("Qwen3.5 GatedDeltaRule currently requires remove_input_padding=true")
        if not default_net().plugin_config.mamba_conv1d_plugin:
            raise ValueError("Qwen3.5 linear attention requires the mamba_conv1d plugin")

        expected_mrope_size = self.config.max_position_embeddings * self.config.rotary_embedding_dim
        if mrope_rotary_cos_sin_size is None:
            mrope_rotary_cos_sin_size = expected_mrope_size
        elif mrope_rotary_cos_sin_size != expected_mrope_size:
            raise ValueError(
                f"Expected mrope_rotary_cos_sin_size={expected_mrope_size}, "
                f"got {mrope_rotary_cos_sin_size}"
            )
        self.gather_context_logits = gather_context_logits
        result = super().prepare_inputs(
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_num_tokens=max_num_tokens,
            use_cache=use_cache,
            max_beam_width=max_beam_width,
            opt_num_tokens=opt_num_tokens,
            max_draft_len=max_draft_len,
            gather_context_logits=gather_context_logits,
            opt_batch_size=opt_batch_size,
            num_hidden_layers=len(self.attention_layer_ids),
            mrope_rotary_cos_sin_size=mrope_rotary_cos_sin_size,
        )

        kv_cache_type = get_kv_cache_type_from_legacy(
            use_cache, default_net().plugin_config.paged_kv_cache
        )
        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=default_net().plugin_config.gpt_attention_plugin,
            use_gemm_plugin=default_net().plugin_config.gemm_plugin,
            remove_input_padding=True,
            kv_cache_type=kv_cache_type,
        )
        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            max_draft_len=max_draft_len,
            opt_batch_size=opt_batch_size,
            opt_num_tokens=opt_num_tokens,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=default_net().plugin_config.multiple_profiles,
            kv_cache_type=kv_cache_type,
        )
        recurrent_inputs = self._prepare_recurrent_inputs(num_profiles, ranges["bb_range"])
        result.update(recurrent_inputs)
        result["host_request_types"] = result["attention_params"].host_request_types
        return result

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir: str | Path | "transformers.PreTrainedModel",
        dtype: str = "bfloat16",
        mapping: Mapping | None = None,
        quant_config: QuantConfig | None = None,
        **kwargs,
    ) -> "Qwen35ForCausalLM":
        """Create and load the dense text-only Qwen3.5 TensorRT-LLM model."""
        import transformers

        if isinstance(hf_model_or_dir, transformers.PreTrainedModel):
            hf_config_or_dir = hf_model_or_dir.config
        else:
            hf_config_or_dir = hf_model_or_dir
        config = Qwen35Config.from_hugging_face(
            hf_config_or_dir,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            **kwargs,
        )
        model = cls(config)
        if isinstance(hf_model_or_dir, transformers.PreTrainedModel):
            weights = convert_hf_qwen35(hf_model_or_dir, config)
        else:
            weights = load_weights_from_hf_checkpoint(hf_model_or_dir, config)
        model.load(weights)
        return model
