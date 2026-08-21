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

from pathlib import Path
from typing import TYPE_CHECKING

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig

if TYPE_CHECKING:
    import transformers


class Qwen35Config(PretrainedConfig):
    """Configuration for the dense, text-only Qwen3.5 TensorRT graph."""

    def __init__(
        self,
        *,
        decoder_layer_types: list[str],
        rotary_base: float = 10_000.0,
        rotary_scaling: dict | None = None,
        rotary_embedding_percentage: float = 0.25,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = True,
        attention_bias: bool = False,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        gated_delta_chunk_size: int = 64,
        state_dtype: str = "float32",
        **kwargs,
    ) -> None:
        layer_type_map = {
            "full_attention": "attention",
            "linear_attention": "linear",
        }
        unknown_layer_types = set(decoder_layer_types) - set(layer_type_map)
        if unknown_layer_types:
            raise ValueError(f"Unsupported Qwen3.5 layer types: {sorted(unknown_layer_types)}")

        self.decoder_layer_types = list(decoder_layer_types)
        self.layer_types = [layer_type_map[layer_type] for layer_type in decoder_layer_types]
        self.rotary_base = rotary_base
        self.rotary_scaling = rotary_scaling
        self.rotary_embedding_percentage = rotary_embedding_percentage
        self.mrope_section = list(mrope_section or [11, 11, 10])
        self.mrope_interleaved = mrope_interleaved
        self.attention_bias = attention_bias
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.gated_delta_chunk_size = gated_delta_chunk_size
        self.state_dtype = state_dtype

        super().__init__(**kwargs)

        if len(self.decoder_layer_types) != self.num_hidden_layers:
            raise ValueError(
                "decoder_layer_types must contain exactly num_hidden_layers entries, "
                f"got {len(self.decoder_layer_types)} and {self.num_hidden_layers}"
            )
        if self.dtype != "bfloat16":
            raise ValueError(
                f"The initial Qwen3.5 implementation only supports bfloat16, got {self.dtype}"
            )
        if self.attention_bias:
            raise ValueError("The initial Qwen3.5 implementation only supports bias-free attention")
        if self.state_dtype != "float32":
            raise ValueError(f"Qwen3.5 recurrent state must use float32, got {self.state_dtype}")
        if self.mapping.tp_size != 1 or self.mapping.pp_size != 1 or self.mapping.cp_size != 1:
            raise ValueError(
                "The initial Qwen3.5 implementation only supports TP=1, PP=1, and CP=1"
            )
        if (
            self.quantization.quant_algo is not None
            or self.quantization.kv_cache_quant_algo is not None
        ):
            raise ValueError("The initial Qwen3.5 implementation does not support quantization")
        if self.linear_num_value_heads % self.linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        if self.gated_delta_chunk_size != 64:
            raise ValueError("The GatedDeltaRule plugin currently requires chunk_size=64")
        if not self.mrope_interleaved:
            raise ValueError("Qwen3.5 requires interleaved MRoPE")
        if sum(self.mrope_section) * 2 != self.rotary_embedding_dim:
            raise ValueError(
                "Twice the sum of mrope_section must equal rotary_embedding_dim, "
                f"got {self.mrope_section} and {self.rotary_embedding_dim}"
            )

    @classmethod
    def from_hugging_face(
        cls,
        hf_config_or_dir: str | Path | "transformers.PretrainedConfig",
        dtype: str = "bfloat16",
        mapping: Mapping | None = None,
        quant_config: QuantConfig | None = None,
        **kwargs,
    ) -> "Qwen35Config":
        """Create a TensorRT-LLM config from a Hugging Face Qwen3.5 config."""
        import transformers

        trust_remote_code = kwargs.pop("trust_remote_code", True)
        if isinstance(hf_config_or_dir, transformers.PretrainedConfig):
            hf_config = hf_config_or_dir
        else:
            hf_config = transformers.AutoConfig.from_pretrained(
                str(hf_config_or_dir), trust_remote_code=trust_remote_code
            )

        if hasattr(hf_config, "text_config"):
            hf_config = hf_config.text_config
        if hf_config.model_type != "qwen3_5_text":
            raise ValueError(
                f"Expected a Qwen3.5 text config, got model_type={hf_config.model_type!r}"
            )

        source_dtype = getattr(hf_config, "dtype", None)
        if source_dtype is None:
            source_dtype = getattr(hf_config, "torch_dtype", None)
        if dtype == "auto" and source_dtype is None:
            dtype = "bfloat16"
        dtype = infer_dtype(dtype, source_dtype)

        rope_parameters = getattr(hf_config, "rope_parameters", None) or {}
        rotary_base = rope_parameters.get("rope_theta", 10_000.0)
        rotary_embedding_percentage = rope_parameters.get(
            "partial_rotary_factor", getattr(hf_config, "partial_rotary_factor", 0.25)
        )
        mrope_section = list(rope_parameters.get("mrope_section", [11, 11, 10]))
        mrope_interleaved = True
        rotary_scaling = dict(rope_parameters)
        rotary_scaling.update(
            type="mrope",
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )

        return cls(
            architecture="Qwen35ForCausalLM",
            dtype=dtype,
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_size=hf_config.head_dim,
            hidden_act=hf_config.hidden_act,
            norm_epsilon=hf_config.rms_norm_eps,
            position_embedding_type="mrope",
            max_position_embeddings=hf_config.max_position_embeddings,
            rotary_embedding_dim=int(hf_config.head_dim * rotary_embedding_percentage),
            rotary_base=rotary_base,
            rotary_scaling=rotary_scaling,
            rotary_embedding_percentage=rotary_embedding_percentage,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
            attention_bias=hf_config.attention_bias,
            decoder_layer_types=list(hf_config.layer_types),
            linear_conv_kernel_dim=hf_config.linear_conv_kernel_dim,
            linear_key_head_dim=hf_config.linear_key_head_dim,
            linear_value_head_dim=hf_config.linear_value_head_dim,
            linear_num_key_heads=hf_config.linear_num_key_heads,
            linear_num_value_heads=hf_config.linear_num_value_heads,
            gated_delta_chunk_size=64,
            state_dtype="float32",
            mapping=mapping,
            quantization=quant_config,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            **kwargs,
        )
