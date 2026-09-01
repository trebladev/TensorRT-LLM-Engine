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
    """Configuration for the dense Qwen3.5 TensorRT graph."""

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
        vision_depth: int | None = None,
        vision_hidden_size: int | None = None,
        vision_intermediate_size: int | None = None,
        vision_num_heads: int | None = None,
        vision_in_channels: int | None = None,
        vision_patch_size: int | None = None,
        vision_temporal_patch_size: int | None = None,
        vision_spatial_merge_size: int | None = None,
        vision_num_position_embeddings: int | None = None,
        vision_output_hidden_size: int | None = None,
        vision_hidden_act: str | None = None,
        vision_deepstack_visual_indexes: list[int] | None = None,
        image_token_id: int | None = None,
        video_token_id: int | None = None,
        vision_start_token_id: int | None = None,
        vision_end_token_id: int | None = None,
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
        self.vision_depth = vision_depth
        self.vision_hidden_size = vision_hidden_size
        self.vision_intermediate_size = vision_intermediate_size
        self.vision_num_heads = vision_num_heads
        self.vision_in_channels = vision_in_channels
        self.vision_patch_size = vision_patch_size
        self.vision_temporal_patch_size = vision_temporal_patch_size
        self.vision_spatial_merge_size = vision_spatial_merge_size
        self.vision_num_position_embeddings = vision_num_position_embeddings
        self.vision_output_hidden_size = vision_output_hidden_size
        self.vision_hidden_act = vision_hidden_act
        self.vision_deepstack_visual_indexes = list(vision_deepstack_visual_indexes or [])
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

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
        if (
            self.mapping.tp_size not in (1, 2)
            or self.mapping.pp_size != 1
            or self.mapping.cp_size != 1
        ):
            raise ValueError(
                "The initial Qwen3.5 implementation only supports TP=1 or TP=2, PP=1, and CP=1"
            )
        tp_partitioned_dimensions = {
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
        }
        for name, dimension in tp_partitioned_dimensions.items():
            if dimension % self.mapping.tp_size != 0:
                raise ValueError(
                    f"{name} must be divisible by TP size, got {dimension} and "
                    f"TP={self.mapping.tp_size}"
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
        if len(self.mrope_section) != 3:
            raise ValueError(
                f"Qwen3.5 mrope_section must contain temporal, height, and width sizes, got {self.mrope_section}"
            )
        if sum(self.mrope_section) * 2 != self.rotary_embedding_dim:
            raise ValueError(
                "Twice the sum of mrope_section must equal rotary_embedding_dim, "
                f"got {self.mrope_section} and {self.rotary_embedding_dim}"
            )
        if self.has_vision:
            required_vision_fields = {
                "vision_hidden_size": self.vision_hidden_size,
                "vision_intermediate_size": self.vision_intermediate_size,
                "vision_num_heads": self.vision_num_heads,
                "vision_in_channels": self.vision_in_channels,
                "vision_patch_size": self.vision_patch_size,
                "vision_temporal_patch_size": self.vision_temporal_patch_size,
                "vision_spatial_merge_size": self.vision_spatial_merge_size,
                "vision_num_position_embeddings": self.vision_num_position_embeddings,
                "vision_output_hidden_size": self.vision_output_hidden_size,
                "vision_hidden_act": self.vision_hidden_act,
            }
            missing_vision_fields = [
                name for name, value in required_vision_fields.items() if value is None
            ]
            if missing_vision_fields:
                raise ValueError(
                    f"A Qwen3.5 vision config is missing required fields: {missing_vision_fields}"
                )
            if self.vision_hidden_size % self.vision_num_heads != 0:
                raise ValueError(
                    "vision_hidden_size must be divisible by vision_num_heads, "
                    f"got {self.vision_hidden_size} and {self.vision_num_heads}"
                )
            if self.vision_output_hidden_size != self.hidden_size:
                raise ValueError(
                    "vision_output_hidden_size must match the text hidden_size, "
                    f"got {self.vision_output_hidden_size} and {self.hidden_size}"
                )

    @property
    def has_vision(self) -> bool:
        """Whether this config contains a Qwen3.5 vision tower."""
        return self.vision_depth is not None

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

        multimodal_config = hf_config
        vision_config = getattr(multimodal_config, "vision_config", None)
        if hasattr(multimodal_config, "text_config"):
            hf_config = multimodal_config.text_config
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
            vision_depth=getattr(vision_config, "depth", None),
            vision_hidden_size=getattr(vision_config, "hidden_size", None),
            vision_intermediate_size=getattr(vision_config, "intermediate_size", None),
            vision_num_heads=getattr(vision_config, "num_heads", None),
            vision_in_channels=getattr(vision_config, "in_channels", None),
            vision_patch_size=getattr(vision_config, "patch_size", None),
            vision_temporal_patch_size=getattr(vision_config, "temporal_patch_size", None),
            vision_spatial_merge_size=getattr(vision_config, "spatial_merge_size", None),
            vision_num_position_embeddings=getattr(vision_config, "num_position_embeddings", None),
            vision_output_hidden_size=getattr(vision_config, "out_hidden_size", None),
            vision_hidden_act=getattr(vision_config, "hidden_act", None),
            vision_deepstack_visual_indexes=getattr(
                vision_config, "deepstack_visual_indexes", None
            ),
            image_token_id=getattr(multimodal_config, "image_token_id", None),
            video_token_id=getattr(multimodal_config, "video_token_id", None),
            vision_start_token_id=getattr(multimodal_config, "vision_start_token_id", None),
            vision_end_token_id=getattr(multimodal_config, "vision_end_token_id", None),
            mapping=mapping,
            quantization=quant_config,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            **kwargs,
        )
