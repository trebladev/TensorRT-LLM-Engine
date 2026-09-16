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

"""Dense Qwen3.5 single-layer MTP draft graph with independent attention KV."""

from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoConfig

from ..._common import default_net
from ...functional import Tensor, concat, gather_last_token_logits
from ...layers import Attention, Linear, RmsNorm
from ..convert_utils import iterate_shard_files, load_state_dict
from ..modeling_utils import PretrainedModel
from .config import Qwen35Config
from .convert import _convert_parameter, _convert_zero_centered_norm, _normalize_hf_name
from .model import Qwen35ForCausalLM


def convert_mtp_weights(
    state_dict: dict[str, torch.Tensor], config: Qwen35Config
) -> dict[str, torch.Tensor]:
    """Convert shared embeddings/head and native ``mtp.*`` checkpoint weights.

    Gemma-style zero-centered norms are converted in float32 before BF16
    rounding, exactly as for the target's attention and MLP layers.
    """
    if config.mapping.world_size != 1:
        raise ValueError("Qwen3.5 MTP currently requires TP=PP=CP=1")
    result = {}
    norms = {
        "mtp.norm.weight": "transformer.ln_f.weight",
        "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
    }
    for original_name, weight in state_dict.items():
        name = _normalize_hf_name(original_name)
        if name in norms:
            result[norms[name]] = _convert_zero_centered_norm(weight)
        elif name == "mtp.fc.weight":
            result["fc.weight"] = weight.to(device="cpu", dtype=torch.bfloat16).contiguous()
        elif name.startswith("mtp.layers.0."):
            result.update(_convert_parameter(name.replace("mtp.", "model.", 1), weight, config))
        elif name in ("model.embed_tokens.weight", "lm_head.weight"):
            result.update(_convert_parameter(name, weight, config))
    return result


class Qwen35MTP(Qwen35ForCausalLM):
    """Autoregressive BF16 draft engine; inputs pair token t+1 with target hidden state t.

    Target states are taken after the target's final normalization. MTP has
    its own pre-FC norms, one full-attention decoder, and final normalization.
    Embeddings and the output head use the target checkpoint's shared weights.
    """

    def __init__(self, config: Qwen35Config) -> None:
        if config.mapping.world_size != 1:
            raise ValueError("Qwen3.5 MTP currently requires TP=PP=CP=1")
        config = deepcopy(config)
        config.num_hidden_layers = 1
        config.decoder_layer_types = ["full_attention"]
        config.layer_types = ["attention"]
        config.vision_depth = None
        super().__init__(config)
        self.pre_fc_norm_embedding = RmsNorm(
            config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.pre_fc_norm_hidden = RmsNorm(
            config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.fc = Linear(config.hidden_size * 2, config.hidden_size, bias=False, dtype=config.dtype)

    def forward(
        self,
        input_ids: Tensor,
        target_hidden_states: Tensor,
        last_token_logits: bool = False,
        **kwargs,
    ) -> Tensor:
        """Process packed, causally ordered token/hidden-state pairs with KV caching."""
        if not kwargs.get("use_cache", True):
            raise ValueError("MTP requires use_cache=true")
        hidden_states = self.fc(
            concat(
                [
                    self.pre_fc_norm_embedding(self.transformer.vocab_embedding(input_ids)),
                    self.pre_fc_norm_hidden(target_hidden_states),
                ],
                dim=-1,
            )
        )
        attention_params = Attention.fill_attention_params(self, kwargs["attention_params"])
        hidden_states, present, _, _ = self.transformer.layers[0](
            hidden_states,
            use_cache=True,
            attention_mask=kwargs.get("attention_mask"),
            kv_cache_params=kwargs["kv_cache_params"],
            attention_params=attention_params,
            mrope_params=kwargs["mrope_params"],
            spec_decoding_params=kwargs.get("spec_decoding_params"),
        )
        hidden_states = self.transformer.ln_f(hidden_states)
        hidden_states.mark_output("mtp_hidden_states", self.dtype)
        if last_token_logits:
            if kwargs.get("last_token_ids") is None:
                raise ValueError("Last-token MTP projection requires last_token_ids")
            # Preserve every position's attention/KV update; only the output
            # projection needs the last valid row, before trailing padding.
            hidden_states = gather_last_token_logits(
                hidden_states,
                kwargs["last_token_ids"],
                default_net().plugin_config.remove_input_padding,
            )
        logits = self.lm_head(hidden_states)
        # An explicit output name lets the worker also accept older engines
        # whose logits still contain every packed token row.
        logits.mark_output(
            "last_token_logits" if last_token_logits else "logits", self._logits_dtype
        )
        if not default_net().plugin_config.paged_kv_cache:
            present.mark_output("present_key_value_0", self.config.kv_dtype)
        return logits

    def prepare_inputs(self, *args, **kwargs) -> dict:
        """Use the target's verification profiles, with an additional packed hidden input."""
        if not 1 <= kwargs.get("max_draft_len", 0) <= 30:
            raise ValueError("MTP requires 1 to 30 draft tokens in engine profiles")
        kwargs["speculative_decoding_draft_tokens_external"] = False
        kwargs["num_hidden_layers"] = 1
        kwargs["mrope_rotary_cos_sin_size"] = (
            self.config.max_position_embeddings * self.config.rotary_embedding_dim
        )
        result = PretrainedModel.prepare_inputs(self, *args, **kwargs)
        profiles = result["input_ids"].profiles
        result["target_hidden_states"] = Tensor(
            name="target_hidden_states",
            dtype=self.dtype,
            shape=[-1, self.config.hidden_size],
            dim_range=OrderedDict(
                num_tokens=[[p.min[0], p.opt[0], p.max[0]] for p in profiles],
                hidden_size=[self.config.hidden_size] * len(profiles),
            ),
        )
        return result

    @classmethod
    def from_hugging_face(cls, hf_model_or_dir: str | Path, **kwargs) -> "Qwen35MTP":
        """Load one native MTP layer, failing if required weights are absent."""
        hf_config = AutoConfig.from_pretrained(hf_model_or_dir)
        text_config = getattr(hf_config, "text_config", hf_config)
        if getattr(text_config, "mtp_num_hidden_layers", 0) != 1:
            raise ValueError("Qwen3.5 MTP requires exactly one native MTP layer")
        if getattr(text_config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated MTP embeddings are not supported")
        config = Qwen35Config.from_hugging_face(hf_config, **kwargs)
        model = cls(config)
        weights = {}
        for shard in iterate_shard_files(str(hf_model_or_dir), rank=0, progress_bar=False):
            weights.update(convert_mtp_weights(load_state_dict(shard), config))
        if "lm_head.weight" not in weights and config.tie_word_embeddings:
            weights["lm_head.weight"] = weights["transformer.vocab_embedding.weight"]
        model.load(weights)
        return model
