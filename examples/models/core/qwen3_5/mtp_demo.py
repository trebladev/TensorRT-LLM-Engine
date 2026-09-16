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

"""Build two TensorRT engines and compare native K=1 MTP with target greedy decoding."""

import argparse
import gc
import json
from pathlib import Path

import tensorrt as trt
import torch
from transformers import AutoTokenizer

from tensorrt_llm import Builder
from tensorrt_llm.models.qwen35.mtp import Qwen35MTP
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import Session

from .mtp import Qwen35MTPGenerator, Qwen35MTPSession
from .target_verification_demo import build_session


def build_draft_engine(
    model: Qwen35MTP,
    max_seq_len: int,
    max_batch_size: int = 1,
    *,
    last_token_logits: bool = False,
    paged_kv_cache: bool = False,
    tokens_per_block: int = 32,
    max_draft_len: int = 1,
) -> trt.IHostMemory:
    """Build a draft engine with continuous or persistent paged attention KV."""
    builder = Builder()
    builder_config = builder.create_builder_config(precision="bfloat16", strongly_typed=True)
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    builder_config.trt_builder_config.builder_optimization_level = 0
    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    network.plugin_config.gpt_attention_plugin = "bfloat16"
    network.plugin_config.gemm_plugin = "bfloat16"
    network.plugin_config.mamba_conv1d_plugin = "bfloat16"
    network.plugin_config.remove_input_padding = True
    network.plugin_config.paged_kv_cache = paged_kv_cache
    if paged_kv_cache:
        network.plugin_config.tokens_per_block = tokens_per_block
    network.plugin_config.paged_state = True
    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        model(
            last_token_logits=last_token_logits,
            **model.prepare_inputs(
                max_batch_size=max_batch_size,
                max_input_len=max_seq_len,
                max_seq_len=max_seq_len,
                max_num_tokens=max_seq_len * max_batch_size,
                opt_num_tokens=min(64, max_seq_len) * max_batch_size,
                use_cache=True,
                max_draft_len=max_draft_len,
                speculative_decoding_draft_tokens_external=True,
            ),
        )
    engine = builder.build_engine(network, builder_config)
    if engine is None:
        raise RuntimeError("Failed to build the MTP draft engine")
    return engine


def save_paged_draft_engine(
    model: Qwen35MTP,
    path: Path,
    max_seq_len: int,
    max_batch_size: int,
    tokens_per_block: int = 32,
    max_draft_len: int = 1,
) -> None:
    """Save the native worker's paged engine and its cache allocation geometry."""
    engine = build_draft_engine(
        model,
        max_seq_len,
        max_batch_size,
        last_token_logits=True,
        paged_kv_cache=True,
        tokens_per_block=tokens_per_block,
        max_draft_len=max_draft_len,
    )
    path.write_bytes(bytes(engine))
    Path(str(path) + ".json").write_text(
        json.dumps(
            {
                "version": 1,
                "dtype": "bfloat16",
                "num_kv_heads": model.config.num_key_value_heads,
                "head_size": model.config.head_size,
                "tokens_per_block": tokens_per_block,
                "max_seq_len": max_seq_len,
                "max_batch_size": max_batch_size,
            },
            indent=2,
        )
        + "\n"
    )


def build_draft_session(model: Qwen35MTP, max_seq_len: int) -> Qwen35MTPSession:
    """Build a continuous-KV draft Session for one persistent request."""
    engine = build_draft_engine(model, max_seq_len)
    return Qwen35MTPSession(Session.from_serialized_engine(engine), model.config, max_seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max_new_tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    capacity = len(prompt) + args.max_new_tokens + 1
    # Fail on absent/incompatible MTP weights before building the target.
    model = Qwen35MTP.from_hugging_face(args.model_dir)
    draft = build_draft_session(model, capacity)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    target = build_session(args.model_dir, capacity, capture_mtp_hidden_states=True)
    generator = Qwen35MTPGenerator(target, draft)
    eos = () if tokenizer.eos_token_id is None else (tokenizer.eos_token_id,)
    result = generator.generate(prompt, args.max_new_tokens, eos)
    target.reset()
    greedy = target.prefill([prompt]).tolist()
    while len(greedy) < args.max_new_tokens and greedy[-1] not in eos:
        greedy.extend(target.decode().tolist())
    if result.tokens != greedy:
        raise AssertionError(f"MTP differs from target greedy: {result.tokens} versus {greedy}")
    print(tokenizer.decode(result.tokens, skip_special_tokens=True))
    print(
        f"Matched {len(result.tokens)} greedy tokens; "
        f"accepted {result.accepted_drafts}/{result.verified_drafts} native MTP candidates"
    )


if __name__ == "__main__":
    main()
