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

"""Build a Session reference engine and compare external-draft and greedy output.

This uses the full dense text model, BF16/TP=1, continuous attention KV, and
paged recurrent snapshots. The candidates are placeholders, not an MTP model.
"""

import argparse
import gc
from pathlib import Path

import tensorrt as trt
import torch
from target_verification import Qwen35VerificationSession
from transformers import AutoTokenizer

from tensorrt_llm import Builder
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import Session


def build_session(model_dir: Path, max_seq_len: int) -> Qwen35VerificationSession:
    """Build directly to retain continuous KV for the correctness reference."""
    model = Qwen35ForCausalLM.from_hugging_face(model_dir, dtype="bfloat16")
    config = model.config
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
    network.plugin_config.paged_kv_cache = False
    network.plugin_config.paged_state = True
    with net_guard(network):
        network.set_named_parameters(model.named_parameters())
        model(
            **model.prepare_inputs(
                max_batch_size=1,
                max_input_len=max_seq_len,
                max_seq_len=max_seq_len,
                max_num_tokens=max_seq_len,
                opt_num_tokens=min(64, max_seq_len),
                use_cache=True,
                max_draft_len=1,
                speculative_decoding_draft_tokens_external=True,
            )
        )
    engine = builder.build_engine(network, builder_config)
    if engine is None:
        raise RuntimeError("Failed to build the verification engine")
    del model, network, builder_config, builder
    gc.collect()
    torch.cuda.empty_cache()
    session = Session.from_serialized_engine(engine)
    return Qwen35VerificationSession(session, config, 1, max_seq_len)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument(
        "--draft_token_id",
        type=int,
        default=None,
        help="External constant candidate; defaults to repeating the pending token",
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max_new_tokens must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    runner = build_session(args.model_dir, len(prompt) + args.max_new_tokens + 1)
    emitted = runner.prefill([prompt]).tolist()
    accepted, iterations = 0, 0
    while len(emitted) < args.max_new_tokens and emitted[-1] != tokenizer.eos_token_id:
        if args.max_new_tokens - len(emitted) == 1:
            emitted.extend(runner.decode().tolist())
            break
        draft = (
            runner.current_tokens
            if args.draft_token_id is None
            else torch.tensor([args.draft_token_id], dtype=torch.int64)
        )
        result = runner.step(draft)
        accepted += int(result.accepted_draft.item())
        iterations += 1
        for token in result.tokens[0].tolist():
            if token >= 0:
                emitted.append(token)
                if token == tokenizer.eos_token_id:
                    break

    runner.reset()
    greedy = runner.prefill([prompt]).tolist()
    while len(greedy) < args.max_new_tokens and greedy[-1] != tokenizer.eos_token_id:
        greedy.extend(runner.decode().tolist())
    if emitted != greedy:
        raise AssertionError(f"Verification differs from greedy: {emitted} versus {greedy}")
    print(tokenizer.decode(emitted, skip_special_tokens=True))
    print(f"Matched greedy output ({len(emitted)} tokens); accepted {accepted}/{iterations} drafts")


if __name__ == "__main__":
    main()
