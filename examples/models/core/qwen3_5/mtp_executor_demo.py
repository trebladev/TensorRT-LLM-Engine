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

"""Build and run native K=1 MTP entirely inside the C++ executor."""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM
from tensorrt_llm.models.qwen35.mtp import Qwen35MTP
from tensorrt_llm.runtime import ModelRunnerCpp

from .mtp_demo import build_draft_engine


def build_engines(model_dir: Path, engine_dir: Path, max_seq_len: int = 128) -> Path:
    """Save a paged-KV target and a continuous-KV MTP draft engine."""
    engine_dir.mkdir(parents=True, exist_ok=True)
    model = Qwen35MTP.from_hugging_face(model_dir)
    draft_path = engine_dir / "mtp.engine"
    draft_path.write_bytes(bytes(build_draft_engine(model, max_seq_len + 1)))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = Qwen35ForCausalLM.from_hugging_face(model_dir, dtype="bfloat16")
    model.capture_mtp_hidden_states = True
    config = BuildConfig(
        max_batch_size=1,
        max_input_len=max_seq_len - 2,
        max_seq_len=max_seq_len,
        max_num_tokens=max_seq_len,
        opt_num_tokens=min(64, max_seq_len),
        max_draft_len=1,
        speculative_decoding_mode=SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL,
    )
    config.plugin_config.gpt_attention_plugin = "bfloat16"
    config.plugin_config.gemm_plugin = "bfloat16"
    config.plugin_config.mamba_conv1d_plugin = "bfloat16"
    build(model, config).save(engine_dir)
    return draft_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--engine_dir", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max_new_tokens", type=int, default=24)
    args = parser.parse_args()
    if args.build:
        build_engines(args.model_dir, args.engine_dir)
        gc.collect()
        torch.cuda.empty_cache()
    runner = ModelRunnerCpp.from_dir(
        str(args.engine_dir),
        max_batch_size=1,
        cuda_graph_mode=False,
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        kv_cache_free_gpu_memory_fraction=0.1,
        mtp_draft_engine_path=str(args.engine_dir / "mtp.engine"),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt = tokenizer.encode(args.prompt, add_special_tokens=True)
    outputs = runner.generate(
        [torch.tensor(prompt, dtype=torch.int32)],
        max_new_tokens=args.max_new_tokens,
        top_k=1,
        end_id=tokenizer.eos_token_id,
        pad_id=0,
        return_dict=True,
        output_sequence_lengths=True,
    )
    length = int(outputs["sequence_lengths"][0, 0])
    tokens = outputs["output_ids"][0, 0, len(prompt) : length].tolist()
    print(tokenizer.decode(tokens, skip_special_tokens=True))
    print(f"C++ executor generated {len(tokens)} tokens with automatic native MTP drafting")


if __name__ == "__main__":
    main()
