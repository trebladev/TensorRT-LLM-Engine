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

import argparse
import time
from pathlib import Path

import tensorrt_llm
from tensorrt_llm._deprecation import emit_engine_arch_deprecation
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a dense Qwen3.5 Hugging Face checkpoint to TensorRT-LLM format."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="The path to the Hugging Face Qwen3.5 checkpoint.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
        help="N-way tensor parallelism size. Qwen3.5 supports 1 or 2.",
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
        help="N-way pipeline parallelism size. The initial implementation only supports 1.",
    )
    parser.add_argument(
        "--cp_size",
        type=int,
        default=1,
        help="N-way context parallelism size. The initial implementation only supports 1.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16"],
        help="The model weight and activation dtype. Qwen3.5 currently only supports BF16.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("tllm_checkpoint"),
        help="The path to save the TensorRT-LLM checkpoint.",
    )
    parser.add_argument("--log_level", type=str, default="info")
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise ValueError(f"The model directory does not exist: {args.model_dir}")
    if args.tp_size not in (1, 2) or args.pp_size != 1 or args.cp_size != 1:
        raise ValueError(
            "Qwen3.5 checkpoint conversion only supports TP=1 or TP=2, PP=1, and CP=1, "
            f"got TP={args.tp_size}, PP={args.pp_size}, and CP={args.cp_size}."
        )


def convert_and_save_hf(args: argparse.Namespace) -> None:
    world_size = args.tp_size
    for rank in range(world_size):
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            tp_size=args.tp_size,
            pp_size=args.pp_size,
            cp_size=args.cp_size,
        )
        model = Qwen35ForCausalLM.from_hugging_face(
            args.model_dir,
            dtype=args.dtype,
            mapping=mapping,
            quant_config=QuantConfig(),
        )
        model.save_checkpoint(args.output_dir, save_config=(rank == 0))
        del model
        release_gc()


def main() -> None:
    emit_engine_arch_deprecation("convert_checkpoint.py")
    print(tensorrt_llm.__version__)

    args = parse_arguments()
    logger.set_level(args.log_level)
    _validate_arguments(args)
    args.output_dir.mkdir(exist_ok=True, parents=True)

    start_time = time.time()
    convert_and_save_hf(args)
    elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"Total time of converting checkpoints: {elapsed}")


if __name__ == "__main__":
    main()
