# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Qwen3.5 TensorRT vision engine without running LLM inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from examples.models.core.qwen3_5 import multimodal_demo as demo
from tensorrt_llm.models.qwen35.config import Qwen35Config


def parse_arguments() -> argparse.Namespace:
    """Parse standalone vision-engine build arguments."""
    parser = argparse.ArgumentParser(
        description="Build a Qwen3.5 TensorRT vision engine for a token profile."
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--vision_engine_dir", type=Path, required=True)
    parser.add_argument("--min_vision_tokens", type=int, required=True)
    parser.add_argument("--opt_vision_tokens", type=int, required=True)
    parser.add_argument("--max_vision_tokens", type=int, required=True)
    parser.add_argument("--vision_workspace_gb", type=int, default=12)
    parser.add_argument("--vision_builder_optimization_level", type=int, default=0)
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate filesystem paths and vision build limits."""
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Hugging Face model directory does not exist: {args.model_dir}")
    token_profile = (
        args.min_vision_tokens,
        args.opt_vision_tokens,
        args.max_vision_tokens,
    )
    if any(token_count <= 0 for token_count in token_profile):
        raise ValueError("vision token profile values must be positive")
    if not args.min_vision_tokens <= args.opt_vision_tokens <= args.max_vision_tokens:
        raise ValueError("vision token profile must satisfy min <= opt <= max")
    if args.vision_workspace_gb <= 0:
        raise ValueError("vision_workspace_gb must be positive")
    if not 0 <= args.vision_builder_optimization_level <= 5:
        raise ValueError("vision_builder_optimization_level must be between 0 and 5")


def main() -> None:
    """Build and serialize the vision engine for the supplied token profile."""
    args = parse_arguments()
    validate_arguments(args)

    hf_config = AutoConfig.from_pretrained(args.model_dir)
    config = Qwen35Config.from_hugging_face(hf_config)
    if not config.has_vision:
        raise ValueError("The supplied Qwen3.5 model does not contain a vision tower")

    minimum_supported_tokens = config.vision_spatial_merge_size**2
    if args.min_vision_tokens < minimum_supported_tokens:
        raise ValueError(
            f"min_vision_tokens must be at least {minimum_supported_tokens} for this model"
        )
    profile = (
        args.min_vision_tokens,
        args.opt_vision_tokens,
        args.max_vision_tokens,
    )
    print(f"Vision token profile (min, opt, max): {profile}")
    print(f"Vision engine directory: {args.vision_engine_dir}")

    hf_model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        config=hf_config,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )
    engine_path = demo._build_vision_engine(
        args.vision_engine_dir,
        config,
        hf_model,
        profile,
        args.vision_workspace_gb,
        args.vision_builder_optimization_level,
    )
    print(f"Vision engine built: {engine_path}")


if __name__ == "__main__":
    main()
