# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import tensorrt as trt
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
from transformers.video_utils import VideoMetadata

from tensorrt_llm import Builder
from tensorrt_llm._common import serialize_engine
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.functional import Tensor
from tensorrt_llm.inputs.multimodal import apply_mm_hashes, hexdigest_to_int32
from tensorrt_llm.inputs.multimodal_data import VideoData
from tensorrt_llm.layers.attention import MropeParams
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.convert import convert_hf_qwen35
from tensorrt_llm.models.qwen35.model import Qwen35VisionModel
from tensorrt_llm.models.qwen35.vision_utils import (
    prepare_qwen35_executor_prompt_inputs,
    prepare_qwen35_mrope_inputs,
    prepare_qwen35_multimodal_cache_input,
    prepare_qwen35_vision_position_inputs,
)
from tensorrt_llm.network import net_guard
from tensorrt_llm.runtime import ModelRunnerCpp, Session
from tensorrt_llm.runtime.session import TensorInfo

_VISION_ENGINE_NAME = "rank0.engine"
_VISION_CONFIG_NAME = "config.json"
_DEFAULT_MIN_PIXELS = 4 * 28 * 28
_DEFAULT_MAX_PIXELS = 256 * 28 * 28
_DEFAULT_VIDEO_NUM_FRAMES = 0


@dataclass(frozen=True)
class Comparison:
    cosine_similarity: float
    mean_absolute_error: float
    max_absolute_error: float


@dataclass(frozen=True)
class DecodedVideo:
    frames: torch.Tensor
    metadata: VideoMetadata


@dataclass(frozen=True)
class CacheObservation:
    request_reused_blocks: int
    request_hit_rate: float
    cumulative_reused_blocks: int
    cumulative_hit_rate: float
    tokens_per_block: int


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an image or video through the Qwen3.5 TensorRT vision and LLM engines, "
            "then compare vision embeddings and context logits with Hugging Face."
        )
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--llm_engine_dir", type=Path, required=True)
    parser.add_argument("--vision_engine_dir", type=Path, required=True)
    media_group = parser.add_mutually_exclusive_group(required=True)
    media_group.add_argument("--image", type=Path)
    media_group.add_argument("--video", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_pixels", type=int, default=_DEFAULT_MIN_PIXELS)
    parser.add_argument("--max_pixels", type=int, default=_DEFAULT_MAX_PIXELS)
    parser.add_argument(
        "--video_num_frames",
        type=int,
        default=_DEFAULT_VIDEO_NUM_FRAMES,
        help="Number of frames to sample uniformly with ffmpeg; use 0 to decode every frame.",
    )
    parser.add_argument(
        "--max_vision_tokens",
        type=int,
        default=None,
        help=(
            "Maximum pre-merge patch-token count accepted by a newly built vision engine. "
            "The current input token count is used when omitted."
        ),
    )
    parser.add_argument(
        "--rebuild_vision_engine",
        action="store_true",
        help="Rebuild the cached vision engine even if rank0.engine already exists.",
    )
    parser.add_argument("--vision_workspace_gb", type=int, default=12)
    parser.add_argument("--vision_builder_optimization_level", type=int, default=0)
    parser.add_argument("--kv_cache_free_gpu_memory_fraction", type=float, default=0.1)
    parser.add_argument(
        "--kv_cache_enable_block_reuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable content-aware LLM prefix-cache reuse for the visual prompt.",
    )
    parser.add_argument(
        "--prefix_cache_requests",
        type=int,
        default=2,
        help="Number of identical LLM requests to issue when block reuse is enabled.",
    )
    parser.add_argument(
        "--enable_chunked_context",
        action="store_true",
        help="Split long prompts across multiple context-phase executor iterations.",
    )
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when the configured HF consistency thresholds are not met.",
    )
    parser.add_argument("--vision_cosine_threshold", type=float, default=0.99)
    parser.add_argument("--logits_cosine_threshold", type=float, default=0.99)
    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"Hugging Face model directory does not exist: {args.model_dir}")
    if not args.llm_engine_dir.is_dir():
        raise FileNotFoundError(f"LLM engine directory does not exist: {args.llm_engine_dir}")
    if args.image is not None and not args.image.is_file():
        raise FileNotFoundError(f"Image does not exist: {args.image}")
    if args.video is not None and not args.video.is_file():
        raise FileNotFoundError(f"Video does not exist: {args.video}")
    if args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if args.min_pixels <= 0 or args.max_pixels <= 0:
        raise ValueError("min_pixels and max_pixels must be positive")
    if args.min_pixels > args.max_pixels:
        raise ValueError("min_pixels must not exceed max_pixels")
    if args.video_num_frames < 0:
        raise ValueError("video_num_frames must be non-negative")
    if args.max_vision_tokens is not None and args.max_vision_tokens <= 0:
        raise ValueError("max_vision_tokens must be positive")
    if args.vision_workspace_gb <= 0:
        raise ValueError("vision_workspace_gb must be positive")
    if not 0 <= args.vision_builder_optimization_level <= 5:
        raise ValueError("vision_builder_optimization_level must be between 0 and 5")
    if not 0 < args.kv_cache_free_gpu_memory_fraction <= 1:
        raise ValueError("kv_cache_free_gpu_memory_fraction must be in (0, 1]")
    if args.prefix_cache_requests <= 0:
        raise ValueError("prefix_cache_requests must be positive")
    if args.kv_cache_enable_block_reuse and args.prefix_cache_requests < 2:
        raise ValueError("prefix_cache_requests must be at least 2 when block reuse is enabled")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive")


def _validate_processor_inputs(
    inputs: dict[str, object],
    required_inputs: set[str],
    modality: str,
) -> dict[str, torch.Tensor]:
    tensor_inputs = {
        name: value for name, value in inputs.items() if isinstance(value, torch.Tensor)
    }
    missing_inputs = sorted(required_inputs - tensor_inputs.keys())
    if missing_inputs:
        raise ValueError(
            f"The Qwen3.5 processor did not return the required {modality} inputs: "
            f"{missing_inputs}. Returned keys: {sorted(inputs.keys())}"
        )
    if tensor_inputs["input_ids"].shape[0] != 1:
        raise ValueError("This Qwen3.5 multimodal demo supports batch size 1")
    return tensor_inputs


def _prepare_image_processor_inputs(
    processor: AutoProcessor,
    image: Image.Image,
    prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return _validate_processor_inputs(
        inputs,
        {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values",
            "image_grid_thw",
        },
        "image",
    )


def _parse_frame_rate(value: str | None) -> float | None:
    if value in (None, "", "N/A", "0/0"):
        return None
    rate = Fraction(value)
    if rate <= 0:
        return None
    return float(rate)


def _probe_video(video_path: Path) -> tuple[int, int, int, float, float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffprobe is required to run the Qwen3.5 video demo") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"ffprobe failed for {video_path}: {error.stderr.strip()}") from error

    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"Expected one selected video stream in {video_path}, got {len(streams)}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    frame_rate = _parse_frame_rate(stream.get("avg_frame_rate"))
    if frame_rate is None:
        frame_rate = _parse_frame_rate(stream.get("r_frame_rate"))
    if frame_rate is None:
        raise ValueError(f"Could not determine the frame rate of {video_path}")

    duration_value = metadata.get("format", {}).get("duration") or stream.get("duration")
    duration = float(duration_value) if duration_value not in (None, "N/A") else 0.0
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value in (None, "N/A"):
        if duration <= 0:
            raise ValueError(f"Could not determine the frame count of {video_path}")
        frame_count = max(1, round(duration * frame_rate))
    else:
        frame_count = int(frame_count_value)
    if frame_count <= 0:
        raise ValueError(f"Video contains no decodable frames: {video_path}")
    if duration <= 0:
        duration = frame_count / frame_rate
    return width, height, frame_count, frame_rate, duration


def _sample_frame_indices(frame_count: int, requested_frames: int) -> list[int]:
    if frame_count <= 0 or requested_frames < 0:
        raise ValueError("frame_count must be positive and requested_frames must be non-negative")
    if requested_frames == 0:
        return list(range(frame_count))
    sample_count = min(frame_count, requested_frames)
    if sample_count == 1:
        return [0]
    return [round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)]


def _decode_video_with_ffmpeg(video_path: Path, requested_frames: int) -> DecodedVideo:
    width, height, frame_count, frame_rate, duration = _probe_video(video_path)
    frame_indices = _sample_frame_indices(frame_count, requested_frames)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
    ]
    if requested_frames > 0:
        select_expression = "+".join(f"eq(n\\,{index})" for index in frame_indices)
        command.extend(["-vf", f"select={select_expression}", "-fps_mode", "passthrough"])
    command.extend(["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
    try:
        result = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is required to run the Qwen3.5 video demo") from error
    except subprocess.CalledProcessError as error:
        stderr = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}") from error

    frame_size = height * width * 3
    expected_size = len(frame_indices) * frame_size
    if len(result.stdout) != expected_size:
        raise RuntimeError(
            f"ffmpeg returned {len(result.stdout)} RGB bytes, expected {expected_size} "
            f"for {len(frame_indices)} frames of size {width}x{height}"
        )
    frames = torch.frombuffer(bytearray(result.stdout), dtype=torch.uint8).reshape(
        len(frame_indices), height, width, 3
    )
    video_metadata = VideoMetadata(
        total_num_frames=frame_count,
        fps=frame_rate,
        width=width,
        height=height,
        duration=duration,
        video_backend="ffmpeg",
        frames_indices=frame_indices,
    )
    print(
        f"Decoded video with ffmpeg: size={width}x{height}, fps={frame_rate:.3f}, "
        f"duration={duration:.3f}s, decoded_frames={len(frame_indices)}/{frame_count}"
    )
    return DecodedVideo(frames=frames, metadata=video_metadata)


def _prepare_video_processor_inputs(
    processor: AutoProcessor,
    video: DecodedVideo,
    prompt: str,
) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video.frames},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={
            "do_sample_frames": False,
            "video_metadata": [video.metadata],
        },
    )
    return _validate_processor_inputs(
        inputs,
        {
            "input_ids",
            "attention_mask",
            "mm_token_type_ids",
            "pixel_values_videos",
            "video_grid_thw",
        },
        "video",
    )


def _video_data_for_hash(video: DecodedVideo) -> VideoData:
    metadata = {
        "total_num_frames": video.metadata.total_num_frames,
        "fps": video.metadata.fps,
        "duration": video.metadata.duration,
        "frames_indices": list(video.metadata.frames_indices),
    }
    return VideoData(frames=list(video.frames.unbind(0)), metadata=metadata)


def _hash_multimodal_content(modality: str, content: object) -> list[int]:
    hashes, _ = apply_mm_hashes({modality: [content]})
    return hexdigest_to_int32(hashes[modality][0])


def _collect_cache_observation(runner: ModelRunnerCpp) -> CacheObservation:
    request_reused_blocks = 0
    request_hit_rate = 0.0
    for per_iteration in runner.session.get_latest_request_stats():
        for request_stats in per_iteration.request_stats:
            request_reused_blocks = max(
                request_reused_blocks,
                int(request_stats.reused_blocks_per_request),
            )
            request_hit_rate = max(
                request_hit_rate,
                float(request_stats.kv_cache_hit_rate_per_request),
            )

    cumulative_reused_blocks = 0
    cumulative_hit_rate = 0.0
    tokens_per_block = 0
    for iteration_stats in runner.session.get_latest_iteration_stats():
        kv_cache_stats = iteration_stats.kv_cache_stats
        cumulative_reused_blocks = int(kv_cache_stats.reused_blocks)
        cumulative_hit_rate = float(kv_cache_stats.cache_hit_rate)
        tokens_per_block = int(kv_cache_stats.tokens_per_block)

    return CacheObservation(
        request_reused_blocks=request_reused_blocks,
        request_hit_rate=request_hit_rate,
        cumulative_reused_blocks=cumulative_reused_blocks,
        cumulative_hit_rate=cumulative_hit_rate,
        tokens_per_block=tokens_per_block,
    )


def _set_eager_attention(hf_config: object) -> None:
    setattr(hf_config, "_attn_implementation", "eager")
    for nested_name in ("text_config", "vision_config"):
        nested_config = getattr(hf_config, nested_name, None)
        if nested_config is not None:
            setattr(nested_config, "_attn_implementation", "eager")


def _load_hf_model(model_dir: Path, hf_config: object) -> torch.nn.Module:
    print("Loading the Hugging Face reference model on CPU...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        config=hf_config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    return model.eval()


def _vision_profile(
    actual_tokens: int,
    max_tokens: int | None,
    spatial_merge_size: int,
) -> tuple[int, int, int]:
    minimum_tokens = spatial_merge_size**2
    if actual_tokens < minimum_tokens:
        raise ValueError(
            f"The visual input produced {actual_tokens} patch tokens, fewer than {minimum_tokens}"
        )
    maximum_tokens = actual_tokens if max_tokens is None else max_tokens
    if maximum_tokens < actual_tokens:
        raise ValueError(
            f"max_vision_tokens={maximum_tokens} is smaller than the current input's "
            f"{actual_tokens} patch tokens"
        )
    return minimum_tokens, actual_tokens, maximum_tokens


def _dynamic_tensor(
    name: str,
    dtype: trt.DataType,
    shape: list[int],
    dimensions: list[tuple[str, list[int] | int]],
) -> Tensor:
    dim_range = OrderedDict((dimension_name, [value]) for dimension_name, value in dimensions)
    return Tensor(name=name, dtype=dtype, shape=shape, dim_range=dim_range)


def _vision_engine_metadata(
    config: Qwen35Config,
    profile: tuple[int, int, int],
) -> dict[str, object]:
    patch_dim = (
        config.vision_in_channels * config.vision_temporal_patch_size * config.vision_patch_size**2
    )
    return {
        "dtype": config.dtype,
        "min_vision_tokens": profile[0],
        "opt_vision_tokens": profile[1],
        "max_vision_tokens": profile[2],
        "patch_dim": patch_dim,
        "vision_hidden_size": config.vision_hidden_size,
        "vision_num_heads": config.vision_num_heads,
        "vision_spatial_merge_size": config.vision_spatial_merge_size,
    }


def _build_vision_engine(
    engine_dir: Path,
    config: Qwen35Config,
    hf_model: torch.nn.Module,
    profile: tuple[int, int, int],
    workspace_gb: int,
    optimization_level: int,
) -> Path:
    print(f"Building the Qwen3.5 vision engine with token profile {profile}...")
    vision_model = Qwen35VisionModel(config)
    hf_visual = hf_model.model.visual
    visual_state_dict = {
        f"model.visual.{name}": parameter for name, parameter in hf_visual.state_dict().items()
    }
    converted_weights = convert_hf_qwen35(visual_state_dict, config)
    missing_weights = []
    for name, parameter in vision_model.named_parameters():
        weight_name = f"visual.{name}"
        if weight_name not in converted_weights:
            missing_weights.append(weight_name)
            continue
        parameter.value = converted_weights[weight_name]
    if missing_weights:
        raise ValueError(f"Missing converted vision weights: {missing_weights}")

    patch_dim = (
        config.vision_in_channels * config.vision_temporal_patch_size * config.vision_patch_size**2
    )
    vision_head_dim = config.vision_hidden_size // config.vision_num_heads
    token_range = list(profile)

    builder = Builder()
    builder_config = builder.create_builder_config(
        name="qwen35_vision",
        precision="bfloat16",
        strongly_typed=True,
    )
    trt_builder_config = builder_config.trt_builder_config
    trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    trt_builder_config.builder_optimization_level = optimization_level
    trt_builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    network = builder.create_network()
    network.plugin_config.gemm_plugin = "bfloat16"

    with net_guard(network):
        network.set_named_parameters(vision_model.named_parameters())
        _, pooled_output = vision_model(
            _dynamic_tensor(
                "pixel_values",
                trt.bfloat16,
                [-1, patch_dim],
                [("vision_tokens", token_range), ("patch_dim", patch_dim)],
            ),
            _dynamic_tensor(
                "position_ids",
                trt.int32,
                [4, -1],
                [("position_corners", 4), ("vision_tokens", token_range)],
            ),
            _dynamic_tensor(
                "position_weights",
                trt.bfloat16,
                [4, -1],
                [("position_corners", 4), ("vision_tokens", token_range)],
            ),
            _dynamic_tensor(
                "rotary_cos",
                trt.bfloat16,
                [-1, vision_head_dim],
                [("vision_tokens", token_range), ("vision_head_dim", vision_head_dim)],
            ),
            _dynamic_tensor(
                "rotary_sin",
                trt.bfloat16,
                [-1, vision_head_dim],
                [("vision_tokens", token_range), ("vision_head_dim", vision_head_dim)],
            ),
            _dynamic_tensor(
                "vision_attention_mask",
                trt.bfloat16,
                [1, 1, -1, -1],
                [
                    ("attention_batch", 1),
                    ("attention_heads", 1),
                    ("vision_tokens", token_range),
                    ("attention_key_tokens", token_range),
                ],
            ),
        )
        pooled_output.mark_output("pooled_output", "bfloat16")

    engine = builder.build_engine(network, builder_config)
    if engine is None:
        raise RuntimeError("TensorRT failed to build the Qwen3.5 vision engine")

    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / _VISION_ENGINE_NAME
    serialize_engine(engine, engine_path)
    metadata = _vision_engine_metadata(config, profile)
    (engine_dir / _VISION_CONFIG_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return engine_path


def _resolve_vision_engine(
    args: argparse.Namespace,
    config: Qwen35Config,
    hf_model: torch.nn.Module,
    actual_tokens: int,
) -> Path:
    engine_path = args.vision_engine_dir / _VISION_ENGINE_NAME
    metadata_path = args.vision_engine_dir / _VISION_CONFIG_NAME
    if args.rebuild_vision_engine or not engine_path.is_file():
        profile = _vision_profile(
            actual_tokens,
            args.max_vision_tokens,
            config.vision_spatial_merge_size,
        )
        return _build_vision_engine(
            args.vision_engine_dir,
            config,
            hf_model,
            profile,
            args.vision_workspace_gb,
            args.vision_builder_optimization_level,
        )

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Vision engine metadata is missing: {metadata_path}. "
            "Run again with --rebuild_vision_engine."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = _vision_engine_metadata(config, (1, 1, 1))
    for name in (
        "dtype",
        "patch_dim",
        "vision_hidden_size",
        "vision_num_heads",
        "vision_spatial_merge_size",
    ):
        if metadata.get(name) != expected_metadata[name]:
            raise ValueError(
                f"Cached vision engine field {name}={metadata.get(name)!r} does not match "
                f"the model value {expected_metadata[name]!r}; rebuild the vision engine"
            )
    maximum_tokens = int(metadata["max_vision_tokens"])
    if actual_tokens > maximum_tokens:
        raise ValueError(
            f"The cached vision engine accepts at most {maximum_tokens} patch tokens, "
            f"but this visual input produced {actual_tokens}; rebuild with "
            f"--max_vision_tokens {actual_tokens} or larger"
        )
    print(f"Loading cached vision engine: {engine_path}")
    return engine_path


def _run_vision_engine(
    engine_path: Path,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    config: Qwen35Config,
) -> tuple[torch.Tensor, Session]:
    session = Session.from_serialized_engine(engine_path.read_bytes())
    position_inputs = prepare_qwen35_vision_position_inputs(
        grid_thw,
        config,
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs = {
        "pixel_values": pixel_values.to(device="cuda", dtype=torch.bfloat16).contiguous(),
        "position_ids": position_inputs.position_ids.contiguous(),
        "position_weights": position_inputs.position_weights.contiguous(),
        "rotary_cos": position_inputs.rotary_cos.contiguous(),
        "rotary_sin": position_inputs.rotary_sin.contiguous(),
        "vision_attention_mask": position_inputs.attention_mask.contiguous(),
    }
    output_info = session.infer_shapes(
        [
            TensorInfo(name, torch_dtype_to_trt(tensor.dtype), tuple(tensor.shape))
            for name, tensor in inputs.items()
        ]
    )
    outputs = {
        info.name: torch.empty(
            tuple(info.shape),
            dtype=trt_dtype_to_torch(info.dtype),
            device="cuda",
        )
        for info in output_info
    }
    stream = torch.cuda.current_stream()
    if not session.run(inputs, outputs, stream.cuda_stream):
        raise RuntimeError("Qwen3.5 vision engine execution failed")
    stream.synchronize()
    return outputs["pooled_output"], session


def _compare(actual: torch.Tensor, expected: torch.Tensor) -> Comparison:
    actual_float = actual.detach().float().cpu()
    expected_float = expected.detach().float().cpu()
    if actual_float.shape != expected_float.shape:
        raise ValueError(
            f"Cannot compare tensors with shapes {tuple(actual_float.shape)} and "
            f"{tuple(expected_float.shape)}"
        )
    difference = (actual_float - expected_float).abs()
    cosine = torch.nn.functional.cosine_similarity(
        actual_float.flatten(), expected_float.flatten(), dim=0
    )
    return Comparison(
        cosine_similarity=float(cosine.item()),
        mean_absolute_error=float(difference.mean().item()),
        max_absolute_error=float(difference.max().item()),
    )


def _print_comparison(name: str, comparison: Comparison) -> None:
    print(
        f"{name}: cosine={comparison.cosine_similarity:.8f}, "
        f"mean_abs_error={comparison.mean_absolute_error:.8f}, "
        f"max_abs_error={comparison.max_absolute_error:.8f}"
    )


def _hf_inputs_on_cuda(processor_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    supported_names = {
        "input_ids",
        "attention_mask",
        "mm_token_type_ids",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
    return {
        name: tensor.to("cuda")
        for name, tensor in processor_inputs.items()
        if name in supported_names
    }


def _token_description(tokenizer: object, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(token_id)
    decoded = tokenizer.decode([token_id], skip_special_tokens=False)
    return f"id={token_id}, token={token!r}, text={decoded!r}"


def _print_top_logits(
    tokenizer: object,
    trt_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    top_k: int,
) -> tuple[int, int, int]:
    top_k = min(top_k, trt_logits.numel())
    trt_ids = torch.topk(trt_logits, top_k).indices.tolist()
    hf_ids = torch.topk(hf_logits, top_k).indices.tolist()
    print("TensorRT top logits:")
    for token_id in trt_ids:
        print(f"  {_token_description(tokenizer, token_id)}")
    print("Hugging Face top logits:")
    for token_id in hf_ids:
        print(f"  {_token_description(tokenizer, token_id)}")
    overlap = len(set(trt_ids) & set(hf_ids))
    print(f"Top-{top_k} overlap: {overlap}/{top_k}")
    return trt_ids[0], hf_ids[0], overlap


def _select_last_context_logits(
    context_logits: torch.Tensor | list[torch.Tensor],
    input_length: int,
    vocab_size: int,
) -> torch.Tensor:
    if isinstance(context_logits, list):
        if len(context_logits) != 1:
            raise ValueError(
                f"This batch-1 demo expected one context-logit tensor, got {len(context_logits)}"
            )
        context_logits = context_logits[0]
    if context_logits.ndim == 3:
        return context_logits[0, input_length - 1, :vocab_size]
    if context_logits.ndim == 2:
        return context_logits[input_length - 1, :vocab_size]
    raise ValueError(
        "context_logits must have shape [batch, sequence, vocab] or [sequence, vocab], "
        f"got {tuple(context_logits.shape)}"
    )


def main() -> None:
    args = parse_arguments()
    _validate_arguments(args)
    if not torch.cuda.is_available():
        raise RuntimeError("The Qwen3.5 multimodal demo requires an NVIDIA GPU")

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    if args.image is not None:
        prompt = args.prompt or "Describe this image in detail."
        with Image.open(args.image) as image_file:
            image = image_file.convert("RGB")
        processor_inputs = _prepare_image_processor_inputs(processor, image, prompt)
        visual_content_hash = (
            _hash_multimodal_content("image", image) if args.kv_cache_enable_block_reuse else None
        )
        modality = "image"
        modality_name = "Image"
        pixel_values_name = "pixel_values"
        grid_name = "image_grid_thw"
    else:
        prompt = args.prompt or "总结一下这段视频"
        decoded_video = _decode_video_with_ffmpeg(args.video, args.video_num_frames)
        processor_inputs = _prepare_video_processor_inputs(processor, decoded_video, prompt)
        visual_content_hash = (
            _hash_multimodal_content("video", _video_data_for_hash(decoded_video))
            if args.kv_cache_enable_block_reuse
            else None
        )
        modality = "video"
        modality_name = "Video"
        pixel_values_name = "pixel_values_videos"
        grid_name = "video_grid_thw"
        del decoded_video

    hf_config = AutoConfig.from_pretrained(args.model_dir)
    _set_eager_attention(hf_config)
    config = Qwen35Config.from_hugging_face(hf_config)
    if not config.has_vision:
        raise ValueError("The supplied Qwen3.5 model does not contain a vision tower")

    multimodal_cache_input = None
    if visual_content_hash is not None:
        multimodal_cache_input = prepare_qwen35_multimodal_cache_input(
            processor_inputs["input_ids"],
            processor_inputs["attention_mask"],
            config,
            modality,
            visual_content_hash,
        )
        print(
            "Multimodal prefix-cache spans: "
            f"positions={multimodal_cache_input.multimodal_positions}, "
            f"lengths={multimodal_cache_input.multimodal_lengths}"
        )

    grid_thw = processor_inputs[grid_name]
    pixel_values = processor_inputs[pixel_values_name]
    actual_patch_tokens = int(grid_thw.prod(dim=-1).sum().item())
    if pixel_values.shape[0] != actual_patch_tokens:
        raise ValueError(
            f"Processor pixel values and {modality_name.lower()} grid disagree: "
            f"{pixel_values.shape[0]} and {actual_patch_tokens}"
        )
    print(f"{modality_name} grid (T, H, W): {grid_thw.tolist()}")
    print(f"Vision patch tokens: {actual_patch_tokens}")

    # MRoPE must be prepared while the original visual placeholder IDs are still present.
    if args.image is not None:
        mrope_inputs = prepare_qwen35_mrope_inputs(
            processor_inputs["input_ids"],
            config,
            attention_mask=processor_inputs["attention_mask"],
            mm_token_type_ids=processor_inputs["mm_token_type_ids"],
            image_grid_thw=grid_thw,
        )
    else:
        mrope_inputs = prepare_qwen35_mrope_inputs(
            processor_inputs["input_ids"],
            config,
            attention_mask=processor_inputs["attention_mask"],
            mm_token_type_ids=processor_inputs["mm_token_type_ids"],
            video_grid_thw=grid_thw,
        )

    hf_model = _load_hf_model(args.model_dir, hf_config)
    vision_engine_path = _resolve_vision_engine(
        args,
        config,
        hf_model,
        actual_patch_tokens,
    )
    trt_visual_features, vision_session = _run_vision_engine(
        vision_engine_path,
        pixel_values,
        grid_thw,
        config,
    )
    print(f"Merged vision tokens: {trt_visual_features.shape[0]}")

    hf_model = hf_model.to("cuda")
    hf_inputs = _hf_inputs_on_cuda(processor_inputs)
    with torch.inference_mode():
        hf_vision_outputs = hf_model.model.visual(
            hf_inputs[pixel_values_name].to(torch.bfloat16),
            grid_thw=hf_inputs[grid_name],
            return_dict=True,
        )
        hf_outputs = hf_model(
            **hf_inputs,
            use_cache=False,
            logits_to_keep=1,
            return_dict=True,
        )
    vision_comparison = _compare(trt_visual_features, hf_vision_outputs.pooler_output)
    _print_comparison("Vision pooled embedding", vision_comparison)
    hf_last_logits = hf_outputs.logits[0, -1, : config.vocab_size].float().cpu()

    del hf_outputs
    del hf_vision_outputs
    del hf_inputs
    del hf_model
    del vision_session
    gc.collect()
    torch.cuda.empty_cache()

    if args.image is not None:
        executor_inputs = prepare_qwen35_executor_prompt_inputs(
            processor_inputs["input_ids"],
            processor_inputs["attention_mask"],
            config,
            image_features=trt_visual_features,
        )
    else:
        executor_inputs = prepare_qwen35_executor_prompt_inputs(
            processor_inputs["input_ids"],
            processor_inputs["attention_mask"],
            config,
            video_features=trt_visual_features,
        )
    input_length = executor_inputs.batch_input_ids[0].numel()
    mrope_params = MropeParams(
        mrope_rotary_cos_sin=mrope_inputs.mrope_rotary_cos_sin,
        mrope_position_deltas=mrope_inputs.mrope_position_deltas,
    )

    runner = ModelRunnerCpp.from_dir(
        engine_dir=str(args.llm_engine_dir),
        max_batch_size=1,
        max_input_len=input_length,
        max_output_len=args.max_new_tokens,
        max_beam_width=1,
        kv_cache_enable_block_reuse=args.kv_cache_enable_block_reuse,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        enable_chunked_context=args.enable_chunked_context,
        use_runtime_defaults=False,
    )
    if not runner.gather_context_logits:
        raise RuntimeError(
            "The LLM engine was not built with --gather_context_logits. "
            "Rebuild it to enable the Hugging Face logits comparison."
        )
    required_prompt_rows = (
        executor_inputs.prompt_table.shape[0] * executor_inputs.prompt_table.shape[1]
    )
    if runner.max_prompt_embedding_table_size < required_prompt_rows:
        raise RuntimeError(
            "The LLM engine prompt table is too small: "
            f"capacity={runner.max_prompt_embedding_table_size}, required={required_prompt_rows}. "
            "Rebuild with a larger --max_prompt_embedding_table_size."
        )

    tokenizer = processor.tokenizer
    end_id = tokenizer.eos_token_id
    if end_id is None:
        raise ValueError("The tokenizer does not define eos_token_id")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id
    request_count = args.prefix_cache_requests if args.kv_cache_enable_block_reuse else 1
    outputs = None
    cache_observations = []
    with torch.inference_mode():
        for request_index in range(request_count):
            request_outputs = runner.generate(
                executor_inputs.batch_input_ids,
                prompt_table=executor_inputs.prompt_table,
                prompt_tasks=executor_inputs.prompt_tasks,
                multimodal_inputs=(
                    [multimodal_cache_input] if multimodal_cache_input is not None else None
                ),
                mrope_params=mrope_params,
                max_new_tokens=args.max_new_tokens,
                end_id=end_id,
                pad_id=pad_id,
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_beams=1,
                return_dict=True,
                output_context_logits=request_index == 0,
                output_sequence_lengths=True,
            )
            if outputs is None:
                outputs = request_outputs
            if args.kv_cache_enable_block_reuse:
                observation = _collect_cache_observation(runner)
                cache_observations.append(observation)
                print(
                    f"Prefix-cache request {request_index + 1}/{request_count}: "
                    f"reused_blocks={observation.request_reused_blocks}, "
                    f"request_hit_rate={observation.request_hit_rate:.2%}"
                )

    if outputs is None:
        raise RuntimeError("The TensorRT runner returned no outputs")
    if args.kv_cache_enable_block_reuse:
        first_observation = cache_observations[0]
        last_observation = cache_observations[-1]
        cumulative_reuse_delta = max(
            0,
            last_observation.cumulative_reused_blocks - first_observation.cumulative_reused_blocks,
        )
        observed_reused_blocks = max(
            last_observation.request_reused_blocks,
            cumulative_reuse_delta,
        )
        print(
            "Multimodal prefix-cache summary: "
            f"observed_reused_blocks={observed_reused_blocks}, "
            f"cumulative_hit_rate={last_observation.cumulative_hit_rate:.2%}, "
            f"tokens_per_block={last_observation.tokens_per_block}"
        )
        if observed_reused_blocks == 0:
            print(
                "WARNING: no reused cache blocks were observed. Ensure the "
                "prompt crosses a cacheable block or recurrent-state snapshot boundary."
            )
        else:
            print("Qwen3.5 multimodal prefix-cache validation: PASS")

    trt_last_logits = _select_last_context_logits(
        outputs["context_logits"],
        input_length,
        config.vocab_size,
    )
    logits_comparison = _compare(trt_last_logits, hf_last_logits)
    _print_comparison("Last context-token logits", logits_comparison)
    trt_top1, hf_top1, _ = _print_top_logits(
        tokenizer,
        trt_last_logits.float().cpu(),
        hf_last_logits,
        args.top_k,
    )

    sequence_length = int(outputs["sequence_lengths"][0, 0].item())
    output_ids = outputs["output_ids"][0, 0, input_length:sequence_length].cpu().tolist()
    generated_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"TensorRT generated text:\n{generated_text}")

    if args.check:
        failures = []
        if vision_comparison.cosine_similarity < args.vision_cosine_threshold:
            failures.append(
                "vision cosine "
                f"{vision_comparison.cosine_similarity:.8f} < {args.vision_cosine_threshold:.8f}"
            )
        if logits_comparison.cosine_similarity < args.logits_cosine_threshold:
            failures.append(
                "logits cosine "
                f"{logits_comparison.cosine_similarity:.8f} < {args.logits_cosine_threshold:.8f}"
            )
        if trt_top1 != hf_top1:
            failures.append(f"top-1 token mismatch: TensorRT={trt_top1}, Hugging Face={hf_top1}")
        if failures:
            raise RuntimeError("Qwen3.5 consistency check failed: " + "; ".join(failures))
        print("Qwen3.5 TensorRT/Hugging Face consistency check: PASS")

    if os.environ.get("QWEN35_EXIT_WITHOUT_RUNTIME_CLEANUP") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
