# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
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
from transformers import AutoConfig, AutoProcessor
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
            "Run an image or video through the Qwen3.5 TensorRT vision and LLM engines."
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
    if not 0 < args.kv_cache_free_gpu_memory_fraction <= 1:
        raise ValueError("kv_cache_free_gpu_memory_fraction must be in (0, 1]")
    if args.prefix_cache_requests <= 0:
        raise ValueError("prefix_cache_requests must be positive")
    if args.kv_cache_enable_block_reuse and args.prefix_cache_requests < 2:
        raise ValueError("prefix_cache_requests must be at least 2 when block reuse is enabled")


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
    actual_tokens: int,
) -> Path:
    engine_path = args.vision_engine_dir / _VISION_ENGINE_NAME
    metadata_path = args.vision_engine_dir / _VISION_CONFIG_NAME
    if not engine_path.is_file():
        raise RuntimeError(
            "The video demo only runs TensorRT engines and cannot build a vision engine. "
            "Build it separately with examples/models/core/qwen3_5/build_vision_engine.sh."
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

    vision_engine_path = _resolve_vision_engine(
        args,
        config,
        actual_patch_tokens,
    )
    trt_visual_features, vision_session = _run_vision_engine(
        vision_engine_path,
        pixel_values,
        grid_thw,
        config,
    )
    print(f"Merged vision tokens: {trt_visual_features.shape[0]}")

    del vision_session

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

    sequence_length = int(outputs["sequence_lengths"][0, 0].item())
    output_ids = outputs["output_ids"][0, 0, input_length:sequence_length].cpu().tolist()
    generated_text = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"TensorRT generated text:\n{generated_text}")

    if os.environ.get("QWEN35_EXIT_WITHOUT_RUNTIME_CLEANUP") == "1":
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
