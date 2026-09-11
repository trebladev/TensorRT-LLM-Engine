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

import itertools
import math
from dataclasses import dataclass
from typing import Literal

import torch

from ..._utils import str_dtype_to_torch
from ...inputs.multimodal import MultimodalInput
from .config import Qwen35Config


@dataclass(frozen=True)
class Qwen35VisionPositionInputs:
    """Host-precomputed position inputs consumed by the Qwen3.5 vision graph."""

    position_ids: torch.Tensor
    position_weights: torch.Tensor
    rotary_cos: torch.Tensor
    rotary_sin: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class Qwen35PromptTuningInputs:
    """Engine inputs that inject Qwen3.5 image and video embeddings."""

    input_ids: torch.Tensor
    prompt_embedding_table: torch.Tensor
    prompt_tasks: torch.Tensor
    prompt_vocab_size: torch.Tensor


@dataclass(frozen=True)
class Qwen35ExecutorPromptInputs:
    """Batched prompt-tuning inputs consumed by ``ModelRunnerCpp.generate``."""

    batch_input_ids: list[torch.Tensor]
    prompt_table: torch.Tensor
    prompt_tasks: str


@dataclass(frozen=True)
class Qwen35MropeInputs:
    """Host-precomputed Qwen3.5 MRoPE inputs consumed by the LLM engine."""

    position_ids: torch.Tensor
    mrope_rotary_cos_sin: torch.Tensor
    mrope_position_deltas: torch.Tensor


def _grid_thw_list(grid_thw: torch.Tensor) -> list[tuple[int, int, int]]:
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {tuple(grid_thw.shape)}")
    grid = [tuple(int(value) for value in row) for row in grid_thw.cpu().tolist()]
    if not grid:
        raise ValueError("grid_thw must contain at least one image or video")
    if any(t <= 0 or h <= 0 or w <= 0 for t, h, w in grid):
        raise ValueError(f"grid_thw entries must be positive, got {grid}")
    return grid


def _reorder_spatial_tensor(
    tensor: torch.Tensor, height: int, width: int, merge: int
) -> torch.Tensor:
    return (
        tensor.view(height // merge, merge, width // merge, merge).permute(0, 2, 1, 3).reshape(-1)
    )


def _validate_visual_features(
    name: str,
    features: torch.Tensor | None,
    token_count: int,
    hidden_size: int,
) -> torch.Tensor | None:
    if features is None:
        if token_count:
            raise ValueError(
                f"Qwen3.5 input contains {token_count} {name} tokens but no {name} features"
            )
        return None
    if features.ndim != 2:
        raise ValueError(
            f"{name}_features must have shape [num_tokens, hidden_size], "
            f"got {tuple(features.shape)}"
        )
    if features.shape[0] != token_count:
        raise ValueError(
            f"Qwen3.5 {name} features and tokens do not match, "
            f"tokens: {token_count}, features: {features.shape[0]}"
        )
    if features.shape[1] != hidden_size:
        raise ValueError(
            f"Qwen3.5 {name} feature hidden size must be {hidden_size}, got {features.shape[1]}"
        )
    return features


def _llm_grid_thw_list(
    name: str,
    grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    *,
    split_temporal: bool,
) -> list[tuple[int, int, int]]:
    if grid_thw is None:
        return []
    grids = _grid_thw_list(grid_thw)
    invalid_grids = [
        grid for grid in grids if grid[1] % spatial_merge_size or grid[2] % spatial_merge_size
    ]
    if invalid_grids:
        raise ValueError(
            f"{name}_grid_thw height and width must be divisible by "
            f"spatial_merge_size={spatial_merge_size}, got {invalid_grids}"
        )
    if split_temporal:
        return [(1, height, width) for time, height, width in grids for _ in range(time)]
    return grids


def _vision_position_ids(
    start_position: int,
    grid_thw: tuple[int, int, int],
    spatial_merge_size: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    time, height, width = grid_thw
    llm_time = time
    llm_height = height // spatial_merge_size
    llm_width = width // spatial_merge_size
    sequence_length = llm_time * llm_height * llm_width
    position_width = torch.arange(
        start_position,
        start_position + llm_width,
        dtype=dtype,
        device=device,
    ).repeat(llm_height * llm_time)
    position_height = torch.arange(
        start_position,
        start_position + llm_height,
        dtype=dtype,
        device=device,
    ).repeat_interleave(llm_width * llm_time)
    position_temporal = torch.full(
        (sequence_length,),
        start_position,
        dtype=dtype,
        device=device,
    )
    return torch.stack((position_temporal, position_height, position_width))


def prepare_qwen35_mrope_inputs(
    input_ids: torch.Tensor,
    config: Qwen35Config,
    *,
    attention_mask: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
) -> Qwen35MropeInputs:
    """Prepare Qwen3.5 interleaved MRoPE inputs before fake-ID replacement.

    Args:
        input_ids: Original token IDs with shape [batch_size, sequence_length].
        config: Qwen3.5 model configuration.
        attention_mask: Optional padding mask with the same shape as input IDs.
            Nonzero entries are retained.
        mm_token_type_ids: Modality IDs with the same shape as input IDs:
            text is 0, image is 1, and video is 2. This is required when image
            or video grids are provided.
        image_grid_thw: Image patch grids with shape [num_images, 3].
        video_grid_thw: Video patch grids with shape [num_videos, 3].

    Returns:
        Full padded 3D position IDs, packed per-request rotary cos/sin tables,
        and generation position deltas. The rotary table has shape
        [batch_size, max_position_embeddings * rotary_embedding_dim].
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"input_ids must use int32 or int64, got {input_ids.dtype}")
    if input_ids.shape[0] == 0:
        raise ValueError("input_ids must contain at least one request")

    if attention_mask is None:
        active_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if attention_mask.shape != input_ids.shape:
            raise ValueError(
                "attention_mask must have the same shape as input_ids, "
                f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        active_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    empty_requests = [
        index for index in range(input_ids.shape[0]) if not torch.any(active_mask[index]).item()
    ]
    if empty_requests:
        raise ValueError(f"attention_mask removes every token from requests {empty_requests}")

    has_multimodal = image_grid_thw is not None or video_grid_thw is not None
    if has_multimodal and mm_token_type_ids is None:
        raise ValueError(
            "mm_token_type_ids is required when image_grid_thw or video_grid_thw is provided"
        )
    if mm_token_type_ids is None:
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    else:
        if mm_token_type_ids.shape != input_ids.shape:
            raise ValueError(
                "mm_token_type_ids must have the same shape as input_ids, "
                f"got {tuple(mm_token_type_ids.shape)} and {tuple(input_ids.shape)}"
            )
        token_type_ids = mm_token_type_ids.to(device=input_ids.device, dtype=torch.int32)
    active_token_types = token_type_ids[active_mask]
    invalid_token_types = sorted(set(active_token_types.cpu().tolist()) - {0, 1, 2})
    if invalid_token_types:
        raise ValueError(f"mm_token_type_ids entries must be 0, 1, or 2, got {invalid_token_types}")

    if has_multimodal and not config.has_vision:
        raise ValueError("Qwen3.5 multimodal MRoPE requires a vision config")
    spatial_merge_size = config.vision_spatial_merge_size if config.has_vision else 1
    image_grids = _llm_grid_thw_list(
        "image",
        image_grid_thw,
        spatial_merge_size,
        split_temporal=False,
    )
    video_grids = _llm_grid_thw_list(
        "video",
        video_grid_thw,
        spatial_merge_size,
        split_temporal=True,
    )
    grids = {1: image_grids, 2: video_grids}
    grid_indices = {1: 0, 2: 0}
    modality_names = {1: "image", 2: "video"}

    position_ids = torch.zeros(
        (3, input_ids.shape[0], input_ids.shape[1]),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    position_deltas = []
    packed_position_ids = torch.zeros(
        (3, input_ids.shape[0], config.max_position_embeddings),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    for batch_index in range(input_ids.shape[0]):
        request_mask = active_mask[batch_index]
        request_token_types = token_type_ids[batch_index][request_mask]
        request_position_groups = []
        current_position = 0
        grouped_token_types = itertools.groupby(
            enumerate(request_token_types.tolist()),
            key=lambda item: item[1],
        )
        for modality_type, group in grouped_token_types:
            group = list(group)
            group_length = len(group)
            if modality_type == 0:
                text_positions = torch.arange(
                    current_position,
                    current_position + group_length,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                request_position_groups.append(text_positions.view(1, -1).expand(3, -1))
                current_position += group_length
                continue

            grid_index = grid_indices[modality_type]
            modality_grids = grids[modality_type]
            if grid_index >= len(modality_grids):
                raise ValueError(
                    f"Request {batch_index} contains a {modality_names[modality_type]} "
                    "token group without a matching grid"
                )
            grid = modality_grids[grid_index]
            grid_indices[modality_type] += 1
            time, height, width = grid
            expected_group_length = (
                time * (height // spatial_merge_size) * (width // spatial_merge_size)
            )
            if group_length != expected_group_length:
                raise ValueError(
                    f"Request {batch_index} {modality_names[modality_type]} token group "
                    f"has length {group_length}, expected {expected_group_length} for grid {grid}"
                )
            request_position_groups.append(
                _vision_position_ids(
                    current_position,
                    grid,
                    spatial_merge_size,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            )
            current_position += max(height, width) // spatial_merge_size

        request_positions = torch.cat(request_position_groups, dim=1)
        request_length = request_positions.shape[1]
        if request_length > config.max_position_embeddings:
            raise ValueError(
                f"Request {batch_index} has {request_length} tokens, exceeding "
                f"max_position_embeddings={config.max_position_embeddings}"
            )
        max_position = int(request_positions.max().item())
        if max_position >= config.max_position_embeddings:
            raise ValueError(
                f"Request {batch_index} MRoPE position {max_position} exceeds "
                f"max_position_embeddings={config.max_position_embeddings}"
            )
        position_ids[:, batch_index, request_mask] = request_positions
        packed_position_ids[:, batch_index, :request_length] = request_positions
        position_deltas.append(max_position + 1 - request_length)

    unused_grids = {
        modality_names[modality_type]: len(modality_grids) - grid_indices[modality_type]
        for modality_type, modality_grids in grids.items()
        if grid_indices[modality_type] != len(modality_grids)
    }
    if unused_grids:
        raise ValueError(f"Qwen3.5 MRoPE grids were not consumed: {unused_grids}")

    half_rotary_dim = config.rotary_embedding_dim // 2
    inv_freq = 1.0 / (
        config.rotary_base
        ** (
            torch.arange(
                0,
                config.rotary_embedding_dim,
                2,
                dtype=torch.float32,
                device=input_ids.device,
            )
            / config.rotary_embedding_dim
        )
    )
    frequencies = packed_position_ids.unsqueeze(-1).to(torch.float32) * inv_freq
    interleaved_frequencies = frequencies[0].clone()
    for dimension, offset in enumerate((1, 2), start=1):
        length = config.mrope_section[dimension] * 3
        interleaved_frequencies[..., offset:length:3] = frequencies[dimension, ..., offset:length:3]
    rotary_cos_sin = torch.stack(
        (interleaved_frequencies.cos(), interleaved_frequencies.sin()),
        dim=-1,
    ).reshape(input_ids.shape[0], -1)
    expected_rotary_size = config.max_position_embeddings * config.rotary_embedding_dim
    if rotary_cos_sin.shape[1] != expected_rotary_size:
        raise ValueError(
            f"Qwen3.5 MRoPE table has size {rotary_cos_sin.shape[1]}, "
            f"expected {expected_rotary_size}"
        )
    if interleaved_frequencies.shape[-1] != half_rotary_dim:
        raise ValueError(
            f"Qwen3.5 MRoPE frequency dimension is {interleaved_frequencies.shape[-1]}, "
            f"expected {half_rotary_dim}"
        )

    return Qwen35MropeInputs(
        position_ids=position_ids,
        mrope_rotary_cos_sin=rotary_cos_sin.contiguous(),
        mrope_position_deltas=torch.tensor(
            position_deltas,
            dtype=torch.int32,
            device=input_ids.device,
        ).unsqueeze(1),
    )


def prepare_qwen35_prompt_tuning_inputs(
    input_ids: torch.Tensor,
    config: Qwen35Config,
    *,
    image_features: torch.Tensor | None = None,
    video_features: torch.Tensor | None = None,
) -> Qwen35PromptTuningInputs:
    """Replace visual placeholders and construct single-task prompt inputs.

    Args:
        input_ids: Original token IDs with shape [num_tokens] or
            [batch_size, sequence_length]. Multimodal position IDs must be
            computed before calling this function.
        config: Qwen3.5 model configuration.
        image_features: Merged image embeddings with shape
            [num_image_tokens, hidden_size].
        video_features: Merged video embeddings with shape
            [num_video_tokens, hidden_size].

    Returns:
        Prompt tuning inputs. Image rows precede video rows in the table.
        prompt_tasks is all zeros because the returned table is one global
        task. If the caller packs input_ids, it must pack prompt_tasks with
        the same mask and order.
    """
    if input_ids.ndim not in (1, 2):
        raise ValueError(
            f"input_ids must have shape [num_tokens] or [batch, sequence], "
            f"got {tuple(input_ids.shape)}"
        )
    if config.image_token_id is None or config.video_token_id is None:
        raise ValueError("Qwen3.5 prompt tuning requires image_token_id and video_token_id")

    image_mask = input_ids == config.image_token_id
    video_mask = input_ids == config.video_token_id
    image_token_count = int(image_mask.sum().item())
    video_token_count = int(video_mask.sum().item())
    image_features = _validate_visual_features(
        "image", image_features, image_token_count, config.hidden_size
    )
    video_features = _validate_visual_features(
        "video", video_features, video_token_count, config.hidden_size
    )

    prompt_dtype = str_dtype_to_torch(config.dtype)
    feature_tensors = [
        features.to(dtype=prompt_dtype)
        for features in (image_features, video_features)
        if features is not None
    ]
    if feature_tensors:
        prompt_device = feature_tensors[0].device
        prompt_embedding_table = torch.cat(
            [features.to(prompt_device) for features in feature_tensors],
            dim=0,
        ).contiguous()
        prompt_vocab_size_value = prompt_embedding_table.shape[0]
    else:
        prompt_device = input_ids.device
        prompt_embedding_table = torch.zeros(
            (1, config.hidden_size),
            dtype=prompt_dtype,
            device=prompt_device,
        )
        prompt_vocab_size_value = 0

    fake_input_ids = input_ids.to(torch.int32).clone()
    if image_token_count:
        fake_input_ids[image_mask] = torch.arange(
            config.vocab_size,
            config.vocab_size + image_token_count,
            dtype=torch.int32,
            device=input_ids.device,
        )
    if video_token_count:
        video_prompt_start = config.vocab_size + image_token_count
        fake_input_ids[video_mask] = torch.arange(
            video_prompt_start,
            video_prompt_start + video_token_count,
            dtype=torch.int32,
            device=input_ids.device,
        )

    return Qwen35PromptTuningInputs(
        input_ids=fake_input_ids,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=torch.zeros_like(fake_input_ids, dtype=torch.int32),
        prompt_vocab_size=torch.tensor(
            [prompt_vocab_size_value],
            dtype=torch.int32,
            device=prompt_device,
        ),
    )


def prepare_qwen35_multimodal_cache_input(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: Qwen35Config,
    modality: Literal["image", "video"],
    content_hash: list[int] | None = None,
    *,
    content_hashes: list[list[int]] | None = None,
) -> MultimodalInput:
    """Build content-aware cache metadata for a batch-1 visual prompt.

    A single image or video hash is duplicated across all of its prompt spans.
    Callers that represent each span as an independent visual item can instead
    provide one hash per span through ``content_hashes``.
    """
    if input_ids.shape[0] != 1 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "Qwen3.5 multimodal prefix caching currently requires batch size 1 "
            "and an attention mask matching input_ids."
        )
    if (content_hash is None) == (content_hashes is None):
        raise ValueError("Provide exactly one of content_hash or content_hashes")
    if config.vision_start_token_id is None:
        raise ValueError("Qwen3.5 multimodal prefix caching requires vision_start_token_id")
    if modality == "image":
        multimodal_token_id = config.image_token_id
        other_multimodal_token_id = config.video_token_id
    elif modality == "video":
        multimodal_token_id = config.video_token_id
        other_multimodal_token_id = config.image_token_id
    else:
        raise ValueError(f"Unsupported Qwen3.5 multimodal cache modality: {modality}")
    if multimodal_token_id is None:
        raise ValueError(f"Qwen3.5 configuration does not define a {modality}_token_id")

    active_mask = attention_mask[0].to(dtype=torch.bool, device=input_ids.device)
    request_ids = input_ids[0][active_mask].cpu().tolist()
    if other_multimodal_token_id is not None and other_multimodal_token_id in request_ids:
        raise ValueError("Qwen3.5 multimodal cache metadata supports one modality per request.")

    positions = []
    lengths = []
    token_index = 0
    while token_index < len(request_ids):
        if request_ids[token_index] != config.vision_start_token_id:
            token_index += 1
            continue
        span_start = token_index
        token_index += 1
        span_end = token_index
        while span_end < len(request_ids) and request_ids[span_end] == multimodal_token_id:
            span_end += 1
        if span_end == token_index:
            continue
        positions.append(span_start)
        lengths.append(span_end - span_start)
        token_index = span_end

    expected_visual_positions = {
        index for index, token_id in enumerate(request_ids) if token_id == multimodal_token_id
    }
    covered_visual_positions = {
        index
        for position, length in zip(positions, lengths)
        for index in range(position + 1, position + length)
    }
    if not positions or covered_visual_positions != expected_visual_positions:
        raise ValueError(
            f"Could not map every Qwen3.5 visual placeholder token to a {modality} cache span."
        )

    if content_hashes is None:
        if content_hash is None:
            raise ValueError("content_hash must be provided when content_hashes is None")
        span_hashes = [content_hash.copy() for _ in positions]
    else:
        if len(content_hashes) != len(positions):
            raise ValueError(
                "content_hashes must contain one hash per Qwen3.5 visual prompt span, "
                f"got {len(content_hashes)} hashes and {len(positions)} spans"
            )
        span_hashes = [item_hash.copy() for item_hash in content_hashes]
    for item_hash in span_hashes:
        if (
            len(item_hash) != 8
            or not all(isinstance(value, int) for value in item_hash)
            or any(value < -(2**31) or value >= 2**31 for value in item_hash)
        ):
            raise ValueError("Each content hash must contain exactly eight signed int32 values")

    return MultimodalInput.from_components(
        mm_hashes=span_hashes,
        mm_positions=positions,
        mm_lengths=lengths,
    )


def prepare_qwen35_executor_prompt_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    config: Qwen35Config,
    *,
    image_features: torch.Tensor | None = None,
    video_features: torch.Tensor | None = None,
) -> Qwen35ExecutorPromptInputs:
    """Construct request-local prompt tables for ``ModelRunnerCpp.generate``.

    Args:
        input_ids: Padded original token IDs with shape
            [batch_size, sequence_length]. Multimodal position IDs must be
            computed before calling this function.
        attention_mask: Padding mask with the same shape as ``input_ids``.
            Nonzero entries are retained in each request.
        config: Qwen3.5 model configuration.
        image_features: Merged image embeddings for the full batch in request
            order, with shape [num_image_tokens, hidden_size].
        video_features: Merged video embeddings for the full batch in request
            order, with shape [num_video_tokens, hidden_size].

    Returns:
        Unpadded request token IDs with request-local fake IDs, a prompt table
        with shape [batch_size, max_visual_tokens, hidden_size], and a
        comma-separated task selector. Each request owns one task; shorter
        tables are padded with zeros.
    """
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must have the same shape as input_ids, "
            f"got {tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
        )
    if input_ids.shape[0] == 0:
        raise ValueError("input_ids must contain at least one request")
    if config.image_token_id is None or config.video_token_id is None:
        raise ValueError("Qwen3.5 prompt tuning requires image_token_id and video_token_id")

    active_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
    request_input_ids = [
        input_ids[index][active_mask[index]] for index in range(input_ids.shape[0])
    ]
    empty_requests = [
        index for index, request_ids in enumerate(request_input_ids) if request_ids.numel() == 0
    ]
    if empty_requests:
        raise ValueError(f"attention_mask removes every token from requests {empty_requests}")

    image_token_count = sum(
        int((request_ids == config.image_token_id).sum().item())
        for request_ids in request_input_ids
    )
    video_token_count = sum(
        int((request_ids == config.video_token_id).sum().item())
        for request_ids in request_input_ids
    )
    image_features = _validate_visual_features(
        "image", image_features, image_token_count, config.hidden_size
    )
    video_features = _validate_visual_features(
        "video", video_features, video_token_count, config.hidden_size
    )

    prompt_device = input_ids.device
    if image_features is not None:
        prompt_device = image_features.device
    elif video_features is not None:
        prompt_device = video_features.device

    batch_input_ids = []
    request_prompt_tables = []
    image_offset = 0
    video_offset = 0
    for request_ids in request_input_ids:
        request_image_count = int((request_ids == config.image_token_id).sum().item())
        request_video_count = int((request_ids == config.video_token_id).sum().item())
        request_image_features = (
            image_features[image_offset : image_offset + request_image_count]
            if request_image_count
            else None
        )
        request_video_features = (
            video_features[video_offset : video_offset + request_video_count]
            if request_video_count
            else None
        )
        prompt_inputs = prepare_qwen35_prompt_tuning_inputs(
            request_ids,
            config,
            image_features=request_image_features,
            video_features=request_video_features,
        )
        batch_input_ids.append(prompt_inputs.input_ids.contiguous())
        request_prompt_tables.append(prompt_inputs.prompt_embedding_table.to(prompt_device))
        image_offset += request_image_count
        video_offset += request_video_count

    max_prompt_tokens = max(table.shape[0] for table in request_prompt_tables)
    prompt_table = torch.zeros(
        (input_ids.shape[0], max_prompt_tokens, config.hidden_size),
        dtype=str_dtype_to_torch(config.dtype),
        device=prompt_device,
    )
    for task_index, table in enumerate(request_prompt_tables):
        prompt_table[task_index, : table.shape[0]] = table

    return Qwen35ExecutorPromptInputs(
        batch_input_ids=batch_input_ids,
        prompt_table=prompt_table.contiguous(),
        prompt_tasks=",".join(str(index) for index in range(input_ids.shape[0])),
    )


def prepare_qwen35_vision_position_inputs(
    grid_thw: torch.Tensor,
    config: Qwen35Config,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str | None = None,
) -> Qwen35VisionPositionInputs:
    """Prepare learned-position interpolation, RoPE, and packed attention metadata."""
    if not config.has_vision:
        raise ValueError("Qwen3.5 vision position inputs require a vision config")
    if not dtype.is_floating_point:
        raise ValueError(f"Vision position inputs require a floating dtype, got {dtype}")

    grid = _grid_thw_list(grid_thw)
    merge = config.vision_spatial_merge_size
    invalid_grids = [(t, h, w) for t, h, w in grid if h % merge != 0 or w % merge != 0]
    if invalid_grids:
        raise ValueError(
            f"Vision grid height and width must be divisible by spatial_merge_size={merge}, "
            f"got {invalid_grids}"
        )

    grid_side = math.isqrt(config.vision_num_position_embeddings)
    if grid_side * grid_side != config.vision_num_position_embeddings:
        raise ValueError(
            "vision_num_position_embeddings must be a perfect square, "
            f"got {config.vision_num_position_embeddings}"
        )

    corner_ids: list[list[torch.Tensor]] = [[] for _ in range(4)]
    corner_weights: list[list[torch.Tensor]] = [[] for _ in range(4)]
    rotary_position_ids = []
    sequence_lengths = []

    for temporal, height, width in grid:
        h_positions = torch.linspace(0, grid_side - 1, height, dtype=torch.float32)
        w_positions = torch.linspace(0, grid_side - 1, width, dtype=torch.float32)
        h_floor = h_positions.to(torch.int64)
        w_floor = w_positions.to(torch.int64)
        h_ceil = (h_floor + 1).clamp(max=grid_side - 1)
        w_ceil = (w_floor + 1).clamp(max=grid_side - 1)
        h_delta = h_positions - h_floor
        w_delta = w_positions - w_floor
        base_h = h_floor * grid_side
        base_h_ceil = h_ceil * grid_side

        indices = (
            (base_h[:, None] + w_floor[None, :]).reshape(-1),
            (base_h[:, None] + w_ceil[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_floor[None, :]).reshape(-1),
            (base_h_ceil[:, None] + w_ceil[None, :]).reshape(-1),
        )
        weights = (
            ((1 - h_delta)[:, None] * (1 - w_delta)[None, :]).reshape(-1),
            ((1 - h_delta)[:, None] * w_delta[None, :]).reshape(-1),
            (h_delta[:, None] * (1 - w_delta)[None, :]).reshape(-1),
            (h_delta[:, None] * w_delta[None, :]).reshape(-1),
        )
        for corner in range(4):
            reordered_ids = _reorder_spatial_tensor(indices[corner], height, width, merge)
            reordered_weights = _reorder_spatial_tensor(weights[corner], height, width, merge)
            corner_ids[corner].append(reordered_ids.repeat(temporal))
            corner_weights[corner].append(reordered_weights.repeat(temporal))

        row_ids = torch.arange(height).unsqueeze(1).expand(-1, width)
        column_ids = torch.arange(width).unsqueeze(0).expand(height, -1)
        row_ids = _reorder_spatial_tensor(row_ids, height, width, merge)
        column_ids = _reorder_spatial_tensor(column_ids, height, width, merge)
        frame_position_ids = torch.stack((row_ids, column_ids), dim=-1)
        rotary_position_ids.append(frame_position_ids.repeat(temporal, 1))
        sequence_lengths.extend([height * width] * temporal)

    position_ids = torch.stack([torch.cat(values).to(torch.int32) for values in corner_ids])
    position_weights = torch.stack([torch.cat(values).to(dtype) for values in corner_weights])

    vision_head_dim = config.vision_hidden_size // config.vision_num_heads
    rotary_dim = vision_head_dim // 2
    inv_freq = 1.0 / (
        10_000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    max_grid_size = max(max(height, width) for _, height, width in grid)
    frequency_table = torch.outer(torch.arange(max_grid_size, dtype=torch.float32), inv_freq)
    frequencies = frequency_table[torch.cat(rotary_position_ids)].flatten(1)
    rotary_embedding = torch.cat((frequencies, frequencies), dim=-1)
    rotary_cos = rotary_embedding.cos().to(dtype)
    rotary_sin = rotary_embedding.sin().to(dtype)

    total_tokens = sum(sequence_lengths)
    attention_mask = torch.full(
        (1, 1, total_tokens, total_tokens),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    offset = 0
    for sequence_length in sequence_lengths:
        attention_mask[
            ..., offset : offset + sequence_length, offset : offset + sequence_length
        ] = 0
        offset += sequence_length

    target_device = grid_thw.device if device is None else torch.device(device)
    return Qwen35VisionPositionInputs(
        position_ids=position_ids.to(target_device),
        position_weights=position_weights.to(target_device),
        rotary_cos=rotary_cos.to(target_device),
        rotary_sin=rotary_sin.to(target_device),
        attention_mask=attention_mask.to(target_device),
    )
