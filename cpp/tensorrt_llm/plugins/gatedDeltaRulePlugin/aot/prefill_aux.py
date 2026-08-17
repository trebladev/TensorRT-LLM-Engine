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

import triton
import triton.language as tl


@triton.jit
def init_chunk_indices_kernel(
    chunk_indices,
    max_chunks,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < max_chunks
    tl.store(chunk_indices + offsets * 2, 0, mask=mask)
    tl.store(chunk_indices + offsets * 2 + 1, max_chunks, mask=mask)


@triton.jit
def prepare_chunk_metadata_kernel(
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    chunk_counter,
    num_requests,
    CHUNK_SIZE: tl.constexpr,
):
    request_idx = tl.program_id(0)
    if request_idx >= num_requests:
        return

    begin = tl.load(cu_seqlens + request_idx).to(tl.int32)
    end = tl.load(cu_seqlens + request_idx + 1).to(tl.int32)
    num_chunks = tl.cdiv(end - begin, CHUNK_SIZE)
    chunk_offset = tl.atomic_add(chunk_counter, num_chunks)
    tl.store(chunk_offsets + request_idx, chunk_offset)

    for local_chunk_idx in range(num_chunks):
        global_chunk_idx = chunk_offset + local_chunk_idx
        tl.store(chunk_indices + global_chunk_idx * 2, request_idx)
        tl.store(chunk_indices + global_chunk_idx * 2 + 1, local_chunk_idx)


@triton.jit
def zero_missing_states_kernel(
    state,
    state_slot_mapping,
    has_initial_state,
    num_requests,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    value_block_idx = tl.program_id(0)
    request_head_idx = tl.program_id(1)
    request_idx = request_head_idx // H
    head_idx = request_head_idx % H
    if request_idx >= num_requests or tl.load(has_initial_state + request_idx) != 0:
        return

    state_slot = tl.load(state_slot_mapping + request_idx).to(tl.int64)
    state += (state_slot * H + head_idx) * V * K
    state_block = tl.make_block_ptr(
        state,
        (V, K),
        (K, 1),
        (value_block_idx * BV, 0),
        (BV, K),
        (1, 0),
    )
    zeros = tl.zeros((BV, K), dtype=tl.float32)
    tl.store(state_block, zeros, boundary_check=(0, 1))


@triton.jit
def gather_states_kernel(
    state,
    final_state,
    state_slot_mapping,
    num_requests,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    value_block_idx = tl.program_id(0)
    request_head_idx = tl.program_id(1)
    request_idx = request_head_idx // H
    head_idx = request_head_idx % H
    if request_idx >= num_requests:
        return

    state_slot = tl.load(state_slot_mapping + request_idx).to(tl.int64)
    state += (state_slot * H + head_idx) * V * K
    final_state += (request_idx * H + head_idx) * V * K
    state_block = tl.make_block_ptr(
        state,
        (V, K),
        (K, 1),
        (value_block_idx * BV, 0),
        (BV, K),
        (1, 0),
    )
    final_state_block = tl.make_block_ptr(
        final_state,
        (V, K),
        (K, 1),
        (value_block_idx * BV, 0),
        (BV, K),
        (1, 0),
    )
    values = tl.load(state_block, boundary_check=(0, 1))
    tl.store(final_state_block, values, boundary_check=(0, 1))
