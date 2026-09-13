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

"""Short packed GDN recurrence with snapshots in the combined state pool.

The recurrence follows the decode kernel in _torch/modules/fla/fused_recurrent.py.
Each program keeps its state tile in registers across verification tokens.
"""

import triton
import triton.language as tl


@triton.jit
def gated_delta_rule_verification(
    q,
    k,
    v,
    g,
    beta,
    o,
    state,
    source_slots,
    target_slots,
    state_stride,
    cu_seqlens,
    scale,
    snapshot_slots,
    cache_steps,
    total_tokens,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NORM_MODE: tl.constexpr,
):
    value_block = tl.program_id(1)
    request_head = tl.program_id(2)
    request = request_head // HV
    head = request_head % HV
    q_head = head // (HV // H)
    begin = tl.load(cu_seqlens + request).to(tl.int64)
    end = tl.load(cu_seqlens + request + 1).to(tl.int64)
    keys = tl.arange(0, BK)
    values = value_block * BV + tl.arange(0, BV)
    mask = (keys[:, None] < K) & (values[None, :] < V)
    state_offsets = head * V * K + keys[:, None] + values[None, :] * K
    source = tl.load(source_slots + request).to(tl.int64)
    hidden = tl.load(state + source * state_stride + state_offsets, mask, other=0)
    for token in range(begin, end):
        query = tl.load(q + (token * H + q_head) * K + keys, keys < K, other=0).to(tl.float32)
        key = tl.load(k + (token * H + q_head) * K + keys, keys < K, other=0).to(tl.float32)
        value = tl.load(v + (token * HV + head) * V + values, values < V, other=0).to(tl.float32)
        if NORM_MODE == 0:
            query = query / (tl.sqrt(tl.sum(query * query)) + 1e-6)
            key = key / (tl.sqrt(tl.sum(key * key)) + 1e-6)
        else:
            query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
            key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        if NORM_MODE == 2:
            query = query.to(tl.bfloat16).to(tl.float32)
            key = key.to(tl.bfloat16).to(tl.float32)
        query *= scale
        hidden *= tl.exp(tl.load(g + token * HV + head))
        value -= tl.sum(hidden * key[:, None], 0)
        value *= tl.load(beta + token * HV + head)
        hidden += key[:, None] * value[None, :]
        output = tl.sum(hidden * query[:, None], 0)
        tl.store(o + (token * HV + head) * V + values, output, values < V)
        snapshot = tl.load(snapshot_slots + token).to(tl.int64)
        if snapshot >= 0:
            tl.store(state + snapshot * state_stride + state_offsets, hidden, mask)
    target = tl.load(target_slots + request).to(tl.int64)
    tl.store(state + target * state_stride + state_offsets, hidden, mask)
