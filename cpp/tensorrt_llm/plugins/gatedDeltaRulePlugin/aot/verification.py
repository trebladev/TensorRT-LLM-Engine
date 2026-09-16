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

"""Packed recurrent verification with per-token state snapshots.

Match the single-token decode update, keeping the FP32 state tile live across
all tokens of a request. The caller commits only the accepted snapshot.
"""

import triton
import triton.language as tl


@triton.jit
def gated_delta_rule_verification_kernel(
    query,
    key,
    value,
    log_decay,
    beta,
    output,
    state,
    final_state,
    source_slots,
    target_slots,
    snapshot_slots,
    cu_seqlens,
    state_stride,
    use_snapshots,
    scale,
    H: tl.constexpr,
    HG: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    value_block = tl.program_id(2)
    begin = tl.load(cu_seqlens + request).to(tl.int64)
    end = tl.load(cu_seqlens + request + 1).to(tl.int64)
    k = tl.arange(0, K)
    v = value_block * BV + tl.arange(0, BV)
    state_offsets = head * V * K + k[:, None] + v[None, :] * K
    source = tl.load(source_slots + request).to(tl.int64)
    target = tl.load(target_slots + request).to(tl.int64)
    hidden = tl.load(state + source * state_stride + state_offsets, v[None, :] < V, 0)

    for token in range(begin, end):
        qk_offsets = (token * HG + head // (H // HG)) * K + k
        q = tl.load(query + qk_offsets).to(tl.float32)
        key_values = tl.load(key + qk_offsets).to(tl.float32)
        values = tl.load(value + (token * H + head) * V + v, v < V, 0).to(tl.float32)
        decay = tl.load(log_decay + token * H + head)
        beta_value = tl.load(beta + token * H + head)
        # Match the existing decode kernel, including epsilon placement.
        q = q / (tl.sqrt(tl.sum(q * q)) + 1e-6)
        key_values = key_values / (tl.sqrt(tl.sum(key_values * key_values)) + 1e-6)
        q *= scale
        hidden *= tl.exp(decay)
        residual = (values - tl.sum(hidden * key_values[:, None], 0)) * beta_value
        hidden += key_values[:, None] * residual[None, :]
        result = tl.sum(hidden * q[:, None], 0)
        tl.store(output + (token * H + head) * V + v, result.to(tl.bfloat16), v < V)
        if use_snapshots:
            snapshot = tl.load(snapshot_slots + token).to(tl.int64)
            if snapshot >= 0:
                tl.store(state + snapshot * state_stride + state_offsets, hidden, v[None, :] < V)

    tl.store(state + target * state_stride + state_offsets, hidden, v[None, :] < V)
    tl.store(final_state + request * H * V * K + state_offsets, hidden, v[None, :] < V)
