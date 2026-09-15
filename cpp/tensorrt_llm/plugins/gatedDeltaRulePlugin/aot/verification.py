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

"""Fused one/two-token chunk algebra, including BF16 rounding and state snapshots."""

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
    begin = tl.load(cu_seqlens + request)
    end = tl.load(cu_seqlens + request + 1)
    length = end - begin
    # Generation inputs have one or two tokens per request. Padding to 16 only
    # supplies the tensor-core tile; it never advances the logical history.
    t = tl.arange(0, 16)
    k = tl.arange(0, K)
    v = value_block * BV + tl.arange(0, BV)
    qk_offsets = ((begin + t[:, None]) * HG + head // (H // HG)) * K + k[None, :]
    q = tl.load(query + qk_offsets, t[:, None] < length, 0).to(tl.float32)
    key_values = tl.load(key + qk_offsets, t[:, None] < length, 0).to(tl.float32)
    q = (q / tl.sqrt(tl.sum(q * q, 1) + 1e-6)[:, None]).to(tl.bfloat16)
    key_values = (key_values / tl.sqrt(tl.sum(key_values * key_values, 1) + 1e-6)[:, None]).to(
        tl.bfloat16
    )
    q0, q1 = tl.split(q.reshape(16, 2, 64).permute(0, 2, 1))
    k0, k1 = tl.split(key_values.reshape(16, 2, 64).permute(0, 2, 1))
    g = tl.load(log_decay + (begin + t) * H + head, t < length, 0)
    g = tl.cumsum(g)
    b = tl.load(beta + (begin + t) * H + head, t < length, 0)
    decay = tl.exp(g)
    relative = g[:, None] - g[None, :]
    relative_decay = tl.exp(tl.where(relative <= 0, relative, float("-inf")))

    # For at most two tokens, (I + strictly_lower(beta K K^T))^-1
    # has only one off-diagonal entry. Preserve the prefill BF16 transition.
    kk = tl.dot(k1, tl.trans(k1), tl.dot(k0, tl.trans(k0)))
    transition = tl.where(
        (t[:, None] > t[None, :]) & (t[:, None] < length), -(kk * relative_decay) * b[:, None], 0.0
    )
    transition = (transition + (t[:, None] == t[None, :]).to(tl.float32)).to(tl.bfloat16)
    val_offsets = ((begin + t[:, None]) * H + head) * V + v[None, :]
    values = tl.load(value + val_offsets, (t[:, None] < length) & (v[None, :] < V), 0)
    u = tl.dot(transition, (values.to(tl.float32) * b[:, None]).to(tl.bfloat16)).to(tl.bfloat16)
    w0 = tl.dot(transition, (k0.to(tl.float32) * b[:, None] * decay[:, None]).to(tl.bfloat16)).to(
        tl.bfloat16
    )
    w1 = tl.dot(transition, (k1.to(tl.float32) * b[:, None] * decay[:, None]).to(tl.bfloat16)).to(
        tl.bfloat16
    )

    source = tl.load(source_slots + request).to(tl.int64)
    target = tl.load(target_slots + request).to(tl.int64)
    half_k = tl.arange(0, 64)
    state_offsets = head * V * K + v[None, :] * K + half_k[:, None]
    h0 = tl.load(state + source * state_stride + state_offsets, v[None, :] < V, 0)
    h1 = tl.load(state + source * state_stride + state_offsets + 64, v[None, :] < V, 0)
    h0_bf16 = h0.to(tl.bfloat16)
    h1_bf16 = h1.to(tl.bfloat16)
    residual = u.to(tl.float32) - (tl.dot(w1, h1_bf16, tl.dot(w0, h0_bf16)))

    # Accumulate K tiles in the same order as the chunk kernels.
    qh = tl.dot(q1, h1_bf16, tl.dot(q0, h0_bf16))
    qk = tl.dot(q1, tl.trans(k1), tl.dot(q0, tl.trans(k0)))
    attention = tl.where(t[:, None] >= t[None, :], qk * relative_decay, 0.0).to(tl.bfloat16)
    # Match chunk_o: scale the history first, then fuse the residual term.
    # Reversing the FMA changes BF16 rounding near cancellation.
    result = tl.fma(
        tl.dot(attention, residual.to(tl.bfloat16)), scale, (qh * decay[:, None]) * scale
    )
    tl.store(output + val_offsets, result.to(tl.bfloat16), (t[:, None] < length) & (v[None, :] < V))

    # Both snapshots are computed from the same FP32 initial state, as in the
    # chunk reference. Sequential recurrent updates have different BF16 rounding.
    for step in range(length):
        step_g = tl.sum(tl.where(t == step, g, 0.0), 0)
        weights = tl.exp(tl.where(step_g - g <= 0, step_g - g, float("-inf")))
        step_values = tl.where((t <= step)[:, None], residual * weights[:, None], 0.0).to(
            tl.bfloat16
        )
        next_h0 = h0 * tl.exp(step_g) + tl.dot(tl.trans(k0), step_values)
        next_h1 = h1 * tl.exp(step_g) + tl.dot(tl.trans(k1), step_values)
        if use_snapshots:
            snapshot = tl.load(snapshot_slots + begin + step).to(tl.int64)
            if snapshot >= 0:
                tl.store(state + snapshot * state_stride + state_offsets, next_h0, v[None, :] < V)
                tl.store(
                    state + snapshot * state_stride + state_offsets + 64, next_h1, v[None, :] < V
                )
        if step == length - 1:
            tl.store(state + target * state_stride + state_offsets, next_h0, v[None, :] < V)
            tl.store(state + target * state_stride + state_offsets + 64, next_h1, v[None, :] < V)
            compact = request * H * V * K + state_offsets
            tl.store(final_state + compact, next_h0, v[None, :] < V)
            tl.store(final_state + compact + 64, next_h1, v[None, :] < V)
