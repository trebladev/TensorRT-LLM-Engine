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

import json

import torch
from ablation import ROOT, load, stats


def main() -> None:
    torch.set_num_threads(1)
    n, t, h, hv, k, v, slots, stride = map(
        int, (ROOT / "slow_inputs/0_meta.txt").read_text().split()
    )
    starts = load("cu_seqlens", torch.int32).cpu().long()[:-1]
    source = load("source", torch.int32).cpu().long()
    snapshots = load("snapshots", torch.int32).cpu().long()[starts]

    def cpu(name: str, dtype: torch.dtype) -> torch.Tensor:
        return load(name, dtype).cpu().double()

    def bf(x: torch.Tensor) -> torch.Tensor:
        return x.bfloat16().double()

    q, key, value = [
        cpu(name, torch.bfloat16).view(t, h, k)[starts] for name in ("query", "key", "value")
    ]
    q = bf(q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt())
    key = bf(key / (key.square().sum(-1, keepdim=True) + 1e-6).sqrt())
    decay = cpu("log_decay", torch.float32).view(t, hv)[starts].exp().unsqueeze(-1)
    beta = cpu("beta", torch.float32).view(t, hv)[starts].unsqueeze(-1)
    state = (
        cpu("state_before", torch.float32)
        .view(slots, stride)[:, : hv * v * k]
        .reshape(slots, hv, v, k)[source]
    )
    expected = cpu("output", torch.bfloat16).view(t, hv, v)[starts]
    expected_state = (
        cpu("state_after", torch.float32)
        .view(slots, stride)[:, : hv * v * k]
        .reshape(slots, hv, v, k)[snapshots]
    )
    results = []
    for omit in ("none", "state", "wu", "residual", "qk", "all"):

        def rounded(x: torch.Tensor, stage: str) -> torch.Tensor:
            return x if omit in (stage, "all") else bf(x)

        w = rounded(key * beta * decay, "wu")
        u = rounded(value * beta, "wu")
        initial = rounded(state, "state")
        residual = u - (initial * w.unsqueeze(-2)).sum(-1)
        qh = (initial * q.unsqueeze(-2)).sum(-1)
        qk = rounded((q * key).sum(-1, keepdim=True), "qk")
        out = bf((qh * decay + qk * rounded(residual, "residual")) * k**-0.5)
        new_state = state * decay.unsqueeze(-1) + rounded(residual, "residual").unsqueeze(
            -1
        ) * key.unsqueeze(-2)
        results.append(
            dict(
                omitted_rounding=omit,
                output=stats(out, expected),
                snapshot=stats(new_state, expected_state),
            )
        )
    (ROOT / "rounding_oracle.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
