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
import os
from pathlib import Path

import torch
from ablation_kernel import gated_delta_rule_verification

ROOT = Path(os.environ.get("QWEN35_GDN_DEBUG_DIR", "/tmp/qwen35_gdn_debug"))


def load(name: str, dtype: torch.dtype, directory: str = "slow_inputs") -> torch.Tensor:
    return (
        torch.frombuffer(bytearray((ROOT / directory / f"0_{name}.bin").read_bytes()), dtype=dtype)
        .clone()
        .cuda()
    )


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float | int]:
    a, b = a.float(), b.float()
    d = a - b
    return dict(
        max_abs=d.abs().max().item(),
        relative_l2=(d.norm() / b.norm()).item(),
        unequal=int((a != b).sum()),
        count=a.numel(),
    )


def main() -> None:
    torch.set_num_threads(1)
    n, t, h, hv, k, v, slots, stride = map(
        int, (ROOT / "slow_inputs/0_meta.txt").read_text().split()
    )
    q, key, value = [load(name, torch.bfloat16) for name in ("query", "key", "value")]
    g, beta = [load(name, torch.float32) for name in ("log_decay", "beta")]
    source, target, cu, snapshots = [
        load(name, torch.int32) for name in ("source", "target", "cu_seqlens", "snapshots")
    ]

    def states(raw: torch.Tensor) -> torch.Tensor:
        return raw.view(slots, stride)[:, : hv * v * k].reshape(slots, hv, v, k)[
            snapshots[snapshots >= 0].long()
        ]

    expected_o = load("output", torch.bfloat16)
    expected_s = states(load("state_after", torch.float32))
    results = []
    for mode in range(3):
        state = load("state_before", torch.float32)
        out = torch.empty_like(value)
        gated_delta_rule_verification[(1, 16, n * hv)](
            q,
            key,
            value,
            g,
            beta,
            out,
            state,
            source,
            target,
            stride,
            cu,
            k**-0.5,
            snapshots,
            2,
            t,
            H=h,
            HV=hv,
            K=k,
            V=v,
            BK=128,
            BV=8,
            NORM_MODE=mode,
            num_warps=1,
            num_stages=3,
        )
        torch.cuda.synchronize()
        row = dict(
            mode=mode,
            output_vs_chunk=stats(out, expected_o),
            snapshots_vs_chunk=stats(states(state), expected_s),
        )
        if mode == 0:
            row["output_vs_captured_fast"] = stats(
                out, load("output", torch.bfloat16, "fast_inputs")
            )
            row["snapshots_vs_captured_fast"] = stats(
                states(state), states(load("state_after", torch.float32, "fast_inputs"))
            )
        results.append(row)
    norms = {}
    for name, x in (("q", q), ("k", key)):
        x = x.float().view(-1, k)
        norm = x.square().sum(-1).sqrt()
        old = x / (norm[:, None] + 1e-6)
        new = x / (norm.square()[:, None] + 1e-6).sqrt()
        norms[name] = dict(
            min_norm=norm.min().item(), max_norm=norm.max().item(), epsilon_change=stats(new, old)
        )
    report = dict(ablations=results, norms=norms)
    (ROOT / "ablation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
