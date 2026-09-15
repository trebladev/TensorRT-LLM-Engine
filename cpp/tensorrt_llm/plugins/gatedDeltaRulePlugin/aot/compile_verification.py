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

"""AOT compile SM89 short-verification cubins; no runtime Triton dependency."""

import argparse
from pathlib import Path

import triton
from compile_prefill import HEAD_CONFIGS, KernelSpec, _archive_cubin, _make_source
from verification import gated_delta_rule_verification_kernel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=int, choices=(89,), default=89)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent.parent / "cubin"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = triton.backends.compiler.GPUTarget("cuda", args.arch, 32)
    for q_heads, v_heads in HEAD_CONFIGS:
        spec = KernelSpec(
            "verification",
            gated_delta_rule_verification_kernel,
            (
                "*bf16:16",
                "*bf16:16",
                "*bf16:16",
                "*fp32:16",
                "*fp32:16",
                "*bf16:16",
                "*fp32:16",
                "*fp32:16",
                "*i32:16",
                "*i32:16",
                "*i32:16",
                "*i32:16",
                "i64",
                "i32",
                "fp32",
                str(v_heads),
                str(q_heads),
                "128",
                "128",
                "16",
            ),
            4,
            1,
            v_heads,
            q_heads,
        )
        compiled = triton.compile(
            _make_source(spec), target=target, options={"num_warps": 4, "num_stages": 1}
        )
        # These values form the C++ runner ABI in gatedDeltaRuleDecodeCubins.cpp.
        if (
            compiled.metadata.shared != 4096
            or compiled.metadata.name != "gated_delta_rule_verification_kernel"
        ):
            raise RuntimeError("Verification kernel ABI changed; update the C++ cubin metadata")
        stem = f"gated_delta_rule_verification_bf16_h{q_heads}_hv{v_heads}_k128_v128_sm{args.arch}"
        path = args.output_dir / f"{stem}.cubin"
        path.write_bytes(compiled.asm["cubin"])
        _archive_cubin(path)
        path.unlink()
        print(
            stem, "shared=", compiled.metadata.shared, "name=", compiled.metadata.name, flush=True
        )


if __name__ == "__main__":
    main()
