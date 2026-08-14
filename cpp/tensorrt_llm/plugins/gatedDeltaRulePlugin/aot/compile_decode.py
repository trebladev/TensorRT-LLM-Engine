#!/usr/bin/env python3
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

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import triton
import triton.backends
import triton.language as tl

NUM_Q_HEADS = 16
NUM_V_HEADS = (16, 32, 48)
HEAD_K_DIM = 128
HEAD_V_DIM = 128
BLOCK_K = 128
BLOCK_V = 8
NUM_WARPS = 1
NUM_STAGES = 3
EXPECTED_SHARED_MEMORY = 16
KERNEL_NAME = "fused_recurrent_gated_delta_rule_update_fwd_kernel"


def _load_pytorch_kernel(repo_root: Path) -> triton.runtime.JITFunction:
    package_names = (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.modules",
        "tensorrt_llm._torch.modules.fla",
    )
    for package_name in package_names:
        package = types.ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package

    op_module = types.ModuleType("tensorrt_llm._torch.modules.fla.op")
    op_module.exp = tl.exp
    sys.modules[op_module.__name__] = op_module

    utils_module = types.ModuleType("tensorrt_llm._torch.modules.fla.utils")
    utils_module.input_guard = lambda function: function
    sys.modules[utils_module.__name__] = utils_module

    module_name = "tensorrt_llm._torch.modules.fla.fused_recurrent"
    kernel_path = repo_root / "tensorrt_llm/_torch/modules/fla/fused_recurrent.py"
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Triton kernel source from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, KERNEL_NAME).fn


def _make_source(kernel: triton.runtime.JITFunction, num_v_heads: int) -> triton.compiler.ASTSource:
    argument_types = (
        "*bf16:16",
        "*bf16:16",
        "*bf16:16",
        "*fp32:16",
        "*fp32:16",
        "*bf16:16",
        "*fp32:16",
        "*i32:16",
        "*i32:16",
        "fp32",
        "*fp32:16",
        "i32",
        "i32",
        "1",
        str(NUM_Q_HEADS),
        str(num_v_heads),
        str(HEAD_K_DIM),
        str(HEAD_V_DIM),
        str(BLOCK_K),
        str(BLOCK_V),
        "1",
        "0",
        "1",
        "1",
        "0",
        "0",
        "0",
    )
    if len(argument_types) != len(kernel.arg_names):
        raise RuntimeError(
            f"Kernel ABI changed: expected {len(argument_types)} arguments, got {len(kernel.arg_names)}"
        )

    constants: dict[str, int] = {}
    signature: dict[str, str] = {}
    attributes: dict[tuple[int], list[list[object]]] = {}
    for index, (argument_name, argument_type) in enumerate(zip(kernel.arg_names, argument_types)):
        try:
            constant = int(argument_type)
        except ValueError:
            constant = None
        if constant is not None:
            constants[argument_name] = constant
            signature[argument_name] = "constexpr"
        else:
            signature[argument_name] = argument_type.split(":")[0]
            if argument_type.endswith(":16"):
                attributes[(index,)] = [["tt.divisibility", 16]]

    return triton.compiler.ASTSource(
        fn=kernel,
        constexprs=constants,
        signature=signature,
        attrs=attributes,
    )


def _archive_cubin(cubin_path: Path) -> Path:
    archive_path = cubin_path.with_suffix(".cubin.tar.zst")
    subprocess.run(
        [
            "cmake",
            "-E",
            "tar",
            "cf",
            archive_path.name,
            "--zstd",
            "--mtime=1970-01-01UTC",
            "--",
            cubin_path.name,
        ],
        cwd=cubin_path.parent,
        check=True,
    )
    return archive_path


def _compile_variant(
    kernel: triton.runtime.JITFunction, arch: int, num_v_heads: int, output_dir: Path
) -> Path:
    target = triton.backends.compiler.GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": NUM_WARPS, "num_stages": NUM_STAGES})
    compiled = triton.compile(
        _make_source(kernel, num_v_heads),
        target=target,
        options=options.__dict__,
    )
    if compiled.metadata.name != KERNEL_NAME:
        raise RuntimeError(
            f"Unexpected kernel symbol {compiled.metadata.name}; expected {KERNEL_NAME}"
        )
    if compiled.metadata.shared != EXPECTED_SHARED_MEMORY:
        raise RuntimeError(
            f"Shared-memory usage changed to {compiled.metadata.shared}; "
            f"update the C++ cubin metadata before regenerating"
        )

    stem = (
        f"gated_delta_rule_decode_bf16_h{NUM_Q_HEADS}_hv{num_v_heads}"
        f"_k{HEAD_K_DIM}_v{HEAD_V_DIM}_sm{arch}"
    )
    cubin_path = output_dir / f"{stem}.cubin"
    cubin_path.write_bytes(compiled.asm[backend.binary_ext])
    archive_path = _archive_cubin(cubin_path)
    cubin_path.unlink()
    print(f"Generated {archive_path} ({archive_path.stat().st_size} bytes)")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT compile GatedDeltaRule decode cubins")
    parser.add_argument("--arch", type=int, choices=(89,), default=89, help="CUDA SM architecture")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "cubin",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernel = _load_pytorch_kernel(repo_root)
    for num_v_heads in NUM_V_HEADS:
        _compile_variant(kernel, args.arch, num_v_heads, args.output_dir.resolve())


if __name__ == "__main__":
    main()
