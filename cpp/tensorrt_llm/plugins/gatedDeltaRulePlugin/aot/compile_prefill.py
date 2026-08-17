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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import triton
import triton.backends
import triton.language as tl

NUM_Q_HEADS = 16
NUM_V_HEADS = (16, 32, 48)
HEAD_K_DIM = 128
HEAD_V_DIM = 128
CHUNK_SIZE = 64


@triton.jit
def _safe_exp(x):
    return tl.exp(tl.where(x <= 0, x, float("-inf")))


@dataclass(frozen=True)
class KernelSpec:
    name: str
    kernel: Any
    argument_types: tuple[str, ...]
    num_warps: int
    num_stages: int
    num_v_heads: int = 0


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load Triton source from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _unwrap_kernel(kernel: Any) -> triton.runtime.JITFunction:
    while not isinstance(kernel, triton.runtime.JITFunction):
        if not hasattr(kernel, "fn"):
            raise TypeError(f"Cannot unwrap Triton kernel of type {type(kernel)}")
        kernel = kernel.fn
    return kernel


def _load_kernels(repo_root: Path) -> dict[str, triton.runtime.JITFunction]:
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

    index_module = types.ModuleType("tensorrt_llm._torch.modules.fla.index")
    index_module.prepare_chunk_indices = lambda *args, **kwargs: None
    index_module.prepare_chunk_offsets = lambda *args, **kwargs: None
    sys.modules[index_module.__name__] = index_module

    utils_module = types.ModuleType("tensorrt_llm._torch.modules.fla.utils")
    utils_module.check_shared_mem = lambda *args, **kwargs: True
    utils_module.input_guard = lambda function: function
    utils_module.is_nvidia_hopper = False
    utils_module.is_tf32_supported = True
    sys.modules[utils_module.__name__] = utils_module

    op_module = types.ModuleType("tensorrt_llm._torch.modules.fla.op")
    op_module.exp = tl.exp
    op_module.safe_exp = _safe_exp
    sys.modules[op_module.__name__] = op_module

    fla_root = repo_root / "tensorrt_llm/_torch/modules/fla"
    wy_module = _load_module("tensorrt_llm._torch.modules.fla.wy_fast", fla_root / "wy_fast.py")
    modules = {
        "l2norm": _load_module("tensorrt_llm._torch.modules.fla.l2norm", fla_root / "l2norm.py"),
        "cumsum": _load_module("tensorrt_llm._torch.modules.fla.cumsum", fla_root / "cumsum.py"),
        "kkt_solve": _load_module(
            "tensorrt_llm._torch.modules.fla.chunk_fwd", fla_root / "chunk_fwd.py"
        ),
        "recompute": wy_module,
        "state": _load_module(
            "tensorrt_llm._torch.modules.fla.chunk_delta_h", fla_root / "chunk_delta_h.py"
        ),
        "output": _load_module("tensorrt_llm._torch.modules.fla.chunk_o", fla_root / "chunk_o.py"),
        "aux": _load_module(
            "gated_delta_rule_prefill_aux", Path(__file__).with_name("prefill_aux.py")
        ),
    }
    return {
        "l2norm": _unwrap_kernel(modules["l2norm"].l2norm_fwd_kernel),
        "init_chunks": _unwrap_kernel(modules["aux"].init_chunk_indices_kernel),
        "prepare_chunks": _unwrap_kernel(modules["aux"].prepare_chunk_metadata_kernel),
        "zero_state": _unwrap_kernel(modules["aux"].zero_missing_states_kernel),
        "gather_state": _unwrap_kernel(modules["aux"].gather_states_kernel),
        "cumsum": _unwrap_kernel(modules["cumsum"].chunk_local_cumsum_scalar_kernel),
        "kkt_solve": _unwrap_kernel(
            modules["kkt_solve"].chunk_gated_delta_rule_fwd_kkt_solve_kernel
        ),
        "recompute": _unwrap_kernel(modules["recompute"].recompute_w_u_fwd_kernel),
        "state": _unwrap_kernel(modules["state"].chunk_gated_delta_rule_fwd_kernel_h_blockdim64),
        "output": _unwrap_kernel(modules["output"].chunk_fwd_kernel_o),
    }


def _make_source(spec: KernelSpec) -> triton.compiler.ASTSource:
    kernel = spec.kernel
    if len(spec.argument_types) != len(kernel.arg_names):
        raise RuntimeError(
            f"{spec.name} ABI changed: expected {len(spec.argument_types)} arguments, "
            f"got {len(kernel.arg_names)} ({kernel.arg_names})"
        )

    constants: dict[str, int] = {}
    signature: dict[str, str] = {}
    attributes: dict[tuple[int], list[list[object]]] = {}
    for index, (argument_name, argument_type) in enumerate(
        zip(kernel.arg_names, spec.argument_types)
    ):
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


def _kernel_specs(kernels: dict[str, triton.runtime.JITFunction]) -> list[KernelSpec]:
    specs = [
        KernelSpec(
            "l2norm",
            kernels["l2norm"],
            ("*bf16:16", "*bf16:16", "fp32", "i32", "128", "16", "128"),
            8,
            3,
        ),
        KernelSpec(
            "init_chunks",
            kernels["init_chunks"],
            ("*i32:16", "i32", "256"),
            1,
            1,
        ),
        KernelSpec(
            "prepare_chunks",
            kernels["prepare_chunks"],
            ("*i32:16", "*i32:16", "*i32:16", "*i32:16", "i32", "64"),
            1,
            1,
        ),
    ]

    for num_v_heads in NUM_V_HEADS:
        specs.extend(
            [
                KernelSpec(
                    "zero_state",
                    kernels["zero_state"],
                    (
                        "*fp32:16",
                        "*i32:16",
                        "*i8:16",
                        "i32",
                        str(num_v_heads),
                        "128",
                        "128",
                        "64",
                    ),
                    4,
                    1,
                    num_v_heads,
                ),
                KernelSpec(
                    "gather_state",
                    kernels["gather_state"],
                    (
                        "*fp32:16",
                        "*fp32:16",
                        "*i32:16",
                        "i32",
                        str(num_v_heads),
                        "128",
                        "128",
                        "64",
                    ),
                    4,
                    1,
                    num_v_heads,
                ),
                KernelSpec(
                    "cumsum",
                    kernels["cumsum"],
                    (
                        "*fp32:16",
                        "*fp32:16",
                        "0",
                        "*i32:16",
                        "*i32:16",
                        "i32",
                        "1",
                        str(num_v_heads),
                        "64",
                        "0",
                        "0",
                        "1",
                        "0",
                    ),
                    8,
                    3,
                    num_v_heads,
                ),
                KernelSpec(
                    "kkt_solve",
                    kernels["kkt_solve"],
                    (
                        "*bf16:16",
                        "*fp32:16",
                        "*fp32:16",
                        "*bf16:16",
                        "*i32:16",
                        "*i32:16",
                        "i32",
                        str(num_v_heads),
                        "16",
                        "128",
                        "64",
                        "16",
                        "64",
                        "1",
                        "1",
                    ),
                    4,
                    2,
                    num_v_heads,
                ),
                KernelSpec(
                    "recompute",
                    kernels["recompute"],
                    (
                        "*bf16:16",
                        "*bf16:16",
                        "*fp32:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*fp32:16",
                        "*i32:16",
                        "*i32:16",
                        "i32",
                        str(num_v_heads),
                        "16",
                        "128",
                        "128",
                        "64",
                        "64",
                        "64",
                        "1",
                    ),
                    4,
                    3,
                    num_v_heads,
                ),
                KernelSpec(
                    "state",
                    kernels["state"],
                    (
                        "*bf16:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*fp32:16",
                        "*bf16:16",
                        "*fp32:16",
                        "*i32:16",
                        "0",
                        "*i32:16",
                        "*i32:16",
                        "i32",
                        "i64",
                        str(num_v_heads),
                        "16",
                        "128",
                        "128",
                        "64",
                        "32",
                        "1",
                        "1",
                        "1",
                        "0",
                        "1",
                        "1",
                    ),
                    4,
                    3,
                    num_v_heads,
                ),
                KernelSpec(
                    "output",
                    kernels["output"],
                    (
                        "*bf16:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*bf16:16",
                        "*fp32:16",
                        "*bf16:16",
                        "*i32:16",
                        "*i32:16",
                        "fp32",
                        "i32",
                        str(num_v_heads),
                        "16",
                        "128",
                        "128",
                        "64",
                        "64",
                        "64",
                        "1",
                        "1",
                    ),
                    4,
                    3,
                    num_v_heads,
                ),
            ]
        )
    return specs


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


def _compile_spec(spec: KernelSpec, arch: int, output_dir: Path) -> Path:
    target = triton.backends.compiler.GPUTarget("cuda", arch, 32)
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options({"num_warps": spec.num_warps, "num_stages": spec.num_stages})
    compiled = triton.compile(_make_source(spec), target=target, options=options.__dict__)
    hv_suffix = f"_hv{spec.num_v_heads}" if spec.num_v_heads else ""
    stem = f"gated_delta_rule_prefill_{spec.name}_bf16_h16{hv_suffix}_k128_v128_sm{arch}"
    cubin_path = output_dir / f"{stem}.cubin"
    cubin_path.write_bytes(compiled.asm[backend.binary_ext])
    archive_path = _archive_cubin(cubin_path)
    cubin_path.unlink()
    print(
        f"Generated {archive_path.name}: symbol={compiled.metadata.name}, "
        f"shared={compiled.metadata.shared}, warps={spec.num_warps}"
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="AOT compile GatedDeltaRule prefill cubins")
    parser.add_argument("--arch", type=int, choices=(89,), default=89)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "cubin",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[5]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    kernels = _load_kernels(repo_root)
    for spec in _kernel_specs(kernels):
        _compile_spec(spec, args.arch, args.output_dir.resolve())


if __name__ == "__main__":
    main()
