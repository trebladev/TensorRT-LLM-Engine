#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${project_dir}"

nccl_version="2.29.2"
job_count="${JOB_COUNT:-48}"
skip_python_env_check=false

usage() {
    cat <<'EOF'
Usage: build_trtllm_sm89.sh [OPTION]

Options:
  --skip-python-env-check  Skip Python environment setup/checks, invoke CMake
                           directly, and do not package a wheel.
  --check-python-env       Run the original Python environment setup/checks
                           and package wheels through scripts/build_wheel.py
                           (default).
  -h, --help               Show this help message.
EOF
}

while (($# > 0)); do
    case "$1" in
    --skip-python-env-check)
        skip_python_env_check=true
        ;;
    --check-python-env)
        skip_python_env_check=false
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Error: unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
done

if ! command -v ccache >/dev/null 2>&1; then
    echo "Error: ccache is not installed or is not in PATH." >&2
    exit 1
fi

if [[ "${skip_python_env_check}" == true ]]; then
    build_dir="${BUILD_DIR:-${project_dir}/cpp/build}"
    cmake_cache="${build_dir}/CMakeCache.txt"
    python_executable="${Python_EXECUTABLE:-${Python3_EXECUTABLE:-$(command -v python3)}}"
    nccl_root="${NCCL_ROOT:-}"

    torch_library_dir="$("${python_executable}" - <<'PY'
from pathlib import Path

import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

    if [[ ! -d "${torch_library_dir}" ]]; then
        echo "Error: PyTorch library directory was not found at ${torch_library_dir}." >&2
        exit 1
    fi

    export LD_LIBRARY_PATH="${torch_library_dir}:${LD_LIBRARY_PATH:-}"

    if [[ -z "${nccl_root}" && -f "${cmake_cache}" ]]; then
        nccl_root="$(sed -n 's/^NCCL_ROOT:[^=]*=//p' "${cmake_cache}" | head -n 1)"
    fi

    cmake_args=(
        -S "${project_dir}/cpp"
        -B "${build_dir}"
        -G Ninja
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CUDA_ARCHITECTURES=89-real
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
        -DBUILD_PYT=ON
        -DBUILD_DEEP_EP=ON
        -DBUILD_DEEP_GEMM=ON
        -DBUILD_FLASH_MLA=ON
        -DBUILD_WHEEL_TARGETS='tensorrt_llm;nvinfer_plugin_tensorrt_llm;th_common;bindings;deep_ep;deep_gemm;pg_utils;flash_mla;executorWorker'
        -DPython_EXECUTABLE="${python_executable}"
        -DPython3_EXECUTABLE="${python_executable}"
        -DNVRTC_DYNAMIC_LINKING=ON
        -DTensorRT_ROOT=/usr/local/tensorrt
    )

    conan_toolchain="${build_dir}/conan/conan_toolchain.cmake"
    if [[ -f "${conan_toolchain}" ]]; then
        cmake_args+=("-DCMAKE_TOOLCHAIN_FILE=${conan_toolchain}")
    fi

    if [[ -n "${nccl_root}" ]]; then
        nccl_library="${nccl_root}/lib/libnccl.so"
        if [[ ! -f "${nccl_library}" ]]; then
            nccl_library="${nccl_root}/lib/libnccl.so.2"
        fi
        nccl_include_dir="${nccl_root}/include"

        if [[ ! -f "${nccl_library}" || ! -f "${nccl_include_dir}/nccl.h" ]]; then
            echo "Error: NCCL library or headers were not found under ${nccl_root}." >&2
            exit 1
        fi

        export LD_LIBRARY_PATH="${nccl_root}/lib:${LD_LIBRARY_PATH:-}"
        cmake_args+=(
            "-DNCCL_ROOT=${nccl_root}"
            "-DNCCL_LIBRARY=${nccl_library}"
            "-DNCCL_INCLUDE_DIR=${nccl_include_dir}"
        )
    fi

    echo "Skipping Python environment setup/checks and invoking CMake directly."
    cmake "${cmake_args[@]}"
    cmake --build "${build_dir}" --config Release --parallel "${job_count}" \
        --target build_wheel_targets -- -d keepdepfile
    exit 0
fi

python3 -m pip install --no-deps --upgrade --force-reinstall \
    "nvidia-nccl-cu13==${nccl_version}"

nccl_root="$(python3 - <<'PY'
import importlib.metadata
from pathlib import Path

distribution = importlib.metadata.distribution("nvidia-nccl-cu13")
print(Path(distribution.locate_file("nvidia/nccl")).resolve())
PY
)"

nccl_library="${nccl_root}/lib/libnccl.so"
nccl_include_dir="${nccl_root}/include"

if [[ ! -f "${nccl_root}/lib/libnccl.so.2" || ! -f "${nccl_include_dir}/nccl.h" ]]; then
    echo "Error: NCCL library or headers were not found under ${nccl_root}." >&2
    exit 1
fi

ln -sfn libnccl.so.2 "${nccl_library}"

export NCCL_ROOT="${nccl_root}"
export LD_LIBRARY_PATH="${NCCL_ROOT}/lib:${LD_LIBRARY_PATH:-}"
export TRTLLM_BUILD_JOB_COUNT="${job_count}"
export TRTLLM_NCCL_VERSION="${nccl_version}"

python3 - <<'PY'
import builtins
import importlib.util
import os
import subprocess
from pathlib import Path

project_dir = Path.cwd()
build_script = project_dir / "scripts/build_wheel.py"
module_spec = importlib.util.spec_from_file_location("trtllm_build_wheel", build_script)
build_wheel = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(build_wheel)

original_setup_venv = build_wheel.setup_venv
nccl_version = os.environ["TRTLLM_NCCL_VERSION"]
nccl_root = Path(os.environ["NCCL_ROOT"])


def setup_venv_with_nccl_restore(*args, **kwargs):
    original_input = builtins.input
    builtins.input = lambda prompt="": print(prompt, end="") or "continue"
    try:
        venv_python, venv_conan = original_setup_venv(*args, **kwargs)
    finally:
        builtins.input = original_input

    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--force-reinstall",
            f"nvidia-nccl-cu13=={nccl_version}",
        ],
        check=True,
    )
    (nccl_root / "lib/libnccl.so").unlink(missing_ok=True)
    (nccl_root / "lib/libnccl.so").symlink_to("libnccl.so.2")

    torch_library_dir = subprocess.check_output(
        [
            str(venv_python),
            "-c",
            "from pathlib import Path; import torch; "
            "print(Path(torch.__file__).resolve().parent / 'lib')",
        ],
        text=True,
    ).strip()
    os.environ["LD_LIBRARY_PATH"] = (
        f"{torch_library_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )
    return venv_python, venv_conan


build_wheel.setup_venv = setup_venv_with_nccl_restore
build_wheel.main(
    use_ccache=True,
    cuda_architectures="89-real",
    skip_building_wheel=False,
    linking_install_binary=False,
    nccl_root=str(nccl_root),
    extra_cmake_vars=[
        f"NCCL_LIBRARY={nccl_root / 'lib/libnccl.so'}",
        f"NCCL_INCLUDE_DIR={nccl_root / 'include'}",
    ],
    generator="Ninja",
    job_count=int(os.environ["TRTLLM_BUILD_JOB_COUNT"]),
    clean=False,
    nvrtc_dynamic_linking=True,
)
PY
