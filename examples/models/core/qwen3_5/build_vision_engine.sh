#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

python_bin="${QWEN35_PYTHON:-python}"
model_dir="${QWEN35_MODEL_DIR:-/root/code_x/Qwen3.5-2B}"
vision_engine_dir="${1:-/tmp/qwen35_vision_engine}"
extra_args=("${@:2}")
plugin_lib="${QWEN35_PLUGIN_LIB:-${repo_root}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"

if [[ ! -d "${model_dir}" ]]; then
    printf 'Qwen3.5 model directory does not exist: %s\n' "${model_dir}" >&2
    exit 1
fi
if [[ ! -f "${plugin_lib}" ]]; then
    printf 'TensorRT-LLM source plugin does not exist: %s\n' "${plugin_lib}" >&2
    printf 'Build it with: cmake --build cpp/build --target bindings --parallel 8\n' >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
torch_lib_dir="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib_dir}:${repo_root}/cpp/build/tensorrt_llm:${repo_root}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRT_LLM_NO_LIB_INIT=1
export QWEN35_PLUGIN_LIB="${plugin_lib}"

cd "${repo_root}"

"${python_bin}" -c '
import ctypes
import os
import runpy
import sys
import traceback

# Load the Python TensorRT runtime before resolving plugin dependencies.
import tensorrt

plugin = ctypes.CDLL(os.environ["QWEN35_PLUGIN_LIB"], mode=ctypes.RTLD_GLOBAL)
plugin.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
plugin.initTrtLlmPlugins.restype = ctypes.c_bool
if not plugin.initTrtLlmPlugins(None, b"tensorrt_llm"):
    raise RuntimeError("Failed to initialize TensorRT-LLM source plugins")

exit_code = 0
try:
    runpy.run_module("examples.models.core.qwen3_5.build_vision_engine", run_name="__main__")
except SystemExit as error:
    exit_code = error.code if isinstance(error.code, int) else 1
except BaseException:
    traceback.print_exc()
    exit_code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
' \
    --model_dir "${model_dir}" \
    --vision_engine_dir "${vision_engine_dir}" \
    --vision_workspace_gb "${QWEN35_VISION_WORKSPACE_GB:-12}" \
    --vision_builder_optimization_level "${QWEN35_VISION_BUILDER_OPTIMIZATION_LEVEL:-0}" \
    "${extra_args[@]}"
