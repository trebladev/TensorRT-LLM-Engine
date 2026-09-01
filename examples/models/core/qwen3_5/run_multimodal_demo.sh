#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 4 ]]; then
    printf 'Usage: %s MODEL_DIR LLM_ENGINE_DIR VISION_ENGINE_DIR IMAGE [DEMO_ARGS...]\n' "$0" >&2
    exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

model_dir="$1"
llm_engine_dir="$2"
vision_engine_dir="$3"
image="$4"
shift 4

python_bin="${QWEN35_PYTHON:-python}"
plugin_lib="${QWEN35_PLUGIN_LIB:-${repo_root}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"

if [[ ! -f "${plugin_lib}" ]]; then
    printf 'TensorRT-LLM source plugin does not exist: %s\n' "${plugin_lib}" >&2
    printf 'Build it with: cmake --build cpp/build --target bindings --parallel 8\n' >&2
    exit 1
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
torch_lib_dir="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib_dir}:${repo_root}/cpp/build/tensorrt_llm:${repo_root}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRT_LLM_NO_LIB_INIT=1
export QWEN35_PLUGIN_LIB="${plugin_lib}"
export QWEN35_EXIT_WITHOUT_RUNTIME_CLEANUP=1
cd "${repo_root}"

"${python_bin}" -c '
import ctypes
import os
import runpy
import sys
import traceback
import torch

plugin = ctypes.CDLL(os.environ["QWEN35_PLUGIN_LIB"], mode=ctypes.RTLD_GLOBAL)
plugin.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
plugin.initTrtLlmPlugins.restype = ctypes.c_bool
if not plugin.initTrtLlmPlugins(None, b"tensorrt_llm"):
    raise RuntimeError("Failed to initialize TensorRT-LLM source plugins")

exit_code = 0
try:
    runpy.run_path(
        "examples/models/core/qwen3_5/multimodal_demo.py",
        run_name="__main__",
    )
except BaseException:
    traceback.print_exc()
    exit_code = 1
finally:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
' \
    --model_dir "${model_dir}" \
    --llm_engine_dir "${llm_engine_dir}" \
    --vision_engine_dir "${vision_engine_dir}" \
    --image "${image}" \
    "$@"
