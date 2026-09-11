#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

python_bin="${QWEN35_PYTHON:-python}"
checkpoint_dir="${1:-/tmp/qwen35_bf16_tp1}"
engine_dir="${2:-/tmp/qwen35_decode_engine}"
extra_args=("${@:3}")
plugin_lib="${QWEN35_PLUGIN_LIB:-${repo_root}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"
build_py="${repo_root}/tensorrt_llm/commands/build.py"

if [[ ! -f "${checkpoint_dir}/config.json" ]]; then
    printf 'TensorRT-LLM checkpoint config does not exist: %s/config.json\n' "${checkpoint_dir}" >&2
    exit 1
fi
if [[ ! -f "${plugin_lib}" ]]; then
    printf 'TensorRT-LLM source plugin does not exist: %s\n' "${plugin_lib}" >&2
    printf 'Build it with: cmake --build cpp/build --target bindings --parallel 8\n' >&2
    exit 1
fi
if [[ ! -f "${build_py}" ]]; then
    printf 'TensorRT-LLM build.py does not exist: %s\n' "${build_py}" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
torch_lib_dir="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib_dir}:${repo_root}/cpp/build/tensorrt_llm:${repo_root}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRT_LLM_NO_LIB_INIT=1
export QWEN35_BUILD_PY="${build_py}"
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
    runpy.run_path(os.environ["QWEN35_BUILD_PY"], run_name="__main__")
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
    --checkpoint_dir "${checkpoint_dir}" \
    --output_dir "${engine_dir}" \
    --max_batch_size 1 \
    --max_input_len 16384 \
    --max_seq_len 16384 \
    --max_num_tokens 4096 \
    --opt_num_tokens 4096 \
    --max_beam_width 1 \
    --max_prompt_embedding_table_size 16384 \
    --kv_cache_type paged \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --mamba_conv1d_plugin bfloat16 \
    --context_fmha enable \
    --use_paged_context_fmha enable \
    --remove_input_padding enable \
    "${extra_args[@]}"
