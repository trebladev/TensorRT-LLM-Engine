#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

python_bin="${QWEN35_PYTHON:-python}"
checkpoint_dir="${1:-${QWEN35_CKPT_DIR:-/tmp/qwen35_bf16_tp1}}"
engine_dir="${2:-${QWEN35_ENGINE_DIR:-/tmp/qwen35_engine_bf16_tp1}}"
extra_args=("${@:3}")
plugin_lib="${QWEN35_PLUGIN_LIB:-${repo_root}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"

max_batch_size="${QWEN35_MAX_BATCH_SIZE:-4}"
max_input_len="${QWEN35_MAX_INPUT_LEN:-4096}"
max_seq_len="${QWEN35_MAX_SEQ_LEN:-8192}"
max_num_tokens="${QWEN35_MAX_NUM_TOKENS:-8192}"
opt_num_tokens="${QWEN35_OPT_NUM_TOKENS:-4096}"

if [[ ! -f "${checkpoint_dir}/config.json" ]]; then
    printf 'TensorRT-LLM checkpoint config does not exist: %s/config.json\n' "${checkpoint_dir}" >&2
    exit 1
fi
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
cd "${repo_root}"

"${python_bin}" -c '
import ctypes
import os
import runpy
import sys
import torch

plugin = ctypes.CDLL(os.environ["QWEN35_PLUGIN_LIB"], mode=ctypes.RTLD_GLOBAL)
plugin.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
plugin.initTrtLlmPlugins.restype = ctypes.c_bool
if not plugin.initTrtLlmPlugins(None, b"tensorrt_llm"):
    raise RuntimeError("Failed to initialize TensorRT-LLM source plugins")
runpy.run_module("tensorrt_llm.commands.build", run_name="__main__")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
' \
    --checkpoint_dir "${checkpoint_dir}" \
    --output_dir "${engine_dir}" \
    --max_batch_size "${max_batch_size}" \
    --max_input_len "${max_input_len}" \
    --max_seq_len "${max_seq_len}" \
    --max_num_tokens "${max_num_tokens}" \
    --opt_num_tokens "${opt_num_tokens}" \
    --max_beam_width 1 \
    --kv_cache_type paged \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --mamba_conv1d_plugin bfloat16 \
    --context_fmha enable \
    --remove_input_padding enable \
    "${extra_args[@]}"
