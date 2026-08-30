#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

engine_dir="${1:-${QWEN35_ENGINE_DIR:-/tmp/qwen35_engine_long_ttft_tp2}}"
if [[ $# -gt 0 ]]; then
    shift
fi
model_dir="${1:-${QWEN35_MODEL_DIR:-/root/code_x/Qwen3.5-2B}}"
if [[ $# -gt 0 ]]; then
    shift
fi

export CUDA_VISIBLE_DEVICES="${QWEN35_CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN35_RUN_SCRIPT="${script_dir}/long_context_ttft.py"
export QWEN35_USE_CHAT_TEMPLATE=0
export QWEN35_MAX_INPUT_LEN="${QWEN35_MAX_INPUT_LEN:-102400}"
export QWEN35_MAX_OUTPUT_LEN="${QWEN35_MAX_OUTPUT_LEN:-8}"
export QWEN35_KV_CACHE_FREE_GPU_MEMORY_FRACTION="${QWEN35_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.7}"
num_trials="${QWEN35_NUM_TRIALS:-3}"
warmup_requests="${QWEN35_WARMUP_REQUESTS:-1}"

exec "${script_dir}/run_demo.sh" \
    "${engine_dir}" \
    "${model_dir}" \
    "long-context-ttft" \
    --prefix_tokens 51200 \
    --long_tokens 102400 \
    --num_trials "${num_trials}" \
    --warmup_requests "${warmup_requests}" \
    --kv_cache_enable_block_reuse \
    "$@"
