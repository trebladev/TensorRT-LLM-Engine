#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

engine_dir="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi
model_dir="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi

input_text=""
if [[ $# -gt 0 && "${1}" != --* ]]; then
    input_text="${1}"
    shift
fi

export QWEN35_RUN_SCRIPT="${script_dir}/prefix_cache_demo.py"
export QWEN35_USE_CHAT_TEMPLATE=0

exec "${script_dir}/run_demo.sh" \
    "${engine_dir}" \
    "${model_dir}" \
    "${input_text}" \
    "$@" \
    --kv_cache_enable_block_reuse
