#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

checkpoint_dir="${1:-${QWEN35_CKPT_DIR:-/tmp/qwen35_ckpt_tp2}}"
engine_dir="${2:-${QWEN35_ENGINE_DIR:-/tmp/qwen35_engine_long_ttft_tp2}}"
extra_args=("${@:3}")

export CUDA_VISIBLE_DEVICES="${QWEN35_CUDA_VISIBLE_DEVICES:-0,1}"
export QWEN35_MAX_BATCH_SIZE="${QWEN35_MAX_BATCH_SIZE:-1}"
export QWEN35_MAX_INPUT_LEN="${QWEN35_MAX_INPUT_LEN:-102400}"
export QWEN35_MAX_SEQ_LEN="${QWEN35_MAX_SEQ_LEN:-102408}"
export QWEN35_MAX_NUM_TOKENS="${QWEN35_MAX_NUM_TOKENS:-4096}"
export QWEN35_OPT_NUM_TOKENS="${QWEN35_OPT_NUM_TOKENS:-4096}"

exec "${script_dir}/build_tp2.sh" \
    "${checkpoint_dir}" \
    "${engine_dir}" \
    "${extra_args[@]}"
