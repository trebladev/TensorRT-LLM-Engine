#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

model_dir="${1:-${QWEN35_MODEL_DIR:-/root/code_x/Qwen3.5-2B}}"
checkpoint_dir="${2:-${QWEN35_CKPT_DIR:-/tmp/qwen35_bf16_tp2}}"
extra_args=("${@:3}")

export QWEN35_TP_SIZE=2
exec "${script_dir}/convert.sh" \
    "${model_dir}" \
    "${checkpoint_dir}" \
    "${extra_args[@]}"
