#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

checkpoint_dir="${1:-${QWEN35_CKPT_DIR:-/tmp/qwen35_bf16_tp2}}"
engine_dir="${2:-${QWEN35_ENGINE_DIR:-/tmp/qwen35_engine_bf16_tp2}}"
extra_args=("${@:3}")

exec "${script_dir}/build.sh" \
    "${checkpoint_dir}" \
    "${engine_dir}" \
    "${extra_args[@]}"
