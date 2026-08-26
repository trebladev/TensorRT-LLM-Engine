#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

python_bin="${QWEN35_PYTHON:-python}"
model_dir="${1:-${QWEN35_MODEL_DIR:-/root/code_x/Qwen3.5-2B}}"
checkpoint_dir="${2:-${QWEN35_CKPT_DIR:-/tmp/qwen35_bf16_tp1}}"
extra_args=("${@:3}")
tp_size="${QWEN35_TP_SIZE:-1}"

if [[ ! -d "${model_dir}" ]]; then
    printf 'Qwen3.5 model directory does not exist: %s\n' "${model_dir}" >&2
    exit 1
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${repo_root}"

"${python_bin}" examples/models/core/qwen3_5/convert_checkpoint.py \
    --model_dir "${model_dir}" \
    --output_dir "${checkpoint_dir}" \
    --dtype bfloat16 \
    --tp_size "${tp_size}" \
    --pp_size 1 \
    --cp_size 1 \
    "${extra_args[@]}"
