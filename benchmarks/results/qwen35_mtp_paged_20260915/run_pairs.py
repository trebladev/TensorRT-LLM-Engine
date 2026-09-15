# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paired runtime measurements before and after persistent paged draft KV."""

import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    report = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[3]
    engine_root = root / "engines/qwen35"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="0",
        LLM_MODELS_ROOT="/root/code_x",
        PYTHONPATH=str(root),
        LD_LIBRARY_PATH=str(engine_root / "runtime"),
        TRTLLM_QWEN35_MTP_DISABLE_DRAFT_BATCHING="0",
    )
    statuses = []
    for trial, modes in enumerate((("old", "new"), ("new", "old"))):
        for mode in modes:
            name = f"focused_{mode}_{trial}"
            plugin = root / "cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so"
            runtime = engine_root / ("runtime_paged" if mode == "new" else "runtime_flow12")
            run_env = env | {
                "LD_LIBRARY_PATH": str(runtime),
                "TLLM_MTP_EXPERIMENT_RUNTIME": str(runtime / "libtensorrt_llm.so"),
                "QWEN35_TEST_PLUGIN": str(plugin),
            }
            command = [
                sys.executable,
                str(report / "run_benchmark.py"),
                "--model_dir",
                "/root/code_x/Qwen3.5-2B",
                "--engine_dir",
                str(engine_root / ("mtp_paged" if mode == "new" else "mtp")),
                "--mode",
                "plain" if mode == "plain" else "mtp",
                "--output",
                str(report / f"{name}.json"),
                "--warmup",
                "3",
                "--repeats",
                "6",
                "--isl",
                "32",
                "64",
                "--concurrency",
                "4",
                "--osl",
                "64",
            ]
            print("START", name, flush=True)
            with (report / f"{name}.log").open("w") as log:
                result = subprocess.run(
                    command,
                    env=run_env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            statuses.append(
                dict(
                    name=name,
                    plugin=str(plugin),
                    runtime=str(runtime),
                    command=command,
                    returncode=result.returncode,
                )
            )
            (report / "focused_status.json").write_text(json.dumps(statuses, indent=2) + "\n")
            print("END", name, result.returncode, flush=True)
            if result.returncode:
                raise RuntimeError(name)


if __name__ == "__main__":
    main()
