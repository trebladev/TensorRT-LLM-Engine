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

"""Run the standard benchmark with an explicitly loaded isolated runtime."""

import os
from pathlib import Path


def main() -> None:
    import tensorrt as trt
    import torch

    print("ENVIRONMENT", trt.__version__, torch.cuda.is_available(), flush=True)
    runtime = os.environ.get("TLLM_MTP_EXPERIMENT_RUNTIME")
    from examples.models.core.qwen3_5.mtp_benchmark import main as benchmark

    loaded = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libtensorrt_llm.so" in line
    }
    print("LOADED_RUNTIME", sorted(loaded), flush=True)
    if runtime and loaded != {str(Path(runtime).resolve())}:
        raise RuntimeError(f"Unexpected runtime libraries: {loaded}")
    print(
        "EXPERIMENT_FLAGS",
        {key: value for key, value in os.environ.items() if key.startswith("TLLM_MTP_EXPERIMENT_")},
        flush=True,
    )
    benchmark()


if __name__ == "__main__":
    main()
