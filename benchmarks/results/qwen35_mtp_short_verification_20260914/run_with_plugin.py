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

"""Run a Python module with an explicitly selected diagnostic plugin library."""

import ctypes
import os
import runpy
import sys
from pathlib import Path

plugin = Path(os.environ["QWEN35_TEST_PLUGIN"]).resolve()
original_cdll = ctypes.CDLL


def load_library(name, *args, **kwargs):
    if str(name).endswith("/libnvinfer_plugin_tensorrt_llm.so"):
        name = str(plugin)
    return original_cdll(name, *args, **kwargs)


ctypes.CDLL = load_library
module = sys.argv.pop(1)
print("PLUGIN", plugin, flush=True)
runpy.run_module(module, run_name="__main__")
