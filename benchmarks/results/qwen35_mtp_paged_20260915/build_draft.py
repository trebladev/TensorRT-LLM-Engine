# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rebuild only the draft; reuse the identical target for paired comparisons."""

import shutil
from pathlib import Path

from examples.models.core.qwen3_5.mtp_demo import save_paged_draft_engine
from tensorrt_llm.models.qwen35.mtp import Qwen35MTP

engine_root = Path(__file__).resolve().parents[3] / "engines/qwen35"
output = engine_root / "mtp_paged"
output.mkdir(parents=True, exist_ok=True)
model = Qwen35MTP.from_hugging_face("/root/code_x/Qwen3.5-2B")
save_paged_draft_engine(model, output / "mtp.engine", 129, 4)
del model
shutil.copyfile(engine_root / "mtp/config.json", output / "config.json")
shutil.copyfile(engine_root / "mtp/rank0.engine", output / "rank0.engine")
print("BUILT", output, flush=True)
