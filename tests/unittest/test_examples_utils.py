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

import json
from pathlib import Path

from examples.utils import read_model_name


def test_read_model_name_qwen35_without_qwen_type(tmp_path: Path) -> None:
    config = {
        "version": "1.3.0",
        "pretrained_config": {
            "architecture": "Qwen35ForCausalLM",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert read_model_name(str(tmp_path)) == ("Qwen35ForCausalLM", None)
