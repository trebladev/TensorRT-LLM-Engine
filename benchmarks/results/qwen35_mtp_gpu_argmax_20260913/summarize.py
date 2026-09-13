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

"""Summarize two fixed-engine benchmark passes and verify exact MTP output equality."""

import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    rows = {}
    for mode in ("before", "argmax", "plain"):
        rows[mode] = [
            row
            for run in range(2)
            for row in json.loads((root / f"{mode}_{run}.json").read_text())["rows"]
        ]
    sequences = tokens = 0
    for old, new in zip(rows["before"], rows["argmax"], strict=True):
        for key in (
            "isl",
            "concurrency",
            "repeat",
            "input_ids",
            "output_ids",
            "accepted",
            "proposals",
        ):
            if old[key] != new[key]:
                raise ValueError(
                    f"MTP mismatch for {key}: ISL {old['isl']}, batch {old['concurrency']}"
                )
        sequences += len(old["output_ids"])
        tokens += sum(map(len, old["output_ids"]))
    summary = []
    for isl in (32, 64):
        for batch in (1, 2, 4):
            entry = {"isl": isl, "concurrency": batch}
            for mode, measurements in rows.items():
                selected = [
                    r for r in measurements if r["isl"] == isl and r["concurrency"] == batch
                ]
                count = sum(sum(map(len, r["output_ids"])) for r in selected)
                entry[f"{mode}_tokens_s"] = count / sum(r["wall_s"] for r in selected)
                entry[f"{mode}_tpot_ms"] = statistics.median(
                    t for r in selected for t in r["tpot_ms"]
                )
            entry["improvement_pct"] = (
                entry["argmax_tokens_s"] / entry["before_tokens_s"] - 1
            ) * 100
            entry["vs_plain_pct"] = (entry["argmax_tokens_s"] / entry["plain_tokens_s"] - 1) * 100
            summary.append(entry)
    with (root / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(
        json.dumps(
            {"identical_sequences": sequences, "identical_tokens": tokens, "summary": summary},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
