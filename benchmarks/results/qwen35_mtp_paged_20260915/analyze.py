# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize paired end-to-end timings and compare generated token IDs."""

import json
from pathlib import Path
from statistics import mean


def main() -> None:
    report = Path(__file__).resolve().parent
    data = {
        (mode, trial): json.loads((report / f"focused_{mode}_{trial}.json").read_text())
        for mode in ("old", "new")
        for trial in range(2)
    }
    summary = []
    for isl in (32, 64):
        for batch in (4,):
            entry = {"isl": isl, "batch": batch}
            for mode in ("old", "new"):
                rows = [
                    row
                    for trial in range(2)
                    for row in data[mode, trial]["rows"]
                    if row["isl"] == isl and row["concurrency"] == batch
                ]
                entry[mode] = {
                    "mean_wall_ms": mean(row["wall_s"] for row in rows) * 1000,
                    "pooled_tokens_per_s": len(rows)
                    * batch
                    * 64
                    / sum(row["wall_s"] for row in rows),
                    "accepted": sum(row["accepted"] for row in rows),
                    "proposals": sum(row["proposals"] for row in rows),
                }
            entry["mtp_latency_reduction_pct"] = (
                1 - entry["new"]["mean_wall_ms"] / entry["old"]["mean_wall_ms"]
            ) * 100
            summary.append(entry)
    checks = []
    for trial in range(2):
        for other in ("old",):
            left, right = data["new", trial]["rows"], data[other, trial]["rows"]
            checks.append(
                {
                    "trial": trial,
                    "comparison": f"new_vs_{other}",
                    "all_acceptance_counts_equal": all(
                        (a["accepted"], a["proposals"]) == (b["accepted"], b["proposals"])
                        for a, b in zip(left, right, strict=True)
                    ),
                    "all_output_ids_equal": all(
                        a["output_ids"] == b["output_ids"] for a, b in zip(left, right, strict=True)
                    ),
                    "different_rows": [
                        {"isl": a["isl"], "batch": a["concurrency"], "repeat": a["repeat"]}
                        for a, b in zip(left, right, strict=True)
                        if a["output_ids"] != b["output_ids"]
                    ],
                }
            )
    result = {"summary": summary, "checks": checks}
    (report / "focused_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
