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

"""Measure fixed-length native K=1 MTP against ordinary TensorRT decoding.

Run each mode in a fresh process with CUDA_VISIBLE_DEVICES selecting one GPU.
Engine loading, tokenization and request preparation are excluded from timings.
NVML reports device memory including reserved pools, not intrinsic MTP overhead.
The hybrid multi-window KV allocator may ignore the requested max_tokens cap.
"""

import argparse
import datetime
import hashlib
import json
import platform
import subprocess
import threading
import time
from pathlib import Path

import pynvml
import torch
from transformers import AutoTokenizer

import tensorrt_llm
import tensorrt_llm.bindings.executor as executor
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM
from tensorrt_llm.runtime import ModelRunnerCpp

_PROMPTS = [
    "Explain how a computer stores information using binary digits. "
    "Give a simple example and describe the steps clearly. ",
    "Write a short story about a traveler who discovers an old library in a quiet mountain village. ",
    "Describe how plants use sunlight and water to grow, and explain why this process matters for life on Earth. ",
    "Write a Python function that sorts a list of numbers and explain how the algorithm works with an example. ",
    "介绍一下中国传统节日的习俗，并说明这些习俗如何体现家庭团聚和文化传承。",
    "Explain the difference between a database index and a table scan, with a practical example of each. ",
    "Plan a three day visit to Paris including museums, parks, food, and ways to travel around the city. ",
    "Describe the water cycle from evaporation to rainfall and explain the role played by oceans and clouds. ",
]


def _build_plain(model_dir: Path, engine_dir: Path) -> None:
    model = Qwen35ForCausalLM.from_hugging_face(model_dir, dtype="bfloat16")
    config = BuildConfig(
        max_batch_size=4, max_input_len=126, max_seq_len=129, max_num_tokens=512, opt_num_tokens=256
    )
    config.plugin_config.gpt_attention_plugin = "bfloat16"
    config.plugin_config.gemm_plugin = "bfloat16"
    config.plugin_config.mamba_conv1d_plugin = "bfloat16"
    engine_dir.mkdir(parents=True, exist_ok=True)
    build(model, config).save(engine_dir)


def _gpu_snapshot(handle) -> dict:
    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return {
        # NVML PIDs may belong to the host namespace, so preserve them directly.
        "compute_processes": [
            {"nvml_pid": p.pid, "used_mib": p.usedGpuMemory / 2**20} for p in processes
        ],
        "device_mib": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20,
        "sm_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
        "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
    }


def _measure(runner: ModelRunnerCpp, prompts: list[list[int]], osl: int, mtp: bool) -> dict:
    ropes = runner._prepare_mrope_executor(prompts, None)
    requests = [
        executor.Request(
            input_token_ids=prompt,
            max_tokens=osl,
            end_id=-1,
            pad_id=0,
            streaming=True,
            sampling_config=executor.SamplingConfig(top_k=1),
            output_config=executor.OutputConfig(exclude_input_from_output=True),
            mrope_config=rope,
        )
        for prompt, rope in zip(prompts, ropes)
    ]
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    start = start_ns / 1e9
    ids = runner.session.enqueue_requests(requests)
    tokens = {rid: [] for rid in ids}
    events = {rid: [] for rid in ids}
    finished = set()
    accepted = proposals = 0
    while len(finished) < len(ids):
        responses = runner.session.await_responses(timeout=datetime.timedelta(seconds=30))
        now = time.perf_counter() - start
        if not responses:
            raise TimeoutError("No executor response in 30 seconds")
        for response in responses:
            if response.has_error():
                raise RuntimeError(response.error_msg)
            rid = response.request_id
            result = response.result
            new = result.output_token_ids[0]
            previous = len(tokens[rid])
            if new:
                if len(new) not in ((1, 2) if mtp else (1,)):
                    raise RuntimeError(f"Unexpected streaming chunk length: {len(new)}")
                if previous == 0 and len(new) != 1:
                    raise RuntimeError("Expected one prefill token")
                if mtp and previous > 0 and osl - previous > 1:
                    proposals += 1
                    accepted += len(new) - 1
                events[rid].append([now, len(new)])
                tokens[rid].extend(new)
            if result.is_final:
                finished.add(rid)
    elapsed = time.perf_counter() - start
    if any(len(t) != osl for t in tokens.values()):
        raise RuntimeError("Output length does not match fixed token budget")
    return {
        "start_ns": start_ns,
        "wall_s": elapsed,
        "tokens_per_s": len(ids) * osl / elapsed,
        "ttft_ms": [events[rid][0][0] * 1000 for rid in ids],
        "tpot_ms": [(events[rid][-1][0] - events[rid][0][0]) * 1000 / (osl - 1) for rid in ids],
        "latency_ms": [events[rid][-1][0] * 1000 for rid in ids],
        "accepted": accepted,
        "proposals": proposals,
        "output_ids": [tokens[rid] for rid in ids],
        "events": [events[rid] for rid in ids],
        "input_ids": prompts,
    }


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--engine_dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["plain", "target_only", "mtp"], default="plain")
    parser.add_argument("--build_plain", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--isl", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--osl", type=int, default=64)
    args = parser.parse_args()
    if args.build_plain:
        _build_plain(args.model_dir, args.engine_dir)
        return
    if args.output is None:
        parser.error("--output is required for measurement")
    if args.repeats < 1 or args.warmup < 1 or args.osl < 2:
        parser.error("repeats/warmup must be positive and osl must be at least 2")
    if any(c not in (1, 2, 3, 4) for c in args.concurrency):
        parser.error("concurrency must be between 1 and 4")
    if any(n < 1 or n + args.osl > 128 for n in args.isl):
        parser.error("ISL must be positive and ISL + OSL must not exceed 128")
    engine_config = json.loads((args.engine_dir / "config.json").read_text())
    speculative = engine_config["build_config"]["max_draft_len"] > 0
    if speculative != (args.mode != "plain"):
        parser.error(
            "plain requires a non-speculative engine; target_only/mtp require a speculative target"
        )
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByUUID(str(torch.cuda.get_device_properties(0).uuid))
    runner = ModelRunnerCpp.from_dir(
        str(args.engine_dir),
        max_batch_size=4,
        cuda_graph_mode=False,
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        max_tokens_in_paged_kv_cache=1024,
        mtp_draft_engine_path=str(args.engine_dir / "mtp.engine") if args.mode == "mtp" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    corpus = [tokenizer.encode(p * 20, add_special_tokens=False) for p in _PROMPTS]
    samples = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            samples.append(_gpu_snapshot(handle))
            stop.wait(0.02)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    rows = []
    try:
        for isl in args.isl:
            for concurrency in args.concurrency:
                for repeat in range(-args.warmup, args.repeats):
                    prompts = [
                        corpus[(max(repeat, 0) + i) % len(corpus)][:isl] for i in range(concurrency)
                    ]
                    row = _measure(runner, prompts, args.osl, args.mode == "mtp")
                    if repeat >= 0:
                        row.update(
                            isl=isl,
                            concurrency=concurrency,
                            repeat=repeat,
                            gpu=_gpu_snapshot(handle),
                        )
                        rows.append(row)
                        print(
                            json.dumps(
                                {
                                    k: row[k]
                                    for k in (
                                        "isl",
                                        "concurrency",
                                        "repeat",
                                        "tokens_per_s",
                                        "accepted",
                                        "proposals",
                                    )
                                }
                            ),
                            flush=True,
                        )
    finally:
        stop.set()
        monitor.join()
        runner.session.shutdown()
    metadata = {
        "mode": args.mode,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tensorrt_llm": tensorrt_llm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "engine_config": engine_config,
        "engine_sha256": _file_sha256(args.engine_dir / "rank0.engine"),
        "draft_sha256": _file_sha256(args.engine_dir / "mtp.engine")
        if args.mode == "mtp"
        else None,
        "sampled_peak_device_mib": max(s["device_mib"] for s in samples),
        "gpu_samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2) + "\n")
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
