# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end persistent native MTP through the C++ executor."""

import gc
import os
from pathlib import Path

import pytest
import torch
from utils.llm_data import llm_models_root

import tensorrt_llm.bindings.executor as executor
from examples.models.core.qwen3_5.mtp_executor_demo import build_engines
from tensorrt_llm.runtime import ModelRunnerCpp


@pytest.fixture(scope="module")
def native_engine(tmp_path_factory):
    cached = os.environ.get("QWEN35_MTP_ENGINE_DIR")
    if cached:
        return Path(cached)
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Qwen3.5-2B checkpoint is required")
    engine_dir = tmp_path_factory.mktemp("qwen35_native_mtp")
    build_engines(root / "Qwen3.5-2B", engine_dir)
    gc.collect()
    torch.cuda.empty_cache()
    return engine_dir


def runner_for(engine_dir, native):
    return ModelRunnerCpp.from_dir(
        str(engine_dir),
        max_batch_size=1,
        cuda_graph_mode=False,
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        kv_cache_free_gpu_memory_fraction=0.1,
        mtp_draft_engine_path=str(engine_dir / "mtp.engine") if native else None,
    )


def generate(runner, prompt, count, end_id=-1):
    result = runner.generate(
        [torch.tensor(prompt, dtype=torch.int32)],
        max_new_tokens=count,
        top_k=1,
        end_id=end_id,
        pad_id=0,
        return_dict=True,
        output_sequence_lengths=True,
    )
    length = int(result["sequence_lengths"][0, 0])
    return result["output_ids"][0, 0, len(prompt) : length].tolist()


def test_native_mtp_greedy_and_reuse(native_engine):
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    baseline = runner_for(native_engine, False)
    try:
        expected = [generate(baseline, prompt, 24) for prompt in prompts]
    finally:
        baseline.session.shutdown()
        del baseline
        gc.collect()
    native = runner_for(native_engine, True)
    try:
        for count in (1, 2, 3, 8, 23, 24):
            for prompt, tokens in zip(prompts, expected):
                assert generate(native, prompt, count) == tokens[:count]
        for prompt, tokens in zip(prompts, expected):
            for eos in tokens[:4]:
                assert generate(native, prompt, 24, eos) == tokens[: tokens.index(eos) + 1]
        assert generate(native, prompts[0], 24) == expected[0]
        # Enqueued requests must serialize safely through the one-active-request worker.
        ropes = native._prepare_mrope_executor(prompts, None)
        requests = [
            executor.Request(
                input_token_ids=prompt,
                max_tokens=24,
                end_id=-1,
                pad_id=0,
                streaming=True,
                sampling_config=executor.SamplingConfig(top_k=1),
                output_config=executor.OutputConfig(exclude_input_from_output=True),
                mrope_config=rope,
            )
            for prompt, rope in zip(prompts, ropes)
        ]
        ids = native.session.enqueue_requests(requests)
        actual = {request_id: [] for request_id in ids}
        finished = set()
        saw_two_tokens = False
        while len(finished) < len(ids):
            responses = native.session.await_responses(timeout=30.0)
            assert responses, "Timed out waiting for queued native MTP requests"
            for response in responses:
                assert not response.has_error(), response.error_msg
                emitted = response.result.output_token_ids[0]
                saw_two_tokens |= len(emitted) == 2
                actual[response.request_id].extend(emitted)
                if response.result.is_final:
                    finished.add(response.request_id)
        assert [actual[request_id] for request_id in ids] == expected
        assert saw_two_tokens, "Expected an accepted candidate plus a bonus token"

        request_id = native.session.enqueue_request(requests[0])
        responses = native.session.await_responses(timeout=30.0)
        assert responses and all(not response.has_error() for response in responses)
        native.session.cancel_request(request_id)
        # Drain the direct executor request before returning to ModelRunnerCpp,
        # whose synchronous response collector owns all pending request IDs.
        while not any(response.result.is_final for response in responses):
            responses = native.session.await_responses(timeout=30.0)
            assert responses, "Timed out waiting for cancellation"
            assert all(not response.has_error() for response in responses)
        assert generate(native, prompts[1], 24) == expected[1]
    finally:
        native.session.shutdown()
