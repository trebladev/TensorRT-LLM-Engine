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


@pytest.fixture(params=[False, True], autouse=True)
def draft_batching_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise merged generation and the explicit equal-length fallback."""
    monkeypatch.setenv("TRTLLM_QWEN35_MTP_DISABLE_DRAFT_BATCHING", "1" if request.param else "0")


@pytest.fixture(scope="module")
def native_engine(tmp_path_factory):
    cached = os.environ.get("QWEN35_MTP_ENGINE_DIR")
    if cached:
        return Path(cached)
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Qwen3.5-2B checkpoint is required")
    engine_dir = tmp_path_factory.mktemp("qwen35_native_mtp")
    build_engines(root / "Qwen3.5-2B", engine_dir, max_batch_size=4)
    gc.collect()
    torch.cuda.empty_cache()
    return engine_dir


def runner_for(engine_dir, native, max_batch_size=1):
    return ModelRunnerCpp.from_dir(
        str(engine_dir),
        max_batch_size=max_batch_size,
        cuda_graph_mode=False,
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        kv_cache_free_gpu_memory_fraction=0.1,
        gather_generation_logits=not native,
        mtp_draft_engine_path=str(engine_dir / "mtp.engine") if native else None,
    )


def generate(runner, prompt, count, end_id=-1, *, return_logits=False):
    result = runner.generate(
        [torch.tensor(prompt, dtype=torch.int32)],
        max_new_tokens=count,
        top_k=1,
        end_id=end_id,
        pad_id=0,
        return_dict=True,
        output_sequence_lengths=True,
        output_generation_logits=return_logits,
    )
    length = int(result["sequence_lengths"][0, 0])
    tokens = result["output_ids"][0, 0, len(prompt) : length].tolist()
    if return_logits:
        return tokens, result["generation_logits"][0, 0].float().cpu()
    return tokens


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_mtp_greedy_and_reuse(native_engine, batch_size):
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    baseline = runner_for(native_engine, False)
    try:
        reference = [generate(baseline, prompt, 24, return_logits=True) for prompt in prompts]
        expected = [tokens for tokens, _ in reference]
    finally:
        baseline.session.shutdown()
        del baseline
        gc.collect()
    native = runner_for(native_engine, True, batch_size)
    try:
        for count in (1, 2, 3, 8, 23, 24):
            for prompt, tokens in zip(prompts, expected):
                assert generate(native, prompt, count) == tokens[:count]
        for prompt, tokens in zip(prompts, expected):
            for eos in tokens[:4]:
                assert generate(native, prompt, 24, eos) == tokens[: tokens.index(eos) + 1]
        assert generate(native, prompts[0], 24) == expected[0]
        # Enqueued requests must preserve independent histories as slots are reused.
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
        native.session.get_latest_iteration_stats()
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
        for i, request_id in enumerate(ids):
            # Changing the final mixed-batch GEMM shape can move a near-tied
            # logit by one BF16 rounding bin. All preceding tokens must match.
            tokens = actual[request_id]
            assert len(tokens) == len(expected[i])
            assert tokens[:-1] == expected[i][:-1]
            if tokens[-1] != expected[i][-1]:
                logits = reference[i][1][-1]
                gap = logits.max() - logits[tokens[-1]]
                assert gap <= torch.finfo(torch.bfloat16).eps * logits.max().abs()
        assert saw_two_tokens, "Expected an accepted candidate plus a bonus token"
        stats = [
            item.inflight_batching_stats for item in native.session.get_latest_iteration_stats()
        ]
        if batch_size > 1:
            assert any(item is not None and item.num_gen_requests > 1 for item in stats)

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


def test_native_mtp_mixed_budgets_and_arrivals(native_engine):
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    baseline = runner_for(native_engine, False)
    try:
        expected = [generate(baseline, prompt, 32) for prompt in prompts]
    finally:
        baseline.session.shutdown()
        del baseline
        gc.collect()
    native = runner_for(native_engine, True, 4)
    try:
        ropes = native._prepare_mrope_executor(prompts, None)
        native.session.get_latest_iteration_stats()
        saw_mixed = False
        for turn in range(3):
            # Keep the synthetic 64-token prompt before its near-tied continuation
            # when cancellation changes batch shape; the rounding case is covered above.
            budgets = (1, 2, 7, 32) if turn == 0 else (24, 25, 20, 32)
            eos_ids = [-1, -1, expected[2][3] if turn == 2 else -1, -1]

            def request(i):
                return executor.Request(
                    input_token_ids=prompts[i],
                    max_tokens=budgets[i],
                    end_id=eos_ids[i],
                    pad_id=0,
                    streaming=True,
                    sampling_config=executor.SamplingConfig(top_k=1),
                    output_config=executor.OutputConfig(exclude_input_from_output=True),
                    mrope_config=ropes[i],
                )

            ids = native.session.enqueue_requests([request(i) for i in range(3)])
            actual = {request_id: [] for request_id in ids}
            finished = set()
            added = False
            while len(finished) < len(ids) or not added:
                responses = native.session.await_responses(timeout=30.0)
                assert responses, "Timed out waiting for mixed native MTP requests"
                for response in responses:
                    assert not response.has_error(), response.error_msg
                    actual[response.request_id].extend(response.result.output_token_ids[0])
                    if response.result.is_final:
                        finished.add(response.request_id)
                if not added:
                    # Admit a fresh context while earlier requests are generating.
                    request_id = native.session.enqueue_request(request(3))
                    ids.append(request_id)
                    actual[request_id] = []
                    added = True
                    if turn == 1 and ids[0] not in finished:
                        native.session.cancel_request(ids[0])
            for i, request_id in enumerate(ids):
                tokens = expected[i][: budgets[i]]
                if eos_ids[i] in tokens:
                    tokens = tokens[: tokens.index(eos_ids[i]) + 1]
                if turn == 1 and i == 0:
                    assert len(actual[request_id]) < len(tokens), (
                        "Cancellation must stop an active request early"
                    )
                    assert actual[request_id] == tokens[: len(actual[request_id])]
                else:
                    assert actual[request_id] == tokens
            stats = [
                item.inflight_batching_stats for item in native.session.get_latest_iteration_stats()
            ]
            saw_mixed |= any(
                item is not None and item.num_gen_requests > 0 and item.num_context_requests > 0
                for item in stats
            )
        assert saw_mixed, "Expected overlapping context and generation requests"
    finally:
        native.session.shutdown()
