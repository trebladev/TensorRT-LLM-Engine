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
import json
import os
from pathlib import Path

import numpy as np
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


def _native_engine(tmp_path_factory, draft_length):
    cached = os.environ.get("QWEN35_MTP_ENGINE_DIR")
    if cached:
        config = json.loads((Path(cached) / "config.json").read_text())
        if config["build_config"]["max_draft_len"] != draft_length:
            pytest.skip("Cached engine has a different draft length")
        return Path(cached)
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Qwen3.5-2B checkpoint is required")
    engine_dir = tmp_path_factory.mktemp("qwen35_native_mtp")
    build_engines(root / "Qwen3.5-2B", engine_dir, max_batch_size=4, max_draft_len=draft_length)
    gc.collect()
    torch.cuda.empty_cache()
    return engine_dir


@pytest.fixture(scope="module")
def native_engine(tmp_path_factory):
    return _native_engine(tmp_path_factory, 1)


@pytest.fixture(scope="module", params=[1, 2, 3])
def multi_native_engine(tmp_path_factory, request):
    return _native_engine(tmp_path_factory, request.param)


def runner_for(engine_dir, native, max_batch_size=1, *, debug_mode=False):
    return ModelRunnerCpp.from_dir(
        str(engine_dir),
        max_batch_size=max_batch_size,
        cuda_graph_mode=False,
        debug_mode=debug_mode,
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        kv_cache_free_gpu_memory_fraction=0.1,
        gather_generation_logits=not native,
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
    tokens = result["output_ids"][0, 0, len(prompt) : length].tolist()
    return tokens


def _baseline_with_logits(engine, prompts, count, monkeypatch, tmp_path):
    """Copy per-forward logits; returned generation-logit buffers may be reused."""
    original_debug_config = executor.DebugConfig
    monkeypatch.setattr(
        executor,
        "DebugConfig",
        lambda **kwargs: original_debug_config(
            debug_output_tensors=True,
            debug_tensor_names=["logits"],
            debug_tensors_max_iterations=0,
        ),
    )
    baseline = runner_for(engine, False, debug_mode=True)
    reference = []
    try:
        for index, prompt in enumerate(prompts):
            directory = tmp_path / f"reference_{index}"
            directory.mkdir()
            monkeypatch.setenv("TMPDIR", str(directory))
            tokens = generate(baseline, prompt, count)
            trace_root = directory / "tllm_debug" / "PP_1" / "TP_1"
            traces = sorted(
                trace_root.glob("iteration_*"), key=lambda path: int(path.name.split("_")[-1])
            )
            logits = [torch.from_numpy(np.load(path / "logits.npy")).float()[0] for path in traces]
            assert len(logits) == len(tokens)
            reference.append((tokens, logits))
    finally:
        baseline.session.shutdown()
    return reference


def _assert_baseline_prefix(actual, expected, logits):
    """Require equal histories until a BF16-scale tie changes the greedy choice.

    The per-forward test below additionally validates every emitted token after
    such a divergence against the target row for its own history.
    """
    assert len(actual) == len(expected)
    for index, (token, baseline_token) in enumerate(zip(actual, expected)):
        if token != baseline_token:
            scores = logits[index]
            assert (
                scores.max() - scores[token] <= torch.finfo(torch.bfloat16).eps * scores.max().abs()
            )
            break


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_mtp_greedy_and_reuse(native_engine, batch_size, monkeypatch, tmp_path):
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    reference = _baseline_with_logits(native_engine, prompts, 24, monkeypatch, tmp_path)
    native = runner_for(native_engine, True, batch_size)
    try:
        for count in (1, 2, 3, 8, 23, 24):
            for prompt, (tokens, logits) in zip(prompts, reference):
                _assert_baseline_prefix(generate(native, prompt, count), tokens[:count], logits)
        for prompt, (tokens, logits) in zip(prompts, reference):
            for eos in tokens[:4]:
                _assert_baseline_prefix(
                    generate(native, prompt, 24, eos),
                    tokens[: tokens.index(eos) + 1],
                    logits,
                )
        _assert_baseline_prefix(generate(native, prompts[0], 24), *reference[0])
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
                saw_two_tokens |= len(emitted) >= 2
                actual[response.request_id].extend(emitted)
                if response.result.is_final:
                    finished.add(response.request_id)
        for i, request_id in enumerate(ids):
            _assert_baseline_prefix(actual[request_id], *reference[i])
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
        _assert_baseline_prefix(generate(native, prompts[1], 24), *reference[1])
    finally:
        native.session.shutdown()


def test_native_mtp_mixed_budgets_and_arrivals(native_engine, monkeypatch, tmp_path):
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    reference = _baseline_with_logits(native_engine, prompts, 32, monkeypatch, tmp_path)
    expected = [tokens for tokens, _ in reference]
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
                    _assert_baseline_prefix(
                        actual[request_id],
                        tokens[: len(actual[request_id])],
                        reference[i][1],
                    )
                else:
                    _assert_baseline_prefix(actual[request_id], tokens, reference[i][1])
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


@pytest.mark.parametrize("batch_size", [1, 4])
def test_native_mtp_multiple_drafts(multi_native_engine, batch_size, monkeypatch, tmp_path):
    """Validate every emitted token against its actual target verification row.

    Wider BF16 forwards can create ties absent in single-token decoding. Check
    the first divergence against independent baseline logits, then check every
    native emission against the target logits for its own accepted history.
    Debug copies avoid the output-buffer aliasing of delayed logits fragments.
    """
    original_debug_config = executor.DebugConfig
    monkeypatch.setattr(
        executor,
        "DebugConfig",
        lambda **kwargs: original_debug_config(
            debug_input_tensors=True,
            debug_output_tensors=True,
            debug_tensor_names=[
                "input_ids",
                "logits",
                "host_request_types",
                "host_past_key_value_lengths",
                "gated_delta_cu_seqlens",
            ],
            debug_tensors_max_iterations=0,
        ),
    )

    def trace_directory(name):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setenv("TMPDIR", str(directory))
        return directory / "tllm_debug" / "PP_1" / "TP_1"

    def read_traces(directory):
        return [
            {path.stem: torch.from_numpy(np.load(path)) for path in entry.glob("*.npy")}
            for entry in sorted(
                directory.glob("iteration_*"), key=lambda path: int(path.name.split("_")[-1])
            )
        ]

    config = json.loads((multi_native_engine / "config.json").read_text())
    draft_length = config["build_config"]["max_draft_len"]
    prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
    baseline = runner_for(multi_native_engine, False, debug_mode=True)
    reference = []
    try:
        for index, prompt in enumerate(prompts):
            directory = trace_directory(f"baseline_{index}")
            tokens = generate(baseline, prompt, 32)
            logits = [item["logits"].float()[0] for item in read_traces(directory)]
            assert len(logits) == len(tokens)
            reference.append((tokens, logits))
    finally:
        baseline.session.shutdown()
        del baseline
        gc.collect()

    native = runner_for(multi_native_engine, True, batch_size, debug_mode=True)
    saw_full_width = saw_rejection = False
    try:
        ropes = native._prepare_mrope_executor(prompts, None)
        for turn, budgets in enumerate(((1, 2, draft_length + 2, 32), (24, 25, 20, 32))):
            eos_ids = [-1, -1, reference[2][0][3] if turn else -1, -1]
            requests = [
                executor.Request(
                    input_token_ids=prompt,
                    max_tokens=budget,
                    end_id=eos,
                    pad_id=0,
                    streaming=True,
                    sampling_config=executor.SamplingConfig(top_k=1),
                    output_config=executor.OutputConfig(exclude_input_from_output=True),
                    mrope_config=rope,
                )
                for prompt, budget, eos, rope in zip(prompts, budgets, eos_ids, ropes)
            ]
            directory = trace_directory(f"native_{turn}")
            ids = native.session.enqueue_requests(requests[:3])
            actual = {request_id: [] for request_id in ids}
            finished = set()
            added = False
            while len(finished) < len(ids) or not added:
                responses = native.session.await_responses(timeout=30.0)
                assert responses, "Timed out waiting for multi-token native MTP"
                for response in responses:
                    assert not response.has_error(), response.error_msg
                    actual[response.request_id].extend(response.result.output_token_ids[0])
                    if response.result.is_final:
                        finished.add(response.request_id)
                if not added:
                    request_id = native.session.enqueue_request(requests[3])
                    ids.append(request_id)
                    actual[request_id] = []
                    added = True

            traces = {request_id: [] for request_id in ids}
            for tensors in read_traces(directory):
                request_ids = tensors["request_ids"].flatten().tolist()
                ends = tensors["gated_delta_cu_seqlens"].tolist()
                types = tensors["host_request_types"].tolist()
                past = tensors["host_past_key_value_lengths"].tolist()
                logits_offset = 0
                for row, request_id in enumerate(request_ids):
                    context = types[row] == 0
                    width = ends[row + 1] - ends[row]
                    rows = 1 if context else width
                    prompt_length = len(prompts[ids.index(request_id)])
                    start = 0 if context else past[row] + 1 - prompt_length
                    traces[request_id].append(
                        (
                            start,
                            tensors["logits"][logits_offset : logits_offset + rows].float().cpu(),
                            tensors["input_ids"][ends[row] : ends[row + 1]].cpu(),
                            context,
                        )
                    )
                    logits_offset += rows
                    saw_full_width |= not context and width == draft_length + 1

            for i, request_id in enumerate(ids):
                tokens = actual[request_id]
                if eos_ids[i] in tokens:
                    assert tokens[-1] == eos_ids[i]
                else:
                    assert len(tokens) == budgets[i]
                expected, baseline_logits = reference[i]
                for index, (token, expected_token) in enumerate(zip(tokens, expected)):
                    if token != expected_token:
                        scores = baseline_logits[index]
                        assert (
                            scores.max() - scores[token]
                            <= torch.finfo(torch.bfloat16).eps * scores.max().abs()
                        )
                        break
                entries = traces[request_id]
                for index, (start, scores, inputs, context) in enumerate(entries):
                    end = entries[index + 1][0] if index + 1 < len(entries) else len(tokens)
                    count = end - start
                    assert 0 < count <= scores.shape[0]
                    emitted = torch.tensor(tokens[start:end], dtype=torch.long)
                    selected = scores[torch.arange(count), emitted]
                    torch.testing.assert_close(selected, scores[:count].amax(-1), atol=0, rtol=0)
                    if not context:
                        assert inputs[0].item() == tokens[start - 1]
                        assert inputs[1:count].tolist() == tokens[start : end - 1]
                        saw_rejection |= count < scores.shape[0]
        assert saw_full_width, "Expected K+1 target verification rows"
        assert saw_rejection, "Expected a rejected draft suffix"
    finally:
        native.session.shutdown()
