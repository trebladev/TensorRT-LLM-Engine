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

"""GPU coverage for Qwen3.5 multi-token verification through the C++ executor."""

import gc
import json

import pytest
import torch
from utils.llm_data import llm_models_root

import tensorrt_llm.bindings.executor as executor
from tensorrt_llm.builder import BuildConfig, build
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode
from tensorrt_llm.models.qwen35.model import Qwen35ForCausalLM
from tensorrt_llm.runtime.model_runner_cpp import ModelRunnerCpp


@pytest.fixture(scope="module", params=[1, 2, 3])
def verification_engine(tmp_path_factory, request):
    root = llm_models_root()
    if root is None:
        pytest.skip("LLM_MODELS_ROOT is required")
    model_dir = next(
        (root / name for name in ("Qwen3.5-2B", "Qwen3.5/Qwen3.5-2B") if (root / name).is_dir()),
        None,
    )
    if model_dir is None:
        pytest.skip("Qwen3.5-2B checkpoint is required")
    model = Qwen35ForCausalLM.from_hugging_face(model_dir, dtype="bfloat16")
    config = BuildConfig(
        max_batch_size=4,
        max_input_len=96,
        max_seq_len=127,
        max_num_tokens=384,
        opt_num_tokens=128,
        max_draft_len=request.param,
        speculative_decoding_mode=SpeculativeDecodingMode.DRAFT_TOKENS_EXTERNAL,
    )
    config.plugin_config.gpt_attention_plugin = "bfloat16"
    config.plugin_config.gemm_plugin = "bfloat16"
    config.plugin_config.mamba_conv1d_plugin = "bfloat16"
    engine_dir = tmp_path_factory.mktemp("qwen35_executor")
    build(model, config).save(engine_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return engine_dir


def _runner(engine_dir, **overrides):
    options = dict(
        rank=0,
        max_batch_size=4,
        max_input_len=96,
        max_output_len=32,
        max_attention_window_size=[
            json.loads((engine_dir / "config.json").read_text())["build_config"]["max_seq_len"]
        ],
        kv_cache_enable_block_reuse=False,
        enable_chunked_context=False,
        use_runtime_defaults=False,
        kv_cache_free_gpu_memory_fraction=0.15,
    )
    options.update(overrides)
    return ModelRunnerCpp.from_dir(str(engine_dir), **options)


def _generate(runner, prompts, candidates=None, *, count=4, end_id=-1, top_k=1):
    options = {} if candidates is None else {"draft_tokens_list": candidates}
    result = runner.generate(
        [torch.tensor(prompt, dtype=torch.int32) for prompt in prompts],
        max_new_tokens=count,
        end_id=end_id,
        pad_id=0,
        top_k=top_k,
        return_dict=True,
        output_sequence_lengths=True,
        **options,
    )
    torch.cuda.synchronize()
    return [
        result["output_ids"][i, 0, len(prompt) : int(result["sequence_lengths"][i, 0])].tolist()
        for i, prompt in enumerate(prompts)
    ]


def test_qwen35_executor_external_draft(verification_engine):
    runner = _runner(verification_engine)
    try:
        # Exercise both sides of the 64-token block boundary and physical slot reuse.
        prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
        baseline = _generate(runner, prompts)
        for turn in range(3):
            accept = [(i + turn) % 2 == 0 for i in range(len(prompts))]
            candidates = [
                [tokens[0] if keep else (tokens[0] + 1) % 8192]
                for tokens, keep in zip(baseline, accept)
            ]
            actual = _generate(runner, prompts, candidates)
            assert actual == [
                tokens[:2] if keep else tokens[:1] for tokens, keep in zip(baseline, accept)
            ]
            assert _generate(runner, prompts) == baseline

        # The runner's draft list requires a candidate for every request. Use the
        # executor API directly to mix requests with and without a draft config.
        mixed = [[baseline[0][0]], None, [(baseline[2][0] + 1) % 8192], None]
        mrope = runner._prepare_mrope_executor(prompts, None)
        requests = [
            executor.Request(
                input_token_ids=prompt,
                max_tokens=4,
                end_id=-1,
                pad_id=0,
                sampling_config=executor.SamplingConfig(top_k=1),
                output_config=executor.OutputConfig(exclude_input_from_output=True),
                mrope_config=rope,
                external_draft_tokens_config=(
                    executor.ExternalDraftTokensConfig(draft) if draft is not None else None
                ),
            )
            for prompt, draft, rope in zip(prompts, mixed, mrope)
        ]
        request_ids = runner.session.enqueue_requests(requests)
        actual = {}
        while len(actual) < len(request_ids):
            responses = runner.session.await_responses(timeout=30.0)
            assert responses, "Timed out waiting for mixed verification requests"
            for response in responses:
                assert not response.has_error(), response.error_msg
                if response.result.is_final:
                    actual[response.request_id] = response.result.output_token_ids[0]
        assert [actual[request_id] for request_id in request_ids] == [
            baseline[0][:2],
            baseline[1],
            baseline[2][:1],
            baseline[3],
        ]

        prompt = prompts[:1]
        token = baseline[0][0]
        # An accepted candidate can itself be EOS or exhaust the output budget.
        assert _generate(runner, prompt, [[token]], count=1) == [[token]]
        # The existing external-draft decoder excludes an accepted EOS from output.
        assert _generate(runner, prompt, [[token]], end_id=token) == [[]]
        assert _generate(runner, prompts) == baseline
        with pytest.raises(RuntimeError, match="greedy topK=1"):
            _generate(runner, prompt, [[token]], top_k=2)
    finally:
        runner.session.shutdown()


def test_qwen35_executor_partial_acceptance(verification_engine):
    draft_length = json.loads((verification_engine / "config.json").read_text())["build_config"][
        "max_draft_len"
    ]
    runner = _runner(verification_engine)
    try:
        prompts = [list(range(1, length + 1)) for length in (17, 63, 64, 65)]
        baseline = _generate(runner, prompts, count=draft_length + 2)
        for turn in range(draft_length + 1):
            accepted = [(turn + i) % (draft_length + 1) for i in range(len(prompts))]
            candidates = [tokens[:draft_length].copy() for tokens in baseline]
            for candidate, count in zip(candidates, accepted):
                if count < draft_length:
                    candidate[count] = (candidate[count] + 1) % 8192
            actual = _generate(runner, prompts, candidates, count=draft_length + 2)
            assert actual == [tokens[: count + 1] for tokens, count in zip(baseline, accepted)]
            assert _generate(runner, prompts, count=draft_length + 2) == baseline
    finally:
        runner.session.shutdown()


@pytest.mark.parametrize("option", ["kv_cache_enable_block_reuse", "enable_chunked_context"])
def test_qwen35_executor_verification_configuration(verification_engine, option):
    with pytest.raises(RuntimeError, match="prefix reuse and chunked context disabled"):
        _runner(verification_engine, **{option: True})
