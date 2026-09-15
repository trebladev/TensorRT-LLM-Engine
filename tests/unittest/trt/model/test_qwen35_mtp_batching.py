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

"""Fixed-width MTP engine padding must not alter valid logits or KV history."""

import os
from pathlib import Path

import pytest
import tensorrt as trt
import torch
from utils.llm_data import llm_models_root

from examples.models.core.qwen3_5.mtp_demo import build_draft_engine
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.mtp import Qwen35MTP
from tensorrt_llm.runtime import Session
from tensorrt_llm.runtime.session import TensorInfo


class _DraftRunner:
    """Build explicit fixed-width inputs and preserve true per-request lengths."""

    def __init__(self, path: Path, config: Qwen35Config) -> None:
        self.session = Session.from_serialized_engine(path.read_bytes())
        self.config = config
        self.names = {
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
            if self.session.engine.get_tensor_mode(self.session.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        }
        self.capacity = self.session.engine.get_tensor_profile_shape("past_key_value_0", 0)[2][3]
        rope_shape = self.session.engine.get_tensor_profile_shape("mrope_rotary_cos_sin", 0)[2]
        self.rope_size = rope_shape[1]

    def prepare(
        self,
        tokens: list[torch.Tensor],
        hidden: list[torch.Tensor],
        caches: list[torch.Tensor],
        lengths: list[int],
        padded: bool,
        poison: bool = False,
    ) -> tuple[dict, dict]:
        batch = len(tokens)
        counts = [t.numel() for t in tokens]
        width = max(counts)
        if not padded and len(set(counts)) != 1:
            raise ValueError("Grouped input requires equal lengths")
        physical_tokens = torch.zeros((batch, width), dtype=torch.int32, device="cuda")
        physical_hidden = torch.zeros(
            (batch, width, self.config.hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        if poison:
            physical_tokens.fill_(1234)
            physical_hidden.fill_(3)
        for i, count in enumerate(counts):
            physical_tokens[i, :count] = tokens[i]
            physical_hidden[i, :count] = hidden[i]

        def cpu(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32)

        def gpu(values: list) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device="cuda")

        inputs = {
            "input_ids": physical_tokens.flatten(),
            "target_hidden_states": physical_hidden.reshape(-1, self.config.hidden_size),
            "past_key_value_0": torch.cat(caches, dim=0),
            "position_ids": gpu(lengths),
            "last_token_ids": gpu([i * width + n for i, n in enumerate(counts)]),
            "context_lengths": gpu([1] * batch),
            "host_context_lengths": cpu([1] * batch),
            # The engine computes the whole causal window; commit only true counts.
            "sequence_length": gpu([n + width for n in lengths]),
            "host_past_key_value_lengths": cpu(lengths),
            "host_request_types": cpu([1] * batch),
            "host_max_attention_window_sizes": cpu([self.capacity]),
            "host_sink_token_length": cpu([0]),
            "host_runtime_perf_knobs": torch.full((16,), -1, dtype=torch.int64),
            "host_context_progress": torch.zeros(1, dtype=torch.int64),
            "cache_indirection": torch.zeros(
                (batch, 1, self.capacity), dtype=torch.int32, device="cuda"
            ),
            "mrope_rotary_cos_sin": torch.zeros(
                (batch, self.rope_size), dtype=torch.float32, device="cuda"
            ),
            "mrope_position_deltas": gpu([[0]] * batch),
            "spec_decoding_use": cpu([int(width > 1)]),
            "spec_decoding_generation_lengths": gpu([width] * batch),
            "spec_decoding_position_offsets": gpu([list(range(width))] * batch),
            "spec_decoding_packed_mask": gpu(
                [[(1 << (j + 1)) - 1] for _ in range(batch) for j in range(width)]
            ),
        }
        inputs = {name: inputs[name] for name in self.names}
        infos = self.session.infer_shapes(
            [TensorInfo(name, torch_dtype_to_trt(t.dtype), t.shape) for name, t in inputs.items()]
        )
        outputs = {
            info.name: torch.empty(
                tuple(info.shape), dtype=trt_dtype_to_torch(info.dtype), device="cuda"
            )
            for info in infos
        }
        return inputs, outputs

    def run(self, prepared: tuple[dict, dict]) -> None:
        inputs, outputs = prepared
        self.session.set_shapes(inputs)
        if not self.session.run(inputs, outputs, torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("Draft enqueue failed")

    def selected_logits(self, prepared: tuple[dict, dict]) -> torch.Tensor:
        inputs, outputs = prepared
        if "last_token_logits" in outputs:
            assert outputs["last_token_logits"].shape[0] == inputs["last_token_ids"].numel()
            return outputs["last_token_logits"]
        return outputs["logits"][inputs["last_token_ids"].long() - 1]


@pytest.fixture(scope="module")
def draft_runner(tmp_path_factory: pytest.TempPathFactory) -> _DraftRunner:
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Set LLM_MODELS_ROOT to a directory containing Qwen3.5-2B")
    config = Qwen35Config.from_hugging_face(root / "Qwen3.5-2B")
    cached = os.environ.get("QWEN35_MTP_ENGINE_DIR")
    if cached:
        path = Path(cached) / "mtp.engine"
    else:
        path = tmp_path_factory.mktemp("mtp_padding") / "mtp.engine"
        model = Qwen35MTP.from_hugging_face(root / "Qwen3.5-2B")
        path.write_bytes(bytes(build_draft_engine(model, 129, 4, last_token_logits=True)))
    return _DraftRunner(path, config)


@pytest.mark.parametrize("seed", [42, 20260914, 20260915])
def test_padding_isolation_and_overwrite(draft_runner: _DraftRunner, seed: int) -> None:
    """Poison filler and unused KV; valid rows must remain bitwise identical.

    Hold the physical shape fixed to isolate masking from GEMM rounding.
    Alternate accepted/rejected widths, shrink the batch, cross cache-block
    boundaries, and finish at the engine's physical KV capacity.
    """
    torch.manual_seed(seed)
    runner = draft_runner
    config = runner.config
    shape = (1, 2, config.num_key_value_heads, runner.capacity, config.head_size)
    lengths = [runner.capacity - 48 + i for i in range(4)]
    caches = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.1 for _ in lengths]
    zero = [cache.clone() for cache in caches]
    poison = [cache.clone() for cache in caches]
    for i, length in enumerate(lengths):
        poison[i][..., length:, :].fill_(7)
    for step in range(28):
        if step in (8, 16):
            zero.pop()
            poison.pop()
            lengths.pop()
        counts = [1 + (step + i) % 2 for i in range(len(lengths))]
        if any(length + max(counts) > runner.capacity for length in lengths):
            break
        tokens = [
            torch.randint(config.vocab_size, (n,), device="cuda", dtype=torch.int32) for n in counts
        ]
        hidden = [
            torch.randn((n, config.hidden_size), device="cuda", dtype=torch.bfloat16)
            for n in counts
        ]
        outputs = []
        for filler, history in ((False, zero), (True, poison)):
            prepared = runner.prepare(tokens, hidden, history, lengths, True, poison=filler)
            runner.run(prepared)
            logits = runner.selected_logits(prepared).clone()
            kv = prepared[1]["present_key_value_0"].clone()
            assert torch.isfinite(logits).all()
            for i, length in enumerate(lengths):
                assert torch.equal(history[i][..., :length, :], kv[i : i + 1, :, :, :length])
            outputs.append((logits, kv))
        assert torch.equal(outputs[0][0], outputs[1][0])
        for i, (length, count) in enumerate(zip(lengths, counts)):
            assert torch.equal(
                outputs[0][1][i, :, :, : length + count], outputs[1][1][i, :, :, : length + count]
            )
        zero = [outputs[0][1][i : i + 1].clone() for i in range(len(lengths))]
        poison = [outputs[1][1][i : i + 1].clone() for i in range(len(lengths))]
        lengths = [length + count for length, count in zip(lengths, counts)]
    # End exactly at physical capacity with a one-token generation window.
    length = runner.capacity - 1
    token = torch.tensor([17], dtype=torch.int32, device="cuda")
    hidden = torch.zeros((1, config.hidden_size), dtype=torch.bfloat16, device="cuda")
    prepared = runner.prepare([token], [hidden], zero[:1], [length], False)
    runner.run(prepared)
    assert torch.isfinite(runner.selected_logits(prepared)).all()


@pytest.fixture(scope="module")
def all_rows_runner(tmp_path_factory: pytest.TempPathFactory) -> _DraftRunner:
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Set LLM_MODELS_ROOT to a directory containing Qwen3.5-2B")
    config = Qwen35Config.from_hugging_face(root / "Qwen3.5-2B")
    cached = os.environ.get("QWEN35_MTP_REFERENCE_ENGINE_DIR")
    if cached:
        path = Path(cached) / "mtp.engine"
    else:
        path = tmp_path_factory.mktemp("mtp_all_rows") / "mtp.engine"
        model = Qwen35MTP.from_hugging_face(root / "Qwen3.5-2B")
        path.write_bytes(bytes(build_draft_engine(model, 129, 4)))
    return _DraftRunner(path, config)


@pytest.mark.parametrize("counts", [(1,), (2,), (1, 2), (2, 1, 2, 1), (1, 2, 1, 2), (2, 2, 2, 2)])
@pytest.mark.parametrize("context", [False, True])
def test_last_token_projection_matches_packed(
    draft_runner: _DraftRunner,
    all_rows_runner: _DraftRunner,
    counts: tuple[int, ...],
    context: bool,
) -> None:
    """Compare selected logits and KV against an engine projecting all rows.

    Context uses ragged packed prefixes; generation uses mixed valid widths
    with trailing padding. The last row must be selected per request, not
    simply at the physical end of each padded slot.
    """
    torch.manual_seed(314159)
    batch = len(counts)
    config = draft_runner.config
    tokens = [
        torch.randint(config.vocab_size, (n,), dtype=torch.int32, device="cuda") for n in counts
    ]
    hidden = [
        torch.randn((n, config.hidden_size), dtype=torch.bfloat16, device="cuda") for n in counts
    ]
    lengths = [0 if context else 17 + i for i in range(batch)]
    caches = [
        torch.zeros(
            (1, 2, config.num_key_value_heads, draft_runner.capacity, config.head_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in counts
    ]
    results = []
    for runner in (all_rows_runner, draft_runner):
        inputs, _ = runner.prepare(tokens, hidden, caches, lengths, True)
        if context:
            inputs["input_ids"] = torch.cat(tokens)
            inputs["target_hidden_states"] = torch.cat(hidden)
            inputs["last_token_ids"] = torch.tensor(
                counts, dtype=torch.int32, device="cuda"
            ).cumsum(0, dtype=torch.int32)
            inputs["host_request_types"].zero_()
            inputs["host_context_lengths"] = torch.tensor(counts, dtype=torch.int32)
            inputs["context_lengths"] = inputs["host_context_lengths"].cuda()
            inputs["sequence_length"] = inputs["context_lengths"].clone()
            inputs["spec_decoding_use"].zero_()
        infos = runner.session.infer_shapes(
            [TensorInfo(name, torch_dtype_to_trt(t.dtype), t.shape) for name, t in inputs.items()]
        )
        outputs = {
            info.name: torch.empty(
                tuple(info.shape), dtype=trt_dtype_to_torch(info.dtype), device="cuda"
            )
            for info in infos
        }
        runner.run((inputs, outputs))
        results.append(
            (runner.selected_logits((inputs, outputs)).clone(), outputs["present_key_value_0"])
        )
    assert "last_token_logits" in outputs, "The optimized engine must return one row per request"
    assert outputs["last_token_logits"].shape == (batch, config.vocab_size)
    # Changing GEMM's row count can change BF16 rounding; cache computation
    # retains all rows and must remain bitwise identical at valid positions.
    torch.testing.assert_close(results[1][0], results[0][0], atol=0.2, rtol=0.02)
    for i, (length, count) in enumerate(zip(lengths, counts)):
        assert torch.equal(
            results[1][1][i, :, :, : length + count], results[0][1][i, :, :, : length + count]
        )
