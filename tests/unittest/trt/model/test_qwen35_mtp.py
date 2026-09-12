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

"""Native MTP logits/cache equivalence and persistent generation regression tests."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
)
from utils.llm_data import llm_models_root

from examples.models.core.qwen3_5.mtp import Qwen35MTPGenerator, Qwen35MTPSession
from examples.models.core.qwen3_5.mtp_demo import build_draft_session
from examples.models.core.qwen3_5.target_verification_demo import build_session
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.models.qwen35.mtp import Qwen35MTP, convert_mtp_weights


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    root = llm_models_root()
    if root is None or not (root / "Qwen3.5-2B").is_dir():
        pytest.skip("Set LLM_MODELS_ROOT to a directory containing Qwen3.5-2B")
    return root / "Qwen3.5-2B"


@pytest.fixture(scope="module")
def mtp_weights(checkpoint: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for shard in checkpoint.glob("*.safetensors"):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("mtp."):
                    weights[name] = handle.get_tensor(name)
                elif name == "model.language_model.embed_tokens.weight":
                    weights[name] = handle.get_slice(name)[:8192]
    assert "mtp.fc.weight" in weights
    return weights


@pytest.fixture(scope="module")
def draft(checkpoint: Path, mtp_weights: dict[str, torch.Tensor]) -> Qwen35MTPSession:
    config = Qwen35Config.from_hugging_face(checkpoint)
    config.vocab_size = 8192
    config.max_position_embeddings = 128
    model = Qwen35MTP(config)
    weights = convert_mtp_weights(mtp_weights, config)
    weights["lm_head.weight"] = weights["transformer.vocab_embedding.weight"]
    model.load(weights)
    return build_draft_session(model, 64)


def _reference(
    checkpoint: Path, weights: dict[str, torch.Tensor], tokens: torch.Tensor, hidden: torch.Tensor
) -> torch.Tensor:
    config = deepcopy(AutoConfig.from_pretrained(checkpoint).text_config)
    config.layer_types = ["full_attention"]
    config.num_hidden_layers = 1
    config._attn_implementation = "eager"
    layer = Qwen3_5DecoderLayer(config, 0).to(device="cuda", dtype=torch.bfloat16).eval()
    layer.load_state_dict(
        {
            name.removeprefix("mtp.layers.0."): value
            for name, value in weights.items()
            if name.startswith("mtp.layers.0.")
        }
    )
    rotary = Qwen3_5TextRotaryEmbedding(config, device="cuda")

    def norm(value: torch.Tensor, name: str) -> torch.Tensor:
        weight = weights[name].cuda().float() + 1
        return (
            value.float()
            * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + config.rms_norm_eps)
            * weight
        ).to(torch.bfloat16)

    with torch.inference_mode():
        embedding = weights["model.language_model.embed_tokens.weight"].cuda()
        fused = torch.cat(
            [
                norm(F.embedding(tokens, embedding), "mtp.pre_fc_norm_embedding.weight"),
                norm(hidden, "mtp.pre_fc_norm_hidden.weight"),
            ],
            dim=-1,
        )
        fused = F.linear(fused, weights["mtp.fc.weight"].cuda()).unsqueeze(0)
        count = tokens.numel()
        positions = torch.arange(count, device="cuda").unsqueeze(0)
        mask = torch.full((count, count), float("-inf"), device="cuda", dtype=torch.bfloat16)
        mask = mask.triu(1)[None, None]
        output = layer(fused, position_embeddings=rotary(fused, positions), attention_mask=mask)
        return F.linear(norm(output, "mtp.norm.weight"), embedding)[0].float()


def test_mtp_logits_and_incremental_cache(
    checkpoint: Path, mtp_weights: dict[str, torch.Tensor], draft: Qwen35MTPSession
) -> None:
    torch.manual_seed(42)
    tokens = torch.arange(11, 29, device="cuda", dtype=torch.int32)
    hidden = torch.randn((18, draft.config.hidden_size), device="cuda", dtype=torch.bfloat16)
    expected = _reference(checkpoint, mtp_weights, tokens.long(), hidden)
    draft.reset()
    full = draft.append(tokens, hidden)
    # BF16 norm conversion/fused attention need a numerical tolerance, but the
    # dominant distribution must agree with the independent HF decoder.
    torch.testing.assert_close(full.float(), expected, atol=0.35, rtol=0.03)
    full_kv = draft._kv[..., :18, :].clone()
    draft.reset()
    chunks = [draft.append(tokens[:13], hidden[:13])]
    for start, end in ((13, 14), (14, 16), (16, 18)):
        chunks.append(draft.append(tokens[start:end], hidden[start:end]))
    incremental = torch.cat(chunks).float()
    # Different GEMM shapes round BF16 differently even for the context prefix.
    torch.testing.assert_close(incremental, full.float(), atol=0.2, rtol=0.02)
    torch.testing.assert_close(draft._kv[..., :18, :], full_kv, atol=0.125, rtol=0.03)
    assert draft.length == 18
    # Reset must not reuse a previous request's KV contents.
    draft.reset()
    repeated = draft.append(tokens[:3], hidden[:3])
    torch.testing.assert_close(repeated.float(), full[:3].float(), atol=0.2, rtol=0.02)


def test_mtp_validation_does_not_advance_cache(draft: Qwen35MTPSession) -> None:
    draft.reset()
    with pytest.raises(ValueError, match="hidden states"):
        draft.append(torch.tensor([1], device="cuda"), torch.zeros((2, 2048), device="cuda"))
    assert draft.length == 0


class _Target:
    """Deterministic target trace to force accept/reject and EOS in both positions."""

    def reset(self) -> None:
        self.position = 0
        self.last_hidden_states = torch.tensor([[0.0], [1.0]])

    def prefill(self, prompts: list[list[int]]) -> torch.Tensor:
        self.reset()
        return torch.tensor([2])

    def decode(self) -> torch.Tensor:
        self.position += 1
        return torch.tensor([2 + self.position])

    def step(self, candidate: torch.Tensor) -> SimpleNamespace:
        expected = 3 + self.position
        accepted = candidate.item() == expected
        self.last_hidden_states = torch.tensor([[expected - 1.0], [expected * 1.0]])
        self.position += 1 + int(accepted)
        return SimpleNamespace(
            accepted_draft=torch.tensor([accepted]),
            tokens=torch.tensor([[expected, expected + 1 if accepted else -1]]),
        )


class _Draft:
    def __init__(self, accept: bool) -> None:
        self.accept = accept
        self.max_seq_len = 64
        self.device = "cpu"
        self.reset()

    def reset(self) -> None:
        self.pairs = []

    def append(self, tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        self.pairs.extend(zip(hidden[:, 0].tolist(), tokens.tolist()))
        logits = torch.zeros((1, 32))
        logits[0, tokens[-1] + 1 if self.accept else 0] = 1
        return logits


@pytest.mark.parametrize("accept", [False, True])
@pytest.mark.parametrize(
    "limit,eos", [(1, ()), (2, ()), (7, ()), (8, ()), (8, (2,)), (8, (3,)), (8, (4,)), (8, (6,))]
)
def test_generation_alignment_limits_and_eos(
    accept: bool, limit: int, eos: tuple[int, ...]
) -> None:
    target, drafter = _Target(), _Draft(accept)
    generator = Qwen35MTPGenerator(target, drafter)
    result = generator.generate([0, 1], limit, eos)
    expected = list(range(2, 2 + limit))
    if eos and eos[0] in expected:
        expected = expected[: expected.index(eos[0]) + 1]
    assert result.tokens == expected
    assert all(token == hidden + 1 for hidden, token in drafter.pairs)
    assert result.accepted_drafts == (result.verified_drafts if accept else 0)
    # Reusing the generator starts a new request with fresh histories.
    assert generator.generate([0, 1], limit, eos) == result


@pytest.fixture(scope="module")
def generator(checkpoint: Path) -> Qwen35MTPGenerator:
    model = Qwen35MTP.from_hugging_face(checkpoint)
    draft = build_draft_session(model, 96)
    del model
    target = build_session(checkpoint, 96, capture_mtp_hidden_states=True)
    return Qwen35MTPGenerator(target, draft)


@pytest.mark.parametrize("limit", [1, 2, 16, 17])
def test_native_mtp_matches_target_greedy(
    generator: Qwen35MTPGenerator, checkpoint: Path, limit: int
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    prompt = tokenizer.encode("The capital of France is", add_special_tokens=True)
    eos = (tokenizer.eos_token_id,)
    result = generator.generate(prompt, limit, eos)
    generator.target.reset()
    expected = generator.target.prefill([prompt]).tolist()
    while len(expected) < limit and expected[-1] not in eos:
        expected.extend(generator.target.decode().tolist())
    assert result.tokens == expected
    if limit >= 16:
        assert result.verified_drafts > 0
        assert result.accepted_drafts > 0


def test_draft_failed_enqueue_retains_history(
    draft: Qwen35MTPSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    draft.reset()
    tokens = torch.tensor([10, 11], device="cuda", dtype=torch.int32)
    hidden = torch.ones((2, draft.config.hidden_size), device="cuda", dtype=torch.bfloat16)
    draft.append(tokens, hidden)
    previous = draft._kv.clone()
    with monkeypatch.context() as patch:
        patch.setattr(draft.session, "run", lambda *args, **kwargs: False)
        with pytest.raises(RuntimeError, match="not committed"):
            draft.append(tokens[:1], hidden[:1])
    assert draft.length == 2
    torch.testing.assert_close(draft._kv, previous)
    draft.append(tokens[:1], hidden[:1])
    assert draft.length == 3
