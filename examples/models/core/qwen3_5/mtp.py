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

"""Persistent, synchronous K=1 MTP using two TensorRT engines (BF16, TP=1)."""

from dataclasses import dataclass

import tensorrt as trt
import torch

from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.runtime import Session
from tensorrt_llm.runtime.session import TensorInfo

from .target_verification import Qwen35VerificationSession


class Qwen35MTPSession:
    """One request's MTP KV history; append only accepted target/token pairs.

    Args:
        session: TensorRT draft engine with continuous KV and K=1 profiles.
        config: Target model dimensions, BF16 and world size one.
        max_seq_len: Maximum number of cached token/hidden-state pairs.
    """

    def __init__(self, session: Session, config: Qwen35Config, max_seq_len: int) -> None:
        if config.mapping.world_size != 1 or config.dtype != "bfloat16":
            raise ValueError("MTP requires BF16 and TP=PP=CP=1")
        if not 2 <= max_seq_len <= config.max_position_embeddings:
            raise ValueError("Invalid MTP cache capacity")
        self.session = session
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._names = {
            session.engine.get_tensor_name(i)
            for i in range(session.engine.num_io_tensors)
            if session.engine.get_tensor_mode(session.engine.get_tensor_name(i))
            == trt.TensorIOMode.INPUT
        }
        if not {"target_hidden_states", "past_key_value_0"}.issubset(self._names):
            raise ValueError("Expected a continuous-KV MTP draft engine")
        _, rope = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=config.max_position_embeddings,
            dim=config.rotary_embedding_dim,
            theta=config.rotary_base,
        )
        self._rope = torch.from_numpy(rope).to(self.device).reshape(1, -1)
        self._kv: torch.Tensor | None = None
        self.length = 0
        self._prompt_length = 0

    def reset(self) -> None:
        """Release the previous request's draft cache."""
        self._kv = None
        self.length = 0
        self._prompt_length = 0

    def append(self, tokens: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        """Append pairs [T] / [T,H], returning draft logits [T,V] on CUDA.

        The first call primes all prompt positions. Later calls append one or
        two accepted positions; rejected target hidden states must be omitted.
        A failed enqueue never publishes new KV or its effective length.
        """
        context = self._kv is None
        count = tokens.numel()
        if tokens.ndim != 1 or tokens.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected an integer token vector")
        if count < 1 or (not context and count not in (1, 2)):
            raise ValueError("MTP generation must append one or two accepted positions")
        if hidden_states.shape != (count, self.config.hidden_size):
            raise ValueError("Target hidden states must match the supplied token positions")
        if self.length + count > self.max_seq_len:
            raise ValueError("MTP history exceeds max_seq_len")
        if ((tokens < 0) | (tokens >= self.config.vocab_size)).any():
            raise ValueError("MTP token is outside the vocabulary")
        kv = self._kv
        if context:
            kv = torch.zeros(
                (1, 2, self.config.num_key_value_heads, self.max_seq_len, self.config.head_size),
                dtype=torch.bfloat16,
                device=self.device,
            )
        prompt_length = count if context else self._prompt_length

        def cpu(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32)

        def gpu(values: list[int] | list[list[int]]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=self.device)

        inputs = {
            "input_ids": tokens.to(device=self.device, dtype=torch.int32).contiguous(),
            "target_hidden_states": hidden_states.to(
                device=self.device, dtype=torch.bfloat16
            ).contiguous(),
            "past_key_value_0": kv,
            "position_ids": gpu([self.length]),
            "last_token_ids": gpu([count]),
            "context_lengths": gpu([prompt_length]),
            "host_context_lengths": cpu([prompt_length]),
            "sequence_length": gpu([self.length + count]),
            "host_past_key_value_lengths": cpu([self.length]),
            "host_request_types": cpu([int(not context)]),
            "host_max_attention_window_sizes": cpu([self.max_seq_len]),
            "host_sink_token_length": cpu([0]),
            "host_runtime_perf_knobs": torch.full((16,), -1, dtype=torch.int64),
            "host_context_progress": torch.zeros(1, dtype=torch.int64),
            "cache_indirection": torch.zeros(
                (1, 1, self.max_seq_len), dtype=torch.int32, device=self.device
            ),
            "mrope_rotary_cos_sin": self._rope,
            "mrope_position_deltas": gpu([[0]]),
            "spec_decoding_use": cpu([int(not context and count == 2)]),
            "spec_decoding_generation_lengths": gpu([2 if count == 2 else 1]),
            "spec_decoding_position_offsets": gpu([[0, 1] if count == 2 else [0]]),
            "spec_decoding_packed_mask": gpu([[1], [3]] if count == 2 else [[1]]),
        }
        missing = self._names - inputs.keys()
        if missing:
            raise ValueError(f"Missing MTP inputs: {sorted(missing)}")
        inputs = {name: inputs[name] for name in self._names}
        infos = self.session.infer_shapes(
            [TensorInfo(name, torch_dtype_to_trt(t.dtype), t.shape) for name, t in inputs.items()]
        )
        outputs = {
            info.name: torch.empty(
                tuple(info.shape), dtype=trt_dtype_to_torch(info.dtype), device=self.device
            )
            for info in infos
        }
        stream = torch.cuda.current_stream(self.device)
        if not self.session.run(inputs, outputs, stream.cuda_stream):
            raise RuntimeError("MTP engine execution failed; draft state was not committed")
        stream.synchronize()
        self._kv = outputs["present_key_value_0"]
        self.length += count
        self._prompt_length = prompt_length
        return outputs["logits"]


@dataclass(frozen=True)
class MTPGenerationResult:
    """Generated tokens (including EOS) and K=1 draft acceptance counts."""

    tokens: list[int]
    accepted_drafts: int
    verified_drafts: int


class Qwen35MTPGenerator:
    """Keep target and draft caches resident for a complete greedy request.

    This correctness baseline processes one request at a time. Both neural
    forwards run as TensorRT engines; PyTorch manages buffers and greedy
    token selection. Each generate call resets both histories, also after a
    preceding failure. Instances must not be used concurrently.
    """

    def __init__(self, target: Qwen35VerificationSession, draft: Qwen35MTPSession) -> None:
        self.target = target
        self.draft = draft

    def generate(
        self, prompt: list[int], max_new_tokens: int, eos_token_ids: tuple[int, ...] = ()
    ) -> MTPGenerationResult:
        """Generate with real MTP candidates, stopping at EOS or the output limit."""
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if not prompt or len(prompt) + max_new_tokens > self.draft.max_seq_len:
            raise ValueError("Prompt and output limit exceed the cache capacity")
        self.target.reset()
        self.draft.reset()
        emitted = self.target.prefill([prompt]).tolist()
        accepted = verified = 0
        if emitted[-1] in eos_token_ids or len(emitted) == max_new_tokens:
            return MTPGenerationResult(emitted, accepted, verified)
        shifted = torch.tensor(prompt[1:] + emitted, dtype=torch.int32, device=self.draft.device)
        logits = self.draft.append(shifted, self.target.last_hidden_states)
        while len(emitted) < max_new_tokens:
            if max_new_tokens - len(emitted) == 1:
                emitted.extend(self.target.decode().tolist())
                break
            candidate = logits[-1:].argmax(dim=-1).to(torch.int32)
            result = self.target.step(candidate)
            count = 1 + int(result.accepted_draft.item())
            accepted += count - 1
            verified += 1
            new_tokens = result.tokens[0, :count]
            for token in new_tokens.tolist():
                emitted.append(token)
                if token in eos_token_ids or len(emitted) == max_new_tokens:
                    return MTPGenerationResult(emitted, accepted, verified)
            # The MTP prefix already covers the old pending token. Append
            # (h_current, correction) on rejection; on acceptance append
            # (h_current, draft), (h_draft, bonus). Never append rejected h_draft.
            logits = self.draft.append(new_tokens, self.target.last_hidden_states[:count])
        return MTPGenerationResult(emitted, accepted, verified)
