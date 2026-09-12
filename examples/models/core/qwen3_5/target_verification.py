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

"""Synchronous K=1 correctness reference; independent of the C++ executor.

Uses continuous attention KV and paged GDN/Conv records. Each request owns
three records: the committed prefix and two scratch records for verification.
A commit changes the selected record and effective KV length together. This
module intentionally does not implement scheduling, EOS handling, or a drafter.
"""

from dataclasses import dataclass

import tensorrt as trt
import torch

from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.functional import RopeEmbeddingUtils
from tensorrt_llm.models.qwen35.config import Qwen35Config
from tensorrt_llm.runtime.session import Session, TensorInfo


@dataclass(frozen=True)
class VerificationResult:
    """New tokens [N, 2], padded with -1, and accepted-draft flags [N] on CUDA.

    The current token was already returned by prefill/the preceding step.
    Rejection emits the correction; acceptance emits the draft and bonus token.
    The last emitted token is pending and is processed by the next engine call.
    """

    tokens: torch.Tensor
    accepted_draft: torch.Tensor
    target_logits: torch.Tensor


@dataclass(frozen=True)
class _CommittedState:
    lengths: torch.Tensor  # CPU: processed tokens, excluding the pending token.
    slots: torch.Tensor  # CPU: one GDN/Conv record index per request.
    kv: dict[str, torch.Tensor]
    logits: torch.Tensor  # CUDA: predicts the pending token.
    hidden_states: torch.Tensor | None = None


class Qwen35VerificationSession:
    """Fixed-batch BF16/TP=1 greedy verification using full state snapshots.

    Build the engine with paged_state=True, paged_kv_cache=False, and one
    external draft token. Calls synchronize the current CUDA stream. A single
    instance/session must not be used concurrently or across CUDA devices.
    """

    def __init__(
        self, session: Session, config: Qwen35Config, batch_size: int, max_seq_len: int
    ) -> None:
        if config.dtype != "bfloat16" or config.mapping.world_size != 1:
            raise ValueError("Qwen3.5 verification requires BF16 and TP=PP=CP=1")
        if batch_size < 1 or max_seq_len < 2:
            raise ValueError("batch_size must be positive and max_seq_len must be at least two")
        if max_seq_len > config.max_position_embeddings:
            raise ValueError("max_seq_len exceeds the MRoPE cache capacity")
        self._session = session
        self._config = config
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._attention_ids = [
            idx for idx, kind in enumerate(config.decoder_layer_types) if kind == "full_attention"
        ]
        self._linear_ids = [
            idx for idx, kind in enumerate(config.decoder_layer_types) if kind == "linear_attention"
        ]
        self._input_names = {
            session.engine.get_tensor_name(idx)
            for idx in range(session.engine.num_io_tensors)
            if session.engine.get_tensor_mode(session.engine.get_tensor_name(idx))
            == trt.TensorIOMode.INPUT
        }
        required = {"state_snapshot_slot_mapping", "spec_decoding_use"}
        required.update(f"past_key_value_{idx}" for idx in range(len(self._attention_ids)))
        required.update(f"recurrent_state_ptr_{idx}" for idx in self._linear_ids)
        if not self._attention_ids or not self._linear_ids or not required <= self._input_names:
            raise ValueError(
                "Expected a Qwen3.5 K=1 engine with continuous KV and paged recurrent state"
            )

        self._state: _CommittedState | None = None
        self._prompt_lengths: torch.Tensor | None = None
        self._state_bytes = (
            config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_key_head_dim
            * 4
        )
        self._conv_dim = (
            2 * config.linear_num_key_heads * config.linear_key_head_dim
            + config.linear_num_value_heads * config.linear_value_head_dim
        )
        history = config.linear_conv_kernel_dim - 1
        record_bytes = self._state_bytes + self._conv_dim * history * 2
        self._records = {
            idx: torch.zeros((3 * batch_size, record_bytes), dtype=torch.uint8, device=self._device)
            for idx in self._linear_ids
        }
        self._pointers = {}
        for idx, records in self._records.items():
            self._pointers[f"recurrent_state_ptr_{idx}"] = torch.tensor(
                [records.data_ptr()], dtype=torch.int64
            )
            self._pointers[f"conv_state_ptr_{idx}"] = torch.tensor(
                [records.data_ptr() + self._state_bytes], dtype=torch.int64
            )
        _, cache = RopeEmbeddingUtils.create_sinusoidal_positions_for_attention_plugin(
            num_pos=config.max_position_embeddings,
            dim=config.rotary_embedding_dim,
            theta=config.rotary_base,
        )
        self._mrope_cache = (
            torch.from_numpy(cache).to(self._device).expand(batch_size, -1).contiguous()
        )

    def _require_state(self) -> _CommittedState:
        if self._state is None:
            raise RuntimeError("Call prefill before decoding")
        return self._state

    @property
    def current_tokens(self) -> torch.Tensor:
        """Pending tokens [N] on CUDA, already emitted but not yet processed."""
        return self._require_state().logits.argmax(dim=-1).to(torch.int32)

    @property
    def past_lengths(self) -> torch.Tensor:
        """Committed effective KV lengths [N] on the CPU."""
        return self._require_state().lengths.clone()

    @property
    def last_hidden_states(self) -> torch.Tensor:
        """Normalized packed target states from the last forward, including rejected rows."""
        hidden = self._require_state().hidden_states
        if hidden is None:
            raise RuntimeError("Build the target with capture_mtp_hidden_states=True")
        return hidden

    def reset(self) -> None:
        """Release committed KV and reuse recurrent slots for a fresh batch."""
        self._state = None
        self._prompt_lengths = None
        for records in self._records.values():
            records.zero_()

    def _run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        missing = self._input_names - inputs.keys()
        if missing:
            raise ValueError(f"Missing verification inputs: {sorted(missing)}")
        inputs = {name: inputs[name] for name in self._input_names}
        infos = self._session.infer_shapes(
            [
                TensorInfo(name, torch_dtype_to_trt(value.dtype), value.shape)
                for name, value in inputs.items()
            ]
        )
        outputs = {
            info.name: torch.empty(
                tuple(info.shape), dtype=trt_dtype_to_torch(info.dtype), device=self._device
            )
            for info in infos
        }
        stream = torch.cuda.current_stream(self._device)
        if not self._session.run(inputs, outputs, stream.cuda_stream):
            raise RuntimeError(
                "Qwen3.5 verification engine execution failed; state was not committed"
            )
        stream.synchronize()
        return outputs

    def _inputs(
        self,
        tokens: torch.Tensor,
        query_lengths: torch.Tensor,
        past_lengths: torch.Tensor,
        source_slots: torch.Tensor,
        target_slots: torch.Tensor,
        kv: dict[str, torch.Tensor],
        *,
        context: bool = False,
        verification: bool = False,
    ) -> dict[str, torch.Tensor]:
        batch = self._batch_size
        ends = query_lengths.cumsum(0).to(device=self._device, dtype=torch.int32)
        num_tokens = 2 if verification else 1
        context_lengths = query_lengths if context else self._prompt_lengths
        return {
            **kv,
            **self._pointers,
            "input_ids": tokens,
            "position_ids": past_lengths.to(self._device),
            "last_token_ids": (
                torch.arange(1, tokens.numel() + 1, dtype=torch.int32, device=self._device)
                if verification
                else ends
            ),
            "context_lengths": context_lengths.to(self._device),
            "host_context_lengths": context_lengths,
            "sequence_length": (past_lengths + query_lengths).to(self._device),
            "host_past_key_value_lengths": past_lengths,
            "host_request_types": torch.full((batch,), int(not context), dtype=torch.int32),
            "host_has_initial_state": torch.full((batch,), int(not context), dtype=torch.int32),
            "host_max_attention_window_sizes": torch.full(
                (len(self._attention_ids),), self._max_seq_len, dtype=torch.int32
            ),
            "host_sink_token_length": torch.zeros(1, dtype=torch.int32),
            "host_runtime_perf_knobs": torch.full((16,), -1, dtype=torch.int64),
            "host_context_progress": torch.zeros(1, dtype=torch.int64),
            "cache_indirection": torch.zeros(
                (batch, 1, self._max_seq_len), dtype=torch.int32, device=self._device
            ),
            "mrope_rotary_cos_sin": self._mrope_cache,
            "mrope_position_deltas": torch.zeros(
                (batch, 1), dtype=torch.int32, device=self._device
            ),
            "gated_delta_cu_seqlens": torch.cat([ends.new_zeros(1), ends]),
            "source_state_slot_mapping": source_slots.to(self._device),
            "target_state_slot_mapping": target_slots.to(self._device),
            "state_snapshot_slot_mapping": torch.full(
                (tokens.numel(),), -1, dtype=torch.int32, device=self._device
            ),
            "spec_decoding_use": torch.tensor([int(verification)], dtype=torch.int32),
            "spec_decoding_generation_lengths": torch.full(
                (batch,), num_tokens, dtype=torch.int32, device=self._device
            ),
            "spec_decoding_position_offsets": torch.arange(
                num_tokens, dtype=torch.int32, device=self._device
            )
            .expand(batch, -1)
            .contiguous(),
            "spec_decoding_packed_mask": torch.tensor(
                [1, 3] if verification else [1], dtype=torch.int32, device=self._device
            )
            .repeat(batch)
            .view(-1, 1),
        }

    def _publish(
        self,
        outputs: dict[str, torch.Tensor],
        lengths: torch.Tensor,
        slots: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        kv = {
            f"past_key_value_{local_idx}": outputs[f"present_key_value_{layer_idx}"]
            for local_idx, layer_idx in enumerate(self._attention_ids)
        }
        # All three caches describe the same processed prefix when this single
        # state reference is published. Rejected KV tails are outside lengths.
        torch.cuda.current_stream(self._device).synchronize()
        self._state = _CommittedState(lengths, slots, kv, logits, outputs.get("mtp_hidden_states"))

    def prefill(self, prompts: list[list[int]]) -> torch.Tensor:
        """Prefill a fresh batch and return the first pending greedy tokens [N]."""
        if self._state is not None:
            raise RuntimeError("Call reset before prefilling another batch")
        if len(prompts) != self._batch_size or any(not prompt for prompt in prompts):
            raise ValueError("Expected one nonempty prompt per request")
        lengths = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.int32)
        if (lengths > self._max_seq_len).any():
            raise ValueError("Prompt exceeds max_seq_len")
        tokens = torch.tensor(
            [token for prompt in prompts for token in prompt],
            dtype=torch.int32,
            device=self._device,
        )
        if ((tokens < 0) | (tokens >= self._config.vocab_size)).any():
            raise ValueError("Prompt token is outside the vocabulary")
        config = self._config
        kv = {
            f"past_key_value_{idx}": torch.zeros(
                (
                    self._batch_size,
                    2,
                    config.num_key_value_heads,
                    self._max_seq_len,
                    config.head_size,
                ),
                dtype=torch.bfloat16,
                device=self._device,
            )
            for idx in range(len(self._attention_ids))
        }
        slots = torch.arange(self._batch_size, dtype=torch.int32) * 3
        inputs = self._inputs(
            tokens, lengths, torch.zeros_like(lengths), slots, slots, kv, context=True
        )
        outputs = self._run(inputs)
        self._prompt_lengths = lengths
        self._publish(outputs, lengths, slots, outputs["logits"])
        return self.current_tokens

    def decode(self) -> torch.Tensor:
        """Consume pending tokens once, commit, and return the next greedy tokens."""
        state = self._require_state()
        if (state.lengths + 1 > self._max_seq_len).any():
            raise ValueError("Decode exceeds max_seq_len")
        slots = state.slots // 3 * 3 + (state.slots + 1) % 3
        inputs = self._inputs(
            self.current_tokens,
            torch.ones_like(state.lengths),
            state.lengths,
            state.slots,
            slots,
            state.kv,
        )
        outputs = self._run(inputs)
        self._publish(outputs, state.lengths + 1, slots, outputs["logits"])
        return self.current_tokens

    def step(self, draft_tokens: torch.Tensor) -> VerificationResult:
        """Verify one external candidate per request and commit the accepted prefix.

        Args:
            draft_tokens: Integer tensor [N]; one candidate following each
                pending token. No sampling distribution is needed for greedy.
        """
        state = self._require_state()
        if draft_tokens.shape != (self._batch_size,) or draft_tokens.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Expected one integer draft token per request")
        draft_tokens = draft_tokens.to(device=self._device)
        if ((draft_tokens < 0) | (draft_tokens >= self._config.vocab_size)).any():
            raise ValueError("Draft token is outside the vocabulary")
        draft_tokens = draft_tokens.to(dtype=torch.int32)
        if (state.lengths + 2 > self._max_seq_len).any():
            raise ValueError(
                "Verification exceeds max_seq_len; use single-token decode for the tail"
            )
        first_slots = state.slots // 3 * 3 + (state.slots + 1) % 3
        second_slots = state.slots // 3 * 3 + (state.slots + 2) % 3
        tokens = torch.stack([self.current_tokens, draft_tokens], dim=1).flatten()
        inputs = self._inputs(
            tokens,
            torch.full_like(state.lengths, 2),
            state.lengths,
            state.slots,
            second_slots,
            state.kv,
            verification=True,
        )
        # Snapshot S1 after current; the normal target store writes S2 after
        # draft. Neither store aliases S0, even when the request rejected its
        # previous draft and therefore switched to the other scratch record.
        inputs["state_snapshot_slot_mapping"][::2] = first_slots.to(self._device)
        outputs = self._run(inputs)
        logits = outputs["logits"].reshape(self._batch_size, 2, -1)
        predicted = logits.argmax(dim=-1).to(torch.int32)
        accepted = predicted[:, 0] == draft_tokens
        accepted_cpu = accepted.to(device="cpu", dtype=torch.int32)
        slots = torch.where(accepted_cpu.bool(), second_slots, first_slots)
        next_logits = logits[torch.arange(self._batch_size, device=self._device), accepted.long()]
        emitted = torch.stack([predicted[:, 0], torch.where(accepted, predicted[:, 1], -1)], dim=1)
        result = VerificationResult(emitted, accepted, logits)
        self._publish(outputs, state.lengths + 1 + accepted_cpu, slots, next_logits)
        return result

    def recurrent_states(self) -> dict[str, torch.Tensor]:
        """Copy committed states in the non-paged engine's layout for comparison."""
        slots = self._require_state().slots.to(device=self._device, dtype=torch.long)
        config = self._config
        states = {}
        history = config.linear_conv_kernel_dim - 1
        for idx, records in self._records.items():
            selected = records.index_select(0, slots)
            states[f"present_recurrent_state_{idx}"] = (
                selected[:, : self._state_bytes]
                .contiguous()
                .view(torch.float32)
                .reshape(
                    self._batch_size,
                    config.linear_num_value_heads,
                    config.linear_value_head_dim,
                    config.linear_key_head_dim,
                )
            )
            states[f"present_conv_state_{idx}"] = (
                selected[:, self._state_bytes :]
                .contiguous()
                .view(torch.bfloat16)
                .reshape(self._batch_size, self._conv_dim, history)
                .transpose(1, 2)
                .contiguous()
            )
        return states
