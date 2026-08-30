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

from typing import Optional, Union

import tensorrt as trt

from ..functional import Tensor, gated_delta_rule
from ..module import Module


class GatedDeltaRule(Module):
    """Gated Delta Rule plugin layer."""

    def __init__(
        self,
        num_q_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        chunk_size: int,
        dtype: Union[str, trt.DataType],
        state_dtype: Union[str, trt.DataType] = "float32",
        state_slot_stride_bytes: int = 0,
        remove_input_padding: Optional[bool] = None,
        paged_state: Optional[bool] = None,
        use_qk_l2norm: bool = True,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.state_dtype = state_dtype
        self.state_slot_stride_bytes = state_slot_stride_bytes
        self.remove_input_padding = remove_input_padding
        self.paged_state = paged_state
        self.use_qk_l2norm = use_qk_l2norm

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        log_decay: Tensor,
        beta: Tensor,
        state: Tensor,
        host_request_types: Tensor,
        cu_seqlens: Tensor,
        state_slot_mapping: Tensor,
        host_has_initial_state: Tensor,
        target_state_slot_mapping: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        return gated_delta_rule(
            query=query,
            key=key,
            value=value,
            log_decay=log_decay,
            beta=beta,
            state=state,
            host_request_types=host_request_types,
            cu_seqlens=cu_seqlens,
            state_slot_mapping=state_slot_mapping,
            host_has_initial_state=host_has_initial_state,
            target_state_slot_mapping=target_state_slot_mapping,
            num_q_heads=self.num_q_heads,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            state_dtype=self.state_dtype,
            state_slot_stride_bytes=self.state_slot_stride_bytes,
            remove_input_padding=self.remove_input_padding,
            paged_state=self.paged_state,
            use_qk_l2norm=self.use_qk_l2norm,
        )
