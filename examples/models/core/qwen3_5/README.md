<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 Dense Text Model

> [!WARNING]
> The `convert_checkpoint.py` and `trtllm-build` workflow is part of the legacy
> TensorRT engine backend. This initial Qwen3.5 implementation is intentionally
> limited to dense text-only BF16 inference with TP=1 or TP=2.

This directory contains the checkpoint conversion entry point for the dedicated
Qwen3.5 TensorRT graph in
[`tensorrt_llm/models/qwen35`](../../../../tensorrt_llm/models/qwen35/). The
implementation does not reuse the earlier Qwen graph because Qwen3.5 combines
full-attention layers with gated-delta linear-attention layers.

## Current support

| Capability | Status |
| --- | --- |
| Dense, text-only Qwen3.5 | Supported |
| Data type | BF16 weights and activations |
| Parallelism | TP=1 or TP=2, PP=1, CP=1 |
| Position embedding | MRoPE |
| Full attention | Supported |
| Gated-delta linear attention | Supported with paged recurrent state |
| KV/state management | Paged KV cache and paged linear-attention state |
| Generation | Prefill followed by decode, beam width 1 |
| Prefix cache / block reuse | Not supported |
| Mixed prefill and decode batch | Not supported |
| Quantization | Not supported |
| Vision inputs | Not supported; vision weights are ignored during conversion |
| Standard legacy runtime/generation | Supported through `ModelRunnerCpp` |
| Speculative decoding, disaggregated serving, and offload | Not supported |

The implementation has been validated with the `Qwen3.5-2B` Hugging Face
checkpoint. Other dense Qwen3.5 sizes using the same text-decoder architecture
are expected to use the same graph, but have not been validated yet.

## Checkpoint conversion

Convert a Hugging Face checkpoint to the TensorRT LLM checkpoint format:

```bash
python examples/models/core/qwen3_5/convert_checkpoint.py \
    --model_dir /path/to/Qwen3.5-2B \
    --output_dir /path/to/qwen3_5_bf16_tp1 \
    --dtype bfloat16
```

The converter applies the following Qwen3.5-specific transformations:

- It keeps model weights in BF16, except `A_log` and `dt_bias`, which remain
  FP32 for gated-delta recurrence.
- It converts zero-centered RMSNorm weights using
  `(1 + weight).to(torch.bfloat16)`.
- It converts the `RmsNormGate` weight directly to BF16 without adding one.
- It splits the packed attention `q_proj` into query and attention-gate
  projections.
- It maps Hugging Face `gate_proj`, `up_proj`, and `down_proj` to the TensorRT
  LLM gated-MLP `fc`, `gate`, and `proj` weights, respectively.
- It ignores the vision tower and converts only `model.language_model`.

The script rejects parallel configurations other than TP=1 or TP=2, PP=1, and
CP=1.

For TP=2, the example provides a helper script with TP2-specific default output
directories:

```bash
examples/models/core/qwen3_5/convert_tp2.sh \
    /path/to/Qwen3.5-2B \
    /path/to/qwen3_5_bf16_tp2
```

## Engine build

The following command matches the continuous-cache profile used by the
bottom-level `Session` validation test:

```bash
trtllm-build \
    --checkpoint_dir /path/to/qwen3_5_bf16_tp1 \
    --output_dir /path/to/qwen3_5_engine \
    --max_batch_size 4 \
    --max_input_len 8193 \
    --max_seq_len 8193 \
    --max_num_tokens 8193 \
    --opt_num_tokens 512 \
    --kv_cache_type continuous \
    --gpt_attention_plugin bfloat16 \
    --gemm_plugin bfloat16 \
    --mamba_conv1d_plugin bfloat16
```

Build a TP=2 checkpoint with the matching helper script. The checkpoint
configuration determines that two engine ranks are generated:

```bash
examples/models/core/qwen3_5/build_tp2.sh \
    /path/to/qwen3_5_bf16_tp2 \
    /path/to/qwen3_5_engine_bf16_tp2
```

An 8K profile has a substantial TensorRT execution-context memory requirement.
Run the build and tests on a GPU with enough free memory for both the engine
weights and activation workspace.

For standard runtime generation, build with paged KV cache (the default). The
runtime allocates both attention KV blocks and gated-delta recurrent-state
blocks through the KV cache manager. Prefix-cache block reuse is disabled for
this initial hybrid-model implementation.

## Validation

Set `LLM_MODELS_ROOT` to a directory containing `Qwen3.5-2B`, then run:

```bash
LLM_MODELS_ROOT=/path/to/models pytest -q \
    tests/unittest/trt/model/test_qwen35_convert.py \
    tests/unittest/trt/model/test_qwen35.py
```

The conversion test reads tensors from the real Hugging Face checkpoint and
checks names, values, and data types. The prefill test builds one TensorRT
engine whose maximum sequence length is 8193 and compares bottom-level
`Session` logits against Hugging Face for batch sizes 1, 2, and 4, including an
8193-token input. It also performs four greedy decode steps and compares every
generated token with Hugging Face.

The standard `ModelRunnerCpp` path has additionally been validated with a
Qwen3.5-2B paged engine using paged attention KV cache and paged gated-delta
state. A 17-token prompt followed by four greedy decode steps produced the same
tokens as Hugging Face: `18, 19, 20, 21`.
