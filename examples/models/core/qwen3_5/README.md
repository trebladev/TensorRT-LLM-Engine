<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 Dense Model

> [!WARNING]
> The `convert_checkpoint.py` and `trtllm-build` workflow is part of the legacy
> TensorRT engine backend. This initial Qwen3.5 implementation is limited to
> dense BF16 inference with TP=1 or TP=2. Multimodal execution uses a separate
> TensorRT vision engine and the prompt-tuning input path.

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
| Prefix cache / block reuse | Supported for full-attention KV and gated-delta recurrent state with beam width 1 |
| Mixed prefill and decode batch | Not supported |
| Quantization | Not supported |
| Vision inputs | Supported for batch-1 image input by `multimodal_demo.py` |
| Standard legacy runtime/generation | Supported through `ModelRunnerCpp` |
| Speculative decoding, disaggregated serving, and offload | Not supported |

The implementation has been validated with the `Qwen3.5-2B` Hugging Face
checkpoint. Other dense Qwen3.5 sizes using the same text-decoder architecture
are expected to use the same graph, but have not been validated yet.

## Planned MTP enablement

> [!NOTE]
> Multi-token prediction (MTP) is not supported by this TensorRT graph yet.
> This section records the intended implementation and validation order; it is
> not a description of currently available functionality.

MTP target verification runs the engine with the current token followed by up
to `K` draft tokens. The engine therefore needs a generation profile that
supports `K + 1` input tokens, returns target logits for every verification
position, and preserves the state associated with the accepted prefix.

Attention KV cache support can reuse much of the existing speculative-decoding
infrastructure because KV entries retain a token dimension. Rejected entries
can be ignored by updating the effective sequence length and, when necessary,
reordering the accepted path. Gated-delta recurrent state is more restrictive:
the state after all draft tokens cannot be rolled back by changing a sequence
length. Convolution state has the same commit problem, although its state is
small enough to retain per-step snapshots.

The implementation should proceed in the following order:

| Stage | Implementation | Test types |
| --- | --- | --- |
| 1 | Define input, logits, accepted-length, and cache-commit semantics | Configuration, interface, and golden-trace tests |
| 2 | Build the target engine with a `K + 1` generation profile and accept externally supplied draft tokens | Engine build/profile, dynamic-shape, and logits-equivalence tests |
| 3 | Add speculative attention and KV cache commit/rollback | Attention-plugin, paged-KV, block-boundary, and cache-lifecycle tests |
| 4 | Support multi-token gated-delta execution using full intermediate state snapshots as the correctness reference | Kernel, plugin, state-equivalence, and snapshot-selection tests |
| 5 | Save and promote the convolution state selected by the accepted prefix | Convolution-kernel and accepted-index promotion tests |
| 6 | Integrate acceptance with atomic KV, gated-delta, and convolution-state commit | Acceptance-unit, cache-consistency, and external-draft end-to-end tests |
| 7 | Convert and integrate the Qwen3.5 MTP modules and their hidden-state/token history | Weight-conversion, MTP-logits, draft-token, and engine-integration tests |
| 8 | Replace full gated-delta snapshots with compact replay | Replay-kernel, full-state-oracle, random-acceptance, and long-sequence tests |
| 9 | Integrate replay metadata and buffers with the cache manager and inflight batching | PNAT, double-buffer, slot-reuse, mixed-batch, and scheduler tests |
| 10 | Enable optimized and distributed configurations | Stress, compatibility, memory, performance, and non-MTP regression tests |

The first correctness milestone should deliberately use a restricted
configuration: Qwen3.5 BF16, TP=1, beam width 1, greedy decoding, a fixed draft
length, a small batch, externally supplied draft tokens, and full gated-delta
intermediate states. This keeps target verification independent of the MTP
drafter and provides a reference for compact replay.

### Gated-delta compact replay

Full intermediate recurrent-state snapshots are sufficient for correctness,
but their storage scales with the draft length and the full recurrent-state
size. This is likely to become a memory-capacity and bandwidth bottleneck at
useful batch sizes. Compact replay should instead retain one checkpoint state
and a bounded history of the update factors needed to reconstruct accepted
state, such as `x`, `B`, `dt`, and cumulative `dA` values.

The replay cache requires:

- A bounded, double-buffered update history for every recurrent-state slot.
- A previous-number-of-accepted-tokens (PNAT) value identifying the valid
  history after the checkpoint.
- An active-buffer index and checkpoint/overflow handling.
- Acceptance-side metadata updates that exclude rejected draft tokens.
- Numerical comparison against the full intermediate-state implementation over
  random acceptance sequences and multiple checkpoint cycles.

Convolution state should initially continue to use per-step intermediate
snapshots followed by accepted-index promotion. Acceptance is expected to run
outside the TensorRT target graph, after target logits are available, with the
runtime or batch manager committing KV, recurrent, and convolution state as one
logical operation.

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
- It maps both `model.visual` and `model.language_model` weights. The standard
  checkpoint still stores the converted vision weights even though
  `trtllm-build` builds only the LLM engine from it.

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
blocks through the KV cache manager.

## Multimodal image demo

`multimodal_demo.py` connects the Hugging Face processor, a TensorRT vision
engine, Qwen3.5 MRoPE preparation, request-local fake token IDs, the prompt
embedding table, and `ModelRunnerCpp`. It also compares:

- The TensorRT and Hugging Face pooled vision embeddings.
- The TensorRT and Hugging Face logits for the last context token, including
  cosine similarity and top-k tokens.

The initial demo supports one local image and runtime batch size 1. It builds
and caches the vision engine on its first run. The LLM engine must enable prompt
tuning and context logits. For example, build an engine with room for up to
4,096 total prompt-table rows:

```bash
examples/models/core/qwen3_5/build.sh \
    /path/to/qwen3_5_bf16_tp1 \
    /path/to/qwen3_5_multimodal_engine \
    --max_prompt_embedding_table_size 4096 \
    --gather_context_logits
```

Run the end-to-end demo and make the HF comparison enforce its default
thresholds:

```bash
examples/models/core/qwen3_5/run_multimodal_demo.sh \
    /path/to/Qwen3.5-2B \
    /path/to/qwen3_5_multimodal_engine \
    /path/to/qwen3_5_vision_engine \
    /path/to/image.jpg \
    --prompt "Describe this image in detail." \
    --check
```

The wrapper loads the source-built TensorRT-LLM plugin and shared libraries used
by `build.sh`. Set `QWEN35_PLUGIN_LIB` when the plugin is in another location.
It exits without normal Python runtime cleanup after printing the result to
avoid an ABI-specific destructor failure when the source bindings and installed
TensorRT Python package use different minor versions.

The demo defaults to `min_pixels=3,136` and `max_pixels=200,704` so that the
Hugging Face eager-attention reference fits on a typical development GPU. Pass
`--min_pixels` and `--max_pixels` to change the processor's resize bounds.
Larger images increase vision attention memory quadratically.

By default, a newly built vision engine is profiled for the current image's
pre-merge patch-token count. Pass `--max_vision_tokens N` when building it if
the cache should accept larger later images, or pass
`--rebuild_vision_engine` to replace an existing profile.

The LLM prompt-table capacity is measured after spatial merging. For batch size
`B` and at most `V` visual tokens per request, build with
`--max_prompt_embedding_table_size >= B * V`. The demo reports both the
pre-merge patch-token count and the merged prompt-token count and fails with a
rebuild hint when the cached vision profile or LLM prompt table is too small.

## Prefix-cache validation demo

The prefix-cache demo turns the default long prompt into two sequential
requests on the same `ModelRunnerCpp` instance. The first request populates the
cache with the long-text prefix. The second request sends the exact same token
prefix followed by `总结上述内容`, then reports request-level statistics when
retained by the executor and the cumulative reused-block delta as a fallback.

```bash
examples/models/core/qwen3_5/run_prefix_cache_demo.sh \
    /path/to/qwen3_5_engine \
    /path/to/Qwen3.5-2B
```

The script aligns the first request to a 256-token recurrent-state snapshot
boundary. A successful run ends with `PASS` and a reused block count greater
than zero. Both requests must run in the same process; invoking `run_demo.sh`
twice creates two executors and cannot reuse the first request's cache.

Add `--compare_direct_ttft` to measure streaming time to first token for the
cold prefix request, the warm prefix-cache summary request, and the same
combined input on a fresh executor:

```bash
examples/models/core/qwen3_5/run_prefix_cache_demo.sh \
    /path/to/qwen3_5_engine \
    /path/to/Qwen3.5-2B \
    --compare_direct_ttft
```

Each executor receives one untimed prompt with the same length but a different
first token before measurement. This warms the same long-context CUDA and
TensorRT paths while ensuring that the measured prompt cannot reuse its prefix.

### 50K/100K TTFT benchmark

Build a TP=2 engine with a 100K sequence profile on GPUs 0 and 1:

```bash
examples/models/core/qwen3_5/build_long_context_tp2.sh \
    /path/to/qwen3_5_bf16_tp2 \
    /tmp/qwen35_engine_long_ttft_tp2
```

Run the benchmark:

```bash
examples/models/core/qwen3_5/run_long_context_ttft.sh \
    /tmp/qwen35_engine_long_ttft_tp2 \
    /path/to/Qwen3.5-2B
```

Each trial uses one executor for all measured requests. It first runs an
untimed, nonmatching 100K warmup, then measures a 100K cold request, a 50K
prefix population request, and a 100K request that shares the populated 50K
prefix. Trials alternate between `cold -> prefix -> hit` and
`prefix -> hit -> cold` order to reduce ordering bias. The final report includes
the median, minimum, and maximum TTFT, per-trial speedup, and reused-block
validation. The default is three trials with one warmup request per trial:

```bash
QWEN35_NUM_TRIALS=3 QWEN35_WARMUP_REQUESTS=1 \
examples/models/core/qwen3_5/run_long_context_ttft.sh \
    /tmp/qwen35_engine_long_ttft_tp2 \
    /path/to/Qwen3.5-2B
```

Here K means 1024 tokens. The reusable boundary is therefore 51,200 tokens and
the long request contains 102,400 tokens. Set
`QWEN35_CUDA_VISIBLE_DEVICES` to select different physical GPUs; the
single-process orchestrator maps its worker device IDs to the two visible
devices. Set `QWEN35_KV_CACHE_FREE_GPU_MEMORY_FRACTION` to tune cache capacity.
Pass `--request_order cold_first` or `--request_order hit_first` to disable
alternating request order.

The following reference result was measured with Qwen3.5-2B BF16, TP=2, two
NVIDIA GeForce RTX 4090 D GPUs, a 4,096-token context chunk budget, a 0.7 KV
cache free-memory fraction, two trials, and one 100K warmup per trial:

| Request | Median TTFT |
| --- | ---: |
| 50K cold prefix population | 8,415.00 ms |
| 100K cold input | 20,548.47 ms |
| 100K input with a 50K prefix hit | 12,626.38 ms |

The 50K prefix hit reduced median TTFT by 7,922.09 ms (38.55%) and produced a
1.63x speedup. Both request orders reused exactly 1,800 blocks: 1,600
full-attention KV blocks at 32 tokens per block and 200 gated-delta recurrent
state snapshots at 256 tokens per snapshot. These numbers are a reference for
this machine and configuration; use the aggregate output from the benchmark
when comparing other GPUs, engine profiles, or cache capacities.

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
