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

### Engine-level external-draft verification (K=1)

The first target-verification path supports BF16, TP=1, beam width 1, and
one externally supplied draft token per request. Build the graph with
`max_draft_len=1` and `speculative_decoding_draft_tokens_external=True` in
`Qwen35ForCausalLM.prepare_inputs` (CLI: `--max_draft_len 1
--speculative_decoding_mode draft_tokens_external`). Execute it directly with
`Session`; `ModelRunnerCpp` still rejects speculative hybrid engines until
acceptance and state commit are implemented.

Each generation verification request packs `[current_token, draft_token]`.
For a batch of N requests, supply:

- `last_token_ids = [1, 2, ..., 2*N]` to return logits at every position.
- `gated_delta_cu_seqlens = [0, 2, ..., 2*N]` and an initial Conv/GDN state
  for each request (`host_has_initial_state=1`).
- `spec_decoding_use=[1]` on the host, and device tensors
  `spec_decoding_generation_lengths=[2]*N`,
  `spec_decoding_position_offsets=[[0, 1]]*N`, and causal
  `spec_decoding_packed_mask=[[1], [3]]` repeated N times.
- The prefix KV lengths and `sequence_length=prefix_length+2`, plus the usual
  state-slot mappings and text MRoPE inputs.

Prefill and ordinary single-token decode use `spec_decoding_use=[0]`.
The GDN and Conv plugins use their stateful prefill kernels for two-token
verification. All resulting caches are **tentative**: retain the prefix state
and do not promote the verification state after rejecting a draft. Automatic
acceptance, accepted-prefix snapshots, and cache commit are not included yet.
The engine contains no MTP drafter. Greedy verification is covered by
`test_qwen35_external_draft_verification`, which compares both logits, final
cache state, and the next decode against sequential single-token execution.

### Remaining implementation order

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

## Multimodal image and video demo

`multimodal_demo.py` connects the Hugging Face processor, a TensorRT vision
engine, Qwen3.5 MRoPE preparation, request-local fake token IDs, the prompt
embedding table, and `ModelRunnerCpp`. Images use Pillow and videos use
`ffprobe` plus `ffmpeg` to sample RGB frames without requiring PyAV or
TorchCodec. The demo also compares:

- The TensorRT and Hugging Face pooled vision embeddings.
- The TensorRT and Hugging Face logits for the last context token, including
  cosine similarity and top-k tokens.

The demo supports one local image or video and runtime batch size 1. It builds
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

Enable content-aware LLM prefix-cache reuse by issuing at least two identical
requests through the same executor:

```bash
examples/models/core/qwen3_5/run_multimodal_demo.sh \
    /path/to/Qwen3.5-2B \
    /path/to/qwen3_5_multimodal_engine \
    /path/to/qwen3_5_vision_engine \
    /path/to/image.jpg \
    --kv_cache_enable_block_reuse \
    --prefix_cache_requests 2
```

The cache key includes the image content hash or the whole-video hash. Video
hashes cover the ordered sampled frames, frame indices, FPS, duration, and total
frame count. A video remains distinct from an image even when it contains a
single identical frame. This caches the LLM KV/recurrent state; the demo still
runs the vision encoder once before submitting the repeated LLM requests.

Build a chunked-context LLM engine for the repository-local `women.gif`. Its
115 full-resolution frames produce 10,440 merged video tokens and an input of
about 10,919 tokens:

```bash
QWEN35_MAX_BATCH_SIZE=1 \
QWEN35_MAX_INPUT_LEN=16384 \
QWEN35_MAX_SEQ_LEN=16384 \
QWEN35_MAX_NUM_TOKENS=4096 \
QWEN35_OPT_NUM_TOKENS=4096 \
examples/models/core/qwen3_5/build.sh \
    /path/to/qwen3_5_bf16_tp1 \
    /path/to/qwen3_5_full_video_engine \
    --max_prompt_embedding_table_size 16384 \
    --gather_context_logits
```

Run all frames with Qwen3.5-2B and the prompt `总结一下这段视频`:

```bash
examples/models/core/qwen3_5/run_video_demo.sh \
    /path/to/qwen3_5_full_video_engine \
    /path/to/qwen3_5_full_video_vision_engine
```

The video wrapper defaults to `/root/code_x/Qwen3.5-2B`,
`/root/code_x/women.gif`, all decoded frames, and chunked context. Override the
first two paths with `QWEN35_MODEL_DIR` and `QWEN35_VIDEO`. Pass
`--video_num_frames N` after the engine directories to uniformly sample `N`
frames, or pass `0` to decode every frame. With `max_num_tokens=4096`, the
10,919-token input is processed in multiple prefill chunks while `max_seq_len=16384`
remains the total request limit.

The wrapper loads the source-built TensorRT-LLM plugin and shared libraries used
by `build.sh`. Set `QWEN35_PLUGIN_LIB` when the plugin is in another location.
It exits without normal Python runtime cleanup after printing the result to
avoid an ABI-specific destructor failure when the source bindings and installed
TensorRT Python package use different minor versions.

### Incremental frames and visual-boundary GDN snapshots

The wrapper enables incremental frames and prefix reuse: encode all frames once
as independent video items, then submit `frame1 + prompt`,
`frame1 + frame2 + prompt`, and so on. Intermediate requests generate one token;
the final request uses `--max_new_tokens`. Use `--no-incremental_video_frames`
to process the video as a single item instead.

The experimental `--gdn_visual_boundary_snapshots` switch is **off by default**.
It uses snapshots at the nearest full KV block boundary at or before each
contiguous visual metadata span's end, replacing redundant 256-token snapshots
inside visual spans instead of simply adding more snapshots. Periodic snapshots
in text and the final-full-block snapshot are unchanged. A visual-internal
periodic snapshot is retained if removing it would leave a gap exceeding the
256-token interval between checkpoints, protecting large images and long text
prefixes from producing unschedulable chunks.
For this demo, the span includes `vision_start` and visual placeholders but not
`vision_end`. With 32-token KV blocks, at most 31 tokens of that span remain
after its visual boundary. No padding or partial KV blocks are introduced.
Sparse/run-layout metadata falls back to the regular snapshot policy.

The scheduler, physical recurrent-state allocation, and capacity budget use the
same sorted, deduplicated boundaries. Extra snapshots cost memory and can add
prefill iterations; this is intended for repeated, growing visual prefixes,
not as a general throughput optimization. For the full 115-frame example, start
with `--kv_cache_free_gpu_memory_fraction 0.5`; the needed fraction depends on
available GPU memory. The final-full-block snapshot is also now a mandatory
chunk end when allocated, including with the new switch off, so it contains an
executed state rather than just an allocated slot. Identical pixels alone are insufficient:
the preceding tokens, item hashes, embeddings, and positional encoding must also
remain consistent. A different tail prompt can follow the reusable prefix.

Rebuild native bindings (existing engines do not need rebuilding), then run:

```bash
cmake --build cpp/build --target bindings --parallel 8
bash examples/models/core/qwen3_5/run_video_demo.sh \
    /tmp/qwen35_decode_engine /tmp/qwen35_full_video_vision_engine \
    --max_new_tokens 512 --gdn_visual_boundary_snapshots \
    --kv_cache_free_gpu_memory_fraction 0.5 \
    --incremental_results_path /tmp/qwen35_visual_snapshots.json
```

The Python extension loaded from `tensorrt_llm/` must also be the rebuilt version;
the demo rejects stale bindings when enabling the switch. Internally, this demo
sets `TRTLLM_GDN_VISUAL_BOUNDARY_SNAPSHOTS` before creating the native executor.

For a cache-restoration correctness baseline, keep the same snapshot switch,
frame sampling and prompt, add `--incremental_isolate_requests`, and choose a
different `--incremental_results_path`. This changes only the visual cache keys
per request, preserving embeddings, token IDs, positions and chunk boundaries.
Compare `generated_token_ids` for every frame. A separate fully cache-disabled
baseline uses `--no-gdn_visual_boundary_snapshots --no-kv_cache_enable_block_reuse`;
its different prefill segmentation can change floating-point rounding and greedy
outputs. Compare with reuse enabled but the new switch disabled to measure the
incremental benefit. Reported block counters aggregate attention
and recurrent pools: multiplying them by the KV block size does **not** give
the exact reused token count. Observed reuse is not itself a correctness check.

The following measurements describe the **earlier additive prototype**, before
the replacement policy above. Development validation used the 115-frame
`women.gif` with the two engines above,
84 embedding tokens per frame, 32-token KV blocks, one output token on intermediate
requests, and `--max_new_tokens 512` on the final request:

| Metric | Fixed interval + final full block | Additional visual boundaries |
| --- | ---: | ---: |
| First request with a nonzero hybrid reusable prefix | Frame 4 | Frame 2 |
| Frame 2 reusable prefix (tokens) | 0 | 64 |
| Frame 115 reusable prefix (tokens) | 10,304 | 10,464 |
| Total tokens requiring prefill across 115 requests | 17,829 | 13,925 |

The extra visual snapshots reduced prefill token work by 21.9% in this run.
These figures use the first scheduled chunk's reusable prefix, not aggregate
block counters. Set `TLLM_LOG_LEVEL_BY_MODULE=debug:batchmgr` to inspect the
`context request scheduled` lines (`reusable N`; absent means zero).
That additive policy matched the isolated-request control on all 115 requests and
all 195 generated tokens, including 81 final-response tokens. The token-work
reduction is not a claim of the same percentage wall-clock speedup.
Validation also passed 65 scheduler tests, five selected GDN/KV tests, and
18 Qwen3.5 embedding/input tests.

Replacement-policy validation on the same 115-frame input:

| Metric | Switch off | Earlier additive policy | Replacement policy |
| --- | ---: | ---: | ---: |
| Full GDN snapshots in the final request's prefix | 104 | 143 | 120 |
| Snapshot writes across all 115 requests | 148 | 187 | 164 |
| Total tokens requiring prefill | 17,829 | 13,925 | 13,925 |

One snapshot here includes all 18 GDN layers; counts exclude the live decode
state. The final-prefix counts include inherited snapshots from previous
requests, not just the boundaries planned for a cold request. The replacement
policy removed 23 snapshots and 23 writes relative to the additive prototype
without reducing the total reusable token prefix. Aggregate block hit rates
can change because there are fewer recurrent blocks, even with unchanged token
reuse. The pool capacity remained 205 state slots in all three modes.

The replacement policy matched the same-segmentation, isolated-cache control
on all 115 requests and all 200 generated tokens (86 on the final request).
It also passed 68 scheduler tests, five selected GDN/KV tests, and 18 embedding
tests. The scheduler tests cover text-only inputs, sparse-metadata fallback,
removed periodic state slots, large-image chunk-size safety, and allocation
budget consistency across image lengths and block alignments.

Image preprocessing defaults to `min_pixels=3,136` and
`max_pixels=200,704` so that the Hugging Face eager-attention reference fits on
a typical development GPU. Pass `--min_pixels` and `--max_pixels` to change the
image resize bounds. Larger visual inputs increase vision attention memory
quadratically.

By default, a newly built vision engine is profiled for the current visual
input's pre-merge patch-token count. Pass `--max_vision_tokens N` when building
it if the cache should accept larger later inputs, or pass
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

### Nsight Systems full-input sweep

An additional TP=1 development measurement profiles full-input TTFT from 8K
through 96K without prefix-cache reuse. It separates kernels that can be
identified as full-attention or linear-attention core work and records the
long-context bottleneck transition. See
[Qwen3.5 Long-Context Nsight Systems Analysis](NSYS_LONG_CONTEXT_ANALYSIS.md)
for the configuration, measurements, classification boundaries, and findings.

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
