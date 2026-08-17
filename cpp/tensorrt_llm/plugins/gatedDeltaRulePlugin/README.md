<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Gated Delta Rule Plugin

This directory contains the TensorRT `GatedDeltaRule` plugin and its Triton
ahead-of-time (AOT) kernels for Qwen3.5 decode and packed-ragged prefill.

## Supported configuration

- Linux and CUDA SM89
- BF16 query, key, value, and output
- FP32 recurrent state
- 16 query/key heads
- 16, 32, or 48 value heads
- Key and value head dimensions of 128
- Chunk size 64
- Q/K L2 normalization enabled
- Packed-ragged prefill with input padding removed

## Current status

The standalone TensorRT V3 plugin currently implements and validates both
decode and Triton AOT chunked prefill. Prefill uses the packed-ragged layout:
Q/K/V have shape `[1, T, ...]`, while `cu_seqlens` with shape `[N + 1]`
partitions the concatenated tokens into `N` requests. The prefill path requires
`remove_input_padding=true` and supports requests with different sequence
lengths in the same homogeneous prefill batch.

The functional test currently contains seven packed-prefill parameter cases.
Coverage includes value-head counts 16, 32, and 48; explicit packed batch sizes
2, 4, and 8; initial and zero-state requests; sequence lengths on and off
32-token boundaries; length 4000; and a sequence longer than 8K at length 8193.
The tested long packed batch uses sequence lengths `(32, 96, 127, 4000, 8193)`.

Stateful coverage validates `paged_state=true` prefill followed by three decode
steps, with every step consuming the state written by the preceding step. It
also validates a non-contiguous, out-of-order `state_slot_mapping` of
`[6, 2, 5]` against an eight-slot state pool.

Dynamic-shape coverage builds two TensorRT optimization profiles. The context
profile accepts packed inputs shaped `[1, T, ...]` through `T=8193`, and the
generation profile accepts decode inputs shaped `[B, 1, ...]` through `B=8`.
The complete standalone functional test file currently passes all 26 cases.

A complete SM89 wheel build has been verified with the plugin and all
checked-in decode and prefill cubin archives included. The repository
`build_trtllm_sm89.sh` helper builds wheels by default; use
`--skip-python-env-check` only for the direct CMake path that does not package a
wheel.

The remaining integration work is to connect the plugin to the Qwen3.5
TensorRT engine graph and runtime pipeline. Padded prefill tensors with a Q/K/V
batch dimension greater than one, mixed prefill/decode batches, non-SM89 cubins,
and additional head configurations are not currently supported.

## Prerequisites

Use a TensorRT-LLM development environment with CUDA, TensorRT, CMake, Conan,
and a Python environment containing Triton 3.6. The checked-in cubins are
already sufficient for a normal plugin build; Triton is only required when
regenerating them after a kernel or AOT signature change.

Run all commands below from the TensorRT-LLM repository root.

## Regenerate the Triton cubins

Skip this step when only changing the C++ runner or plugin wiring.

```bash
python3 cpp/tensorrt_llm/plugins/gatedDeltaRulePlugin/aot/compile_decode.py --arch 89
python3 cpp/tensorrt_llm/plugins/gatedDeltaRulePlugin/aot/compile_prefill.py --arch 89
```

The scripts write deterministic `.cubin.tar.zst` archives to
`cpp/tensorrt_llm/plugins/gatedDeltaRulePlugin/cubin/`. A complete SM89 set
contains three decode archives and 24 prefill archives.

The Triton 3.6 AOT ABI appends `global_scratch` and `profile_scratch` launch
arguments. If the Triton version or a kernel signature changes, verify the
generated kernel symbol, shared-memory usage, warp count, and C++ launch
parameter order before committing regenerated archives.

## Build the plugin

### Incremental build

When `cpp/build` is already configured, build only the shared plugin target:

```bash
cmake --build cpp/build \
  --target nvinfer_plugin_tensorrt_llm \
  --parallel 8
```

The resulting library is:

```text
cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
```

The plugin CMake configuration automatically discovers the C++ sources in
this directory and embeds the SM89 cubin archives.

### Configure and build from the repository root

For a new build directory, use the repository build helper so that Conan and
the TensorRT-LLM CMake toolchain are configured consistently:

```bash
python3 scripts/build_wheel.py \
  --cpp_only \
  --configure_cmake \
  --cuda_architectures "89-real" \
  --build_dir cpp/build \
  --job_count 8
```

Use `--clean` only when a clean rebuild is intended, because it removes the
existing build directory before configuring it again.

After changing the set of cubin archives, rerun the configure command above
if the existing build does not automatically reconfigure its globbed sources.

## Verify the build

```bash
test -f cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so
cmake --build cpp/build --target nvinfer_plugin_tensorrt_llm --parallel 8
```

The functional coverage is located in
`tests/unittest/trt/functional/test_gated_delta_rule_plugin.py`. It covers
decode, packed-ragged prefill, paged-state continuity, arbitrary state-slot
mapping, and dynamic context/generation profiles for value-head counts 16, 32,
and 48.
