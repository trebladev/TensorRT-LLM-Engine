<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.5 Long-Context Nsight Systems Analysis

This document records a development measurement of Qwen3.5-2B long-context
prefill on one NVIDIA GeForce RTX 4090 D. It is not a portable performance
claim. Results depend on the GPU, engine profile, TensorRT-LLM revision, input
contents, and profiling configuration.

## Measurement configuration

- TensorRT engine backend, BF16, TP=1, batch size 1.
- One full input per measurement; prefix-cache block reuse was disabled.
- The engine supported a 256K sequence length with an 8,192-token
  `max_num_tokens` budget, so longer inputs used chunked prefill.
- Input lengths ranged from 8K through 96K in 8K increments, where K means
  1,024 tokens.
- Each measurement used a non-captured 8K warmup request before profiling the
  interval from request submission to the first generated token.
- Nsight Systems 2025.5.1 collected CUDA, kernel, and NVTX activity.

## Classification

The following measurements are sums of kernels whose ownership can be
identified from their names. They are not complete layer times.

- Full-attention core: `fmha_v2`, `kernel_mha`, and
  `applyBiasRopeUpdateKVCacheV2`.
- Linear-attention core: `mamba_conv1d`, `chunk_gated_delta_rule`,
  `chunk_fwd_kernel_o`, `recompute_w_u`, `l2norm_fwd`, and recurrent-state or
  chunk-metadata kernels.
- Generic GEMM: GEMM kernels that cannot be reliably assigned to a complete
  full-attention, linear-attention, or MLP layer from Nsight Systems names
  alone.

Generic GEMM is an unclassified subset of the remaining kernels. Do not add it
to the full- and linear-attention values and interpret the sum as a complete
model breakdown.

## Results

All times are milliseconds. Percentages in parentheses use total GPU kernel
time as the denominator.

| Input | TTFT | Kernel time | Full core | Linear core | Generic GEMM | Full / Linear |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 432.967 | 261.744 | 18.021 (6.88%) | 64.952 (24.82%) | 152.683 (58.33%) | 0.28 |
| 16K | 750.345 | 549.356 | 66.949 (12.19%) | 129.791 (23.63%) | 302.398 (55.05%) | 0.52 |
| 24K | 1,073.860 | 868.450 | 146.647 (16.89%) | 195.039 (22.46%) | 452.415 (52.09%) | 0.75 |
| 32K | 1,422.869 | 1,217.863 | 256.419 (21.05%) | 260.211 (21.37%) | 602.714 (49.49%) | 0.99 |
| 40K | 1,805.403 | 1,597.643 | 396.481 (24.82%) | 325.253 (20.36%) | 753.244 (47.15%) | 1.22 |
| 48K | 2,233.484 | 2,008.491 | 566.658 (28.21%) | 390.937 (19.46%) | 904.063 (45.01%) | 1.45 |
| 56K | 2,647.569 | 2,442.055 | 763.874 (31.28%) | 455.170 (18.64%) | 1,052.082 (43.08%) | 1.68 |
| 64K | 3,189.349 | 2,913.450 | 991.637 (34.04%) | 520.539 (17.87%) | 1,206.207 (41.40%) | 1.91 |
| 72K | 3,662.832 | 3,404.413 | 1,250.109 (36.72%) | 584.548 (17.17%) | 1,350.624 (39.67%) | 2.14 |
| 80K | 4,174.507 | 3,930.889 | 1,538.117 (39.13%) | 649.324 (16.52%) | 1,500.200 (38.16%) | 2.37 |
| 88K | 4,735.819 | 4,492.919 | 1,859.601 (41.39%) | 714.966 (15.91%) | 1,650.937 (36.75%) | 2.60 |
| 96K | 5,364.758 | 5,086.421 | 2,209.118 (43.43%) | 781.194 (15.36%) | 1,804.577 (35.48%) | 2.83 |

## Findings

- At 8K, generic GEMM was the largest classified aggregate. The identifiable
  linear-attention core took 64.952 ms, compared with 18.021 ms for the
  full-attention core.
- At 32K, the full- and linear-attention core times were nearly equal. Full
  attention exceeded linear attention at 40K.
- At 48K, FMHA became the longest individual kernel: 561.91 ms, compared with
  538.56 ms for the longest individual GEMM kernel.
- From 40K through 72K, full-attention core time exceeded linear-attention core
  time, while the aggregate generic GEMM time remained larger.
- At 80K, full-attention core time exceeded aggregate generic GEMM time for the
  first time. Full attention was the main classified bottleneck from 80K
  through 96K.
- Increasing the input length by 12x, from 8K to 96K, increased TTFT by 12.39x,
  linear-attention core time by 12.03x, generic GEMM time by 11.82x, and
  full-attention core time by 122.59x. Linear attention and GEMM therefore
  scaled approximately with input length, while full attention grew strongly
  superlinearly.
- At 96K, full-attention core time accounted for 43.43% of kernel time, generic
  GEMM for 35.48%, and linear-attention core time for 15.36%. Full attention
  accounted for 73.88% of the explicitly classified attention-core time.
- Kernel timeline coverage increased from 97.90% at 8K to 99.54% at 96K. The
  longest observed idle gap was less than approximately 2.25 ms, indicating
  that long-context TTFT was dominated by continuous GPU execution rather
  than large host-side launch gaps.

## Evidence limits

Kernel timeline coverage is an upper-bound activity signal, not measured SM
utilization. These reports did not collect GPU metrics or CPU sampling, so they
cannot establish occupancy, register pressure, instruction stalls, or memory
bandwidth saturation. Kernel-internal root-cause analysis requires a separate
Nsight Compute capture.

All reports passed the Nsight Systems report integrity, GPU mapping, NVTX, and
CUDA runtime-to-kernel correlation checks. The profiling process for the 16K
through 88K runs exited with a known `munmap_chunk()` error after each report
had already been written; report validation did not find missing activity.
