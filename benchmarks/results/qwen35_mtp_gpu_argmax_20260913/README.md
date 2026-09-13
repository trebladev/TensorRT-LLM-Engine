# Qwen3.5 TRT MTP：GPU argmax 优化验证

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

2026-09-13，基于 `60b0de491`。本轮将 native MTP 最后一行 draft logits 的 CPU argmax 改为复用已有 MTP GPU greedy reduction，只回传每请求一个 int32 token ID，保留相同最大值时选择最小 token ID 的行为。target/draft 引擎和模型计算不变，无须重建引擎。

结论：并发 4 吞吐提升约 9%–10%，并发 2 提升约 5%。这只缓解了回退：并发 4 相对普通解码仍慢约 18%–24%。

## 吞吐

单位为输出 tokens/s，按总输出 token 数除以总批次耗时汇总。

| ISL | 并发 | 原 MTP | GPU argmax MTP | 相对原 MTP | 普通解码 | 相对普通解码 |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 1 | 197.3 | 203.5 | +3.1% | 182.0 | +11.8% |
| 32 | 2 | 310.6 | 325.9 | +4.9% | 363.6 | -10.4% |
| 32 | 4 | 490.1 | 533.4 | +8.8% | 703.6 | -24.2% |
| 64 | 1 | 201.4 | 207.2 | +2.9% | 184.0 | +12.7% |
| 64 | 2 | 326.7 | 342.5 | +4.8% | 360.3 | -4.9% |
| 64 | 4 | 510.5 | 563.6 | +10.4% | 689.8 | -18.3% |

并发 4 的请求级中位 TPOT：ISL 32 从 7.87 ms 降至 7.12 ms；ISL 64 从 7.07 ms 降至 6.47 ms。完整延迟汇总见 `summary.csv`。

## 方法和适用范围

- GPU 0：RTX 4090 D；Qwen3.5-2B BF16，TP=PP=CP=1，K=1，greedy。ISL 32/64，OSL 固定 64，并发 1/2/4。
- 沿用上一轮普通引擎 `/tmp/qwen35_plain_benchmark_engine` 和 MTP 引擎 `/tmp/qwen35_batch_mtp_engine`。引擎 SHA256 和配置保存在各 JSON 中。
- 每种模式两轮、每配置每轮预热 2 次、测量 5 次。运行顺序为 before_0 → plain_0 → argmax_0 → argmax_1 → before_1 → plain_1；优化探索和回归测试位于测量运行之间。两轮用于检查顺序/波动，不是置信区间。
- 沿用 `mtp_benchmark.py`：从 enqueue_requests 前计时到全部 final，排除模型加载、tokenization、请求准备；关闭 CUDA Graph、prefix reuse、chunked context。不是持续到达的 serving 压测，也不代表已调优普通解码性能。
- 同机其他 GPU 持续有负载，GPU 0 还有其他进程显存占用；保留全部 NVML 样本，不能完全排除共享 CPU/同卡干扰。小幅单请求收益应谨慎解读。
- KV 容量仍沿用原测量路径，混合注意力分配可能忽略 max_tokens 上限；本次没有据显存差值推断必要开销。

## 正确性与代码检查

- 两轮共 60 个测量批次、140 条输出序列、8,960 个输出 token，优化前后逐 token 完全一致；每批接受数及 proposal 数完全一致。
- `mtpPackedGreedyTest` 通过：六种词表大小（1、3、129、256、257、248320），随机值、跨线程相同最大值、全负无穷、首位/末位 NaN，检查 packed 最后行和 CPU std::max_element 一致。
- `tests/unittest/trt/model/test_qwen35_native_mtp.py`：3 passed。包括 EOS、预算、缓存边界、slot/request 复用、并发、混合预算与错峰到达。
- 修改的源码通过完整 pre-commit 检查；日志已保存。
- 本轮验证的是优化前后 MTP 的一致性。普通解码与 MTP 在更广泛输入下的数值等价问题仍未在本轮解决。

## 未保留的实验与下一步

先尝试了以短递推 kernel 替换 target GDN 两 token 验证的 chunk prefill 路径。12 个状态/快照测试通过，但并发 4 的端到端回归在第 17 个 token 出现分歧。该实验代码与 cubin 已撤回，运行插件已恢复；不能用状态误差容限通过代替端到端验证。实验补丁和失败日志暂存于 `/tmp/qwen35_mtp_perf_fix`，未纳入最终实现。

后续主要工作仍是降低 target verification 成本并保持数值行为，以及减少 draft 分组 forward 和 KV 打包/回写。此次没有改动这些路径。

## 复现与产物

```bash
# 增量构建；本机链接测试程序需指向当前 venv 的 Torch 库。
LD_LIBRARY_PATH=/root/code_x/TensorRT-LLM/.venv-3.12/lib/python3.12/site-packages/torch/lib cmake --build cpp/build --target tensorrt_llm mtpPackedGreedyTest --parallel 8
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/root/code_x/TensorRT-LLM/.venv-3.12/lib/python3.12/site-packages/torch/lib cpp/build/tests/unit_tests/kernels/mtpPackedGreedyTest
# 将构建的运行库安装到当前 Python 环境后：
CUDA_VISIBLE_DEVICES=0 LLM_MODELS_ROOT=/root/code_x QWEN35_MTP_ENGINE_DIR=/tmp/qwen35_batch_mtp_engine .venv-3.12/bin/python -m pytest tests/unittest/trt/model/test_qwen35_native_mtp.py -q
CUDA_VISIBLE_DEVICES=0 LLM_MODELS_ROOT=/root/code_x .venv-3.12/bin/python -m examples.models.core.qwen3_5.mtp_benchmark --model_dir /root/code_x/Qwen3.5-2B --engine_dir /tmp/qwen35_batch_mtp_engine --mode mtp --output /tmp/mtp_gpu_argmax.json
.venv-3.12/bin/python benchmarks/results/qwen35_mtp_gpu_argmax_20260913/summarize.py
```

- `before_0/1.json`、`argmax_0/1.json`、`plain_0/1.json` ：原始测量（JSON 仅压缩空白）；运行日志保留在本地。
- `runtime_sha256.json`：原始/优化运行库和恢复后的插件哈希；运行库备份保存在 `/tmp/qwen35_mtp_perf_fix`。
- `summary.csv`、`summarize.py`：汇总及严格输出一致性检查。
- `argmax_kernel_test.log`、`argmax_native_tests.log`、`precommit.log`：验证记录，保留在本地，未纳入版本控制。
