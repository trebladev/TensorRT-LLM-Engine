<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.5 MTP 第 4 项：持久化 draft KV

已实现固定请求槽位的 paged draft KV。短上下文 batch 4 的两轮配对测试中，
ISL32 平均降低 0.59%，ISL64 持平；没有稳定可确认的性能提升。
全部配对样本的输入、输出 token、accepted/proposals 一致。

此前优化已提交为 `f379934e910a45a780db72b21aaf6b7e2de4eae1`，含 DCO 签名，
全部 pre-commit 检查通过。此报告对应的第 4 项改动单独提交，未包含于上述提交。

## 实现

- 复用 GPT attention plugin 的 paged KV 接口，32 tokens/block，单层、BF16、TP1。
- 每个活跃请求固定拥有一个槽位；batch 输入通过 block offsets 引用该槽位。
- attention 直接读取历史并原位追加 KV，不再逐请求 gather/scatter 整段 KV。
- 仅真实输入推进逻辑长度；padding 写入是暂时的，下一次 append 从逻辑长度覆盖。
- 请求完成、取消、暂停或重新 prefill 时，等待未完成的 draft 工作后回收槽位。
- 不含 prefix sharing、动态块分配或共享 target KV。旧 continuous engine 保留原路径。
- 原生 executor demo 默认构建 paged draft；Python Session 数值参考仍使用 continuous KV。
- `mtp.engine.json` 保存池分配几何信息，必须和对应的 `mtp.engine` 一起使用。

## 测量

RTX 4090 D，CUDA_VISIBLE_DEVICES=0，Qwen3.5-2B，BF16，TP1，K1，greedy，
batch4，OSL64。ISL32/64，每轮3次预热、6次测量，旧→新、新→旧，共12次/配置。
不启用 CUDA graphs/overlap/prefix reuse/chunked context。没有删除任何样本。
同一 target（SHA256 已验证）、同一 short-GDN plugin；仅 draft engine 和 worker runtime 改变。
`run_native.py` 检查 `/proc/self/maps`，确认加载指定的 runtime。

| ISL | 第1+2项后 continuous ms/整批 | 第4项 paged ms/整批 | 延迟降低 | 旧 tokens/s | 新 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 373.521 | 371.322 | 0.59% | 685.370 | 689.429 |
| 64 | 367.540 | 367.540 | 约0% | 696.524 | 696.523 |

分轮均值：

| 轮次 | ISL32 旧/新 ms | ISL64 旧/新 ms |
| --- | --- | --- |
| 0（旧→新） | 369.645 / 374.525 | 368.005 / 366.988 |
| 1（新→旧） | 377.397 / 368.119 | 367.074 / 368.092 |

两轮改善方向不一致，不能把整体0.59%当成稳定收益。
ISL32 accepted/proposals 总数均为1252/1738；ISL64均为1340/1662。
采样进程峰值显存：旧22142 MiB，新22140 MiB；这包含运行时预留池，
不代表 draft KV 本身大小。GPU0另有两个各648 MiB的进程，测试期间保留。

该小模型的 draft KV 只有2个KV heads、head size256。旧布局 batch4、capacity129
一次整段 KV 大约1.008 MiB，进出约2.016 MiB；新池预留5页/请求，共1.25 MiB。
这解释了为何取消拷贝不一定改变总延迟。此前
`../qwen35_mtp_gemm_debug_20260914/README.md` 的独立层测量显示，
last-row draft 的 LM head 约1.084 ms，占其层时间75–79%；第4项并未减少这部分计算。
这是此前 profiling 的定位证据，不能当作本次 paged engine 的精确阶段分解。
本轮只比较 MTP 新旧路径，未重新测 plain 或更长上下文。

## 验证

- paged 整模型 `test_qwen35_native_mtp.py`：6 passed，121.33 s。未放宽断言。
- paged `qwen35MtpWorkerTest`：2 passed。混合宽度容量边界、相反槽位分配、
  32/64/96/128页边界、batch缩减、ID复用均覆盖。
- 同一新运行时配旧 continuous draft，worker测试：2 passed。
- 全部7个改动文件的 pre-commit：通过；ruff和git diff --check：通过。
- `focused_summary.json`：输出及接受数配对校验；`artifacts.json`：文件大小和SHA256。

## Engine 与运行时

以下路径均相对仓库根目录，均为实体文件，未使用/tmp engine或符号链接：

| 用途 | 路径 |
| --- | --- |
| 新 target | `engines/qwen35/mtp_paged/rank0.engine` |
| 新 draft + 元数据 | `engines/qwen35/mtp_paged/mtp.engine`、`mtp.engine.json` |
| 新运行时 | `engines/qwen35/runtime_paged/libtensorrt_llm.so` |
| 对照 engine | `engines/qwen35/mtp/{rank0.engine,mtp.engine,config.json}` |
| 对照运行时 | `engines/qwen35/runtime_flow12/libtensorrt_llm.so` |
| 共用插件 | `cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so` |

## 复现

从仓库根目录运行（需要已构建的运行时和插件）：

```bash
# 仅重建 draft 并复制相同 target 到仓库目录。
CUDA_VISIBLE_DEVICES=0 LLM_MODELS_ROOT=/root/code_x \
PYTHONPATH=$PWD:$PWD/benchmarks/results/qwen35_mtp_paged_20260915 \
QWEN35_TEST_PLUGIN=$PWD/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so \
.venv-3.12/bin/python benchmarks/results/qwen35_mtp_short_verification_20260914/run_with_plugin.py build_draft

# 新旧配对测速及分析。
.venv-3.12/bin/python benchmarks/results/qwen35_mtp_paged_20260915/run_pairs.py
.venv-3.12/bin/python benchmarks/results/qwen35_mtp_paged_20260915/analyze.py

# 整模型正确性。
CUDA_VISIBLE_DEVICES=0 LLM_MODELS_ROOT=/root/code_x PYTHONPATH=. \
LD_LIBRARY_PATH=$PWD/engines/qwen35/runtime_paged \
QWEN35_TEST_PLUGIN=$PWD/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so \
QWEN35_MTP_ENGINE_DIR=$PWD/engines/qwen35/mtp_paged \
.venv-3.12/bin/python benchmarks/results/qwen35_mtp_short_verification_20260914/run_with_plugin.py \
pytest -xq tests/unittest/trt/model/test_qwen35_native_mtp.py

# 独立 C++ worker；需 TensorRT10.15，而系统默认10.13不能读此engine。
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=$PWD/.venv-3.12/lib/python3.12/site-packages/tensorrt_libs \
QWEN35_MTP_ENGINE_DIR=$PWD/engines/qwen35/mtp_paged \
cpp/build/tests/unit_tests/batch_manager/qwen35MtpWorkerTest
```
