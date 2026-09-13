<!-- Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

# Qwen3.5 TRT MTP：GDN 快慢路径数值排查

日期：2026-09-13。基线提交：`689cfb478`。模型：Qwen3.5-2B。

结论：实验性的 FP32 短递推与现有 chunk verification 路径具有不同的数值语义。
同输入、同初始状态下，首个 GDN 已产生差异，早于接受判断和状态提交。
归一化 epsilon 位置确有不一致，但在本样本中影响很小；Q/K 以及其他中间张量的 BF16 舍入影响显著。
当前证据不足以支持直接启用短递推，现有输出一致性要求应保留。

## 端到端复现

- GPU 0，原生 C++ executor，batch 4，greedy，关闭 EOS、CUDA graph、cache reuse 和 chunked context。
- prompt token IDs 分别为 `range(1, n + 1)`，n 为 17、63、64、65；输出长度 24。
- 复用 `/tmp/qwen35_batch_mtp_engine` 中的 target 和 MTP engine。
- 独立 debug plugin 同时支持两条路径，通过进程内 `ctypes.CDLL` 重定向加载，未替换安装目录中的库。
- 64-token prompt 前 16 个输出一致，第 17 个输出出现分歧；其余三个 prompt 的全部输出一致。

| 路径 | token 80 logit | token 26685 logit | 选中 token |
|---|---:|---:|---:|
| chunk | 21.75 | 21.75 | 80 |
| 短递推 | 21.75 | 21.875 | 26685 |

对应 target callback 的输入长度为 80，logits shape 为 `[2, 1, 248320]`，比较第一行。
0.125 是这一量级的一个 BF16 间隔。实际 logits 已不同，不能归因于 argmax 的平局规则。
完整输出及该次 callback 见 [token_divergence.json](token_divergence.json)。

## 最早的算子差异

捕获第一轮 verification 的 18 个 GDN 调用。首层 Q/K/V、log_decay、beta、有效源状态均逐位相同，
source/target/snapshot mapping 和 cu_seqlens 也相同。
首层输出的最大绝对差为 0.00048828125，相对 L2 差约 0.1792%，6550/16384 个元素不同。
下一 GDN 层的 Q/K/V 已出现传播后的差异。

见 [capture_comparison.json](capture_comparison.json)。状态比较只包含有效 source/snapshot 槽位；
snapshot mapping 中的 `-1` 被排除，不能把它当成 Python 的最后一行索引。
此证据定位的是算术分歧的起点，不代表已穷尽验证所有后续缓存行为。

## 同输入归一化消融

固定首层 chunk 捕获的输入和状态，逐次恢复状态池，直接运行实验 Triton kernel。

| 实验 | 输出相对 L2 差 | 有效 snapshot 相对 L2 差 |
|---|---:|---:|
| 原短递推 | 0.179200% | 0.040431% |
| epsilon 移入 sqrt | 0.179200% | 0.040431% |
| 再将 normalized Q/K 转 BF16 后转回 FP32 | 0.152150% | 0.032376% |

原短递推使用 `x / (sqrt(sum(x*x)) + eps)`；chunk 使用 `x / sqrt(sum(x*x) + eps)`，并存成 BF16。
本样本中 Q/K 的最小范数约为 0.204/0.207，修正 epsilon 引入的 normalized Q/K 相对变化约为百万分之一。
该结果不能推广到范数更小的输入。

JIT 回放与捕获的 AOT 短递推并非逐位相同：输出仅 1/16384 个元素不同，最大差约 9.31e-10；
有效 snapshot 最大差约 4.77e-7。这一残差远小于快慢路径差异，仍保留在原始指标中。
见 [ablation.json](ablation.json)。只修归一化尚未通过端到端验证，不能作为已完成修复。

## 中间 BF16 舍入对照

对每个请求的首 verification token 构建显式数值对照；此时 chunk 的下三角变换对角项为 1，
可直接展开 W/U、初始状态乘积、residual、QK 和输出计算。
对照使用 CPU FP64 做计算，并在 chunk 对应位置显式转 BF16，最后输出也转 BF16。
它用于区分舍入影响，不声称复现 GPU 的 FP32 reduction/FMA 顺序。

| 对照：省略哪一项 BF16 舍入 | 输出相对 L2 差 | 不同输出元素 / 8192 |
|---|---:|---:|
| 全部保留 | 0.000806% | 1 |
| dot 使用的初始 H | 0.162913% | 1835 |
| W/U | 0.124657% | 1008 |
| residual | 0.079041% | 900 |
| QK | 0.109018% | 909 |
| 上述全部省略 | 0.162520% | 2691 |

全部保留时，有效 snapshot 最大差约 4.77e-7，相对 L2 差约 2.56e-8。
只要省略主要舍入点，输出或状态误差就明显增大。
这些消融不是可相加的贡献比例；仍有第二个 verification token 的 chunk 展开及 GPU reduction 顺序需要对齐。
见 [rounding_oracle.json](rounding_oracle.json)。

另尝试了现有 Python chunk reference 的整段回放，但未获得可靠匹配（包含非有限输出）；
改变 state kernel 的 autotune 配置也未解决。该回放不作为本报告的数值基准，原因尚未确认。
上述对照均以实际 plugin 捕获值为基准。

## 建议的实现方向

1. 保留现有 chunk 路径。将本次真实输入、有效初始状态、快照和输出作为短序列优化的对照样本。
2. 针对 K=1 的两 token verification，优先尝试保留 chunk 舍入点与运算次序的特化，减少 padding、metadata 和 launch 开销。
3. 先验证同输入 output、首 token snapshot、最终 target state，再跑 batch 1/2/4 的原生 MTP 输出一致性回归。
4. 全部通过后，用无捕获、无 callback 的运行测量性能。本次捕获含同步，不能用于性能结论。

FP32 中间精度更高不保证与已有 BF16 chunk 输出一致；目前没有可安全启用的快路径修复。

## 本机复查

大体积输入、引擎和隔离库保留在 `/tmp`，未收入本目录。以下命令依赖本机已保存的捕获文件：

```sh
CUDA_VISIBLE_DEVICES=0 .venv-3.12/bin/python benchmarks/results/qwen35_gdn_numerics_20260913/ablation.py
CUDA_VISIBLE_DEVICES=0 .venv-3.12/bin/python benchmarks/results/qwen35_gdn_numerics_20260913/rounding_oracle.py
```

可通过 `QWEN35_GDN_DEBUG_DIR` 修改数据目录；默认 `/tmp/qwen35_gdn_debug`。
脚本将重算的 JSON 写入该目录。本目录中的 JSON 为此次运行的归档。
复现脚本、捕获补丁及日志位于该目录的 `reproduce.py`、`instrumented_experiment.patch`、`slow.log`、`fast.log`。
实验 kernel 的原始新文件包为 `/tmp/qwen35_mtp_perf_fix/short_verification_new_files.tar.gz`。

端到端捕获命令（fast 运行另设 `QWEN35_GDN_FAST=1`，并将 slow 路径名替换为 fast）：

```sh
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/code_x/TensorRT-LLM LLM_MODELS_ROOT=/root/code_x \
QWEN35_GDN_CAPTURE=/tmp/qwen35_gdn_debug/slow_inputs QWEN35_GDN_CAPTURE_LIMIT=18 \
.venv-3.12/bin/python /tmp/qwen35_gdn_debug/reproduce.py \
  --plugin /tmp/qwen35_gdn_debug/debug_plugin.so --output /tmp/qwen35_gdn_debug/slow.json
```

生产源码已恢复，本次提交仅包含调查记录和回放脚本。
安装库 SHA-256 与调查前备份一致：`66e43ea5f5e21bd6875de9b7f99392077acc327c72a70c0684279db1fce027ae`。
CMake 构建目录的库也与备份一致：`872520ca07cf91d03f476206048290f0f7925e978d87b5978da3cc775e2242c9`。
