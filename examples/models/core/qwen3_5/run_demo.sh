#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"

python_bin="${QWEN35_PYTHON:-python}"
engine_dir="${1:-${QWEN35_ENGINE_DIR:-/tmp/qwen35_engine_bf16_tp1}}"
model_dir="${2:-${QWEN35_MODEL_DIR:-/root/code_x/Qwen3.5-2B}}"
default_input_text='1.2.2 基于神经辐射场的重建方法
新视角合成（Novel View Synthesis, NVS）是三维重建领域的一个重要分支，
其目标是从给定视角的图片中渲染出场景的新视角图像。传统方法中，使用网
格和点云作为场景表示的方式，通常依赖于深度信息和运动结构恢复（Structure
from Motion, SfM）[34]来渲染新视角，然而这些方法在生成高质量图像方面面临
挑战。相比之下，基于多平面图像（Multi-Plane Images, MPIs）的场景表示方
法[35-42]通过将场景表示为特征图或堆叠的图像，能够生成高质量的渲染结果，
但这类方法在视角多样性方面存在一定的局限性。
2020 年，NeRF（Neural Radiance Fields）[5]的提出彻底革新了新视角合成领
域。NeRF 首次引入了神经辐射场作为场景表示方法，结合体积渲染技术，实现
了高质量的新视角合成。具体而言，NeRF 通过神经网络建模辐射场的参数（颜
色和密度），将三维空间坐标和视角方向映射到这些参数上，并利用透明度混合
（alpha-blending）技术进行像素颜色渲染，得到不同分辨率的图像。自此，基于
神经辐射场的场景表达方法逐渐成为该领域的研究热点。
1）NeRF 渲染质量改进
虽然原始版本的 NeRF 能够渲染出高质量的图片，但其渲染仍然存在局限性，
许多工作针对 NeRF 的渲染做了针对性的改进。2021 年，NeRF-W[43]对 NeRF 进
行了一系列扩展，以解决非标准化图像（即非同一台相机或有失真的图像）的问
题，从而能够从互联网上获取的无结构化信息的图像集合中实现精确重建。2022
4
浙江省硕士学位论文
年，NeILF[44]将场景光照表示为神经入射光场，并将材质属性表示为由多层感
知器（Multilayer Perceptron，MLP）建模的表面双向反射分布函数（Bidirectional
Reflectance Distribution Function，BRDF）；Point-NeRF[45]利用带有相关神经特征
的神经 3D 点云来建模辐射场，结合了体积渲染和多视图立体几何的优势。2023
年，MobileNeRF[46]引入了一种基于纹理多边形的新 NeRF 表示，能够通过标准
渲染管线高效合成新图像，并在包括手机在内的多种计算平台上实现实时渲染；
Tetra-NeRF[47]提出使用通过 Delaunay 三角剖分获得的四面体自适应表示，而不
是均匀细分或基于点的表示，从而实现高效训练与高质量渲染；Zip-NeRF[48]通
过结合 Mip-NeRF 360[49]和哈希编码技术，有效解决了基于网格表示的抗锯齿问
题。
2）NeRF 重建质量改进
NeRF 的渲染质量得益于神经辐射场的隐式表达，但是隐式表达无法直接获
得表面信息，因此 NeRF 重建的三维模型质量不佳。许多工作通过改进 NeRF，
获得了更好的三维模型。
在 DVR[50]和 IDR[51]这类表面渲染方法的基础上，许多工作利用占据函数
或符号距离函数（Signed Distance Function，SDF）来描述场景，并通过梯度下
降方法优化模型。2021 年，UNISURF[52]通过体积渲染优化了一个二值占据函
数。这种方法不再依赖于手动标注的掩码信息，而是通过神经网络直接从多视
角图像中推断出物体的表面信息，其中体积渲染（volume rendering）在这里起到
了关键作用，它通过在三维空间中进行采样和积分，逐步逼近物体的真实表面。
VolSDF[53]进一步扩展了这一思想，将其应用于 SDF。通过将体渲染与 SDF 结
合，并使用 Eikonal Loss 来归一化隐式函数来保证重建稳定性，VolSDF 能够更精
确地重建复杂的几何形状，尤其是在重建复杂拓扑结构的物体时。NeuS[54]则深
入分析了使用体渲染优化 SDF 场时可能引入的偏差问题，为解决这个问题 NeuS
引入了一种无偏且具有遮挡感知能力的加权算法。这种算法能够更准确地评估
每个采样点对最终表面重建的贡献，从而减少偏差，并提高重建表面的精度。上
述这些工作的改进已经可以从神经辐射场中提取到一定精度的三维模型。
2022 年以来，在上述工作的基础上，更多高质量的工作进一步提高了重建
效果。HF-NeuS[55]引入了一个新的 MLP 来模拟位移场，捕捉更精细的高频细节
以实现高精度的。PET-NeuS[56]对空间点使用了三平面位置编码，提高了 MLP 的
表达能力以实现高质量重建，PermutoSDF[57]通过将置换不变性和符号距离函数
（SDF）结合，实现了对复杂几何形状的高效、鲁棒的三维表示和重建。Neural
Warp[58]通过在渲染过程中引入多试图的光度不变性，在提高渲染质量的同时改
善了重建完整度。
2024 年，PoRF[59]通过两阶段的相机位姿优化方法，持续优化相机位姿，实
5
浙江省硕士学位论文
现了不准确位姿输入下的高性能重建。
3）NeRF 训练速度改进
上述基于 NeRF 的方法能渲染出高质量的图片和高精度的三维模型，但是，
这些方法仍然需要数小时乃至数天的时间进行训练。冗长的训练时间已然成为
NeRF 大规模应用的最大阻碍之一。因此，许多工作在加速 NeRF 训练方面进行
了研究。
2021 年，KiloNeRF[60]将 NeRF 分解为多个小型 MLP，并使用知识蒸馏的方
法进行训练以加速 NeRF 的训练。2022 年，DVGO[61]使用两个显式的可学习网
格对场景进行建模，并结合由粗到细的两阶段训练方法将 NeRF 的训练速度提升
至 10 分钟级别、Plenoxels[62]同样使用稀疏网格对场景进行建模，并使用球协函
数对光照进行建模，在不降低训练时间的情况下大幅减少训练时间。这些工作让
NeRF 的训练时间减少到 1 小时以内。
2022 年 10 月，随着 Instant-NGP[63]的出现，NeRF 的训练时长进入了分钟时
代。其使用多分辨率哈希网格对空间中三维点进行编码，并结合哈希表以及跳
跃采样让空间采样点快速接近物体表面。Instant-NGP 提供了一个高效的 cuda 实
现，能够在 5 分钟之内完成 NeRF 训练的同时，保持了很高的渲染质量。 总结一下上述文字'
input_text="${3:-${QWEN35_INPUT_TEXT:-${default_input_text}}}"
extra_args=("${@:4}")

bindings_dir="${QWEN35_BINDINGS_DIR:-${repo_root}/cpp/build/tensorrt_llm/nanobind}"
plugin_lib="${QWEN35_PLUGIN_LIB:-${repo_root}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so}"

max_input_length="${QWEN35_MAX_INPUT_LEN:-1536}"
max_output_len="${QWEN35_MAX_OUTPUT_LEN:-512}"
kv_cache_fraction="${QWEN35_KV_CACHE_FREE_GPU_MEMORY_FRACTION:-0.2}"
use_chat_template="${QWEN35_USE_CHAT_TEMPLATE:-1}"

if [[ ! -f "${engine_dir}/config.json" ]]; then
    printf 'TensorRT-LLM engine config does not exist: %s/config.json\n' "${engine_dir}" >&2
    exit 1
fi
if [[ ! -d "${model_dir}" ]]; then
    printf 'Qwen3.5 tokenizer directory does not exist: %s\n' "${model_dir}" >&2
    exit 1
fi
if ! compgen -G "${bindings_dir}/bindings*.so" >/dev/null; then
    printf 'TensorRT-LLM source bindings do not exist in: %s\n' "${bindings_dir}" >&2
    printf 'Build them with: cmake --build cpp/build --target bindings --parallel 8\n' >&2
    exit 1
fi
if [[ ! -f "${plugin_lib}" ]]; then
    printf 'TensorRT-LLM source plugin does not exist: %s\n' "${plugin_lib}" >&2
    printf 'Build it with: cmake --build cpp/build --target bindings --parallel 8\n' >&2
    exit 1
fi
if [[ "${use_chat_template}" != "0" && "${use_chat_template}" != "1" ]]; then
    printf 'QWEN35_USE_CHAT_TEMPLATE must be 0 or 1, got: %s\n' "${use_chat_template}" >&2
    exit 1
fi

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
torch_lib_dir="$("${python_bin}" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${torch_lib_dir}:${repo_root}/cpp/build/tensorrt_llm/thop:${repo_root}/cpp/build/tensorrt_llm/runtime/utils:${repo_root}/cpp/build/tensorrt_llm:${repo_root}/cpp/build/tensorrt_llm/kernels/decoderMaskedMultiheadAttention${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export TRT_LLM_NO_LIB_INIT=1
export QWEN35_BINDINGS_DIR="${bindings_dir}"
export QWEN35_PLUGIN_LIB="${plugin_lib}"
export QWEN35_MODEL_DIR="${model_dir}"
export QWEN35_USE_CHAT_TEMPLATE="${use_chat_template}"
cd "${repo_root}"

"${python_bin}" -c '
import ctypes
import importlib.util
import os
import runpy
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.environ["QWEN35_BINDINGS_DIR"])
import bindings

package_spec = importlib.util.find_spec("tensorrt_llm")
if package_spec is None or package_spec.loader is None:
    raise ImportError("Unable to find the tensorrt_llm Python package")
package = importlib.util.module_from_spec(package_spec)
package.bindings = bindings
sys.modules["tensorrt_llm"] = package
sys.modules["tensorrt_llm.bindings"] = bindings
for module_name, module in list(sys.modules.items()):
    if module_name.startswith("bindings."):
        sys.modules[f"tensorrt_llm.{module_name}"] = module
package_spec.loader.exec_module(package)

plugin = ctypes.CDLL(os.environ["QWEN35_PLUGIN_LIB"], mode=ctypes.RTLD_GLOBAL)
plugin.initTrtLlmPlugins.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
plugin.initTrtLlmPlugins.restype = ctypes.c_bool
if not plugin.initTrtLlmPlugins(None, b"tensorrt_llm"):
    raise RuntimeError("Failed to initialize TensorRT-LLM source plugins")

if os.environ["QWEN35_USE_CHAT_TEMPLATE"] == "1":
    input_text_index = sys.argv.index("--input_text") + 1
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ["QWEN35_MODEL_DIR"],
        trust_remote_code=True,
    )
    sys.argv[input_text_index] = tokenizer.apply_chat_template(
        [{"role": "user", "content": sys.argv[input_text_index]}],
        tokenize=False,
        add_generation_prompt=True,
    )

sys.path.insert(0, os.path.join(os.getcwd(), "examples"))
runpy.run_path("examples/run.py", run_name="__main__")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
' \
    --engine_dir "${engine_dir}" \
    --tokenizer_dir "${model_dir}" \
    --input_text "${input_text}" \
    --max_input_length "${max_input_length}" \
    --max_output_len "${max_output_len}" \
    --num_beams 1 \
    --top_k 1 \
    --top_p 0.0 \
    --temperature 1.0 \
    --no-kv_cache_enable_block_reuse \
    --kv_cache_free_gpu_memory_fraction "${kv_cache_fraction}" \
    "${extra_args[@]}"
