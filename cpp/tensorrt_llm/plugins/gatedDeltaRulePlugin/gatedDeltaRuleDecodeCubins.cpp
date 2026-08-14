/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gatedDeltaRuleDecodeCubins.h"

#include "tensorrt_llm/common/config.h"

#include <array>

namespace tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels
{

#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
extern unsigned char const gated_delta_rule_decode_bf16_h16_hv16_k128_v128_sm89_cubin[];
extern unsigned int const gated_delta_rule_decode_bf16_h16_hv16_k128_v128_sm89_cubin_len;
extern unsigned char const gated_delta_rule_decode_bf16_h16_hv32_k128_v128_sm89_cubin[];
extern unsigned int const gated_delta_rule_decode_bf16_h16_hv32_k128_v128_sm89_cubin_len;
extern unsigned char const gated_delta_rule_decode_bf16_h16_hv48_k128_v128_sm89_cubin[];
extern unsigned int const gated_delta_rule_decode_bf16_h16_hv48_k128_v128_sm89_cubin_len;
#endif

} // namespace tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels

namespace tensorrt_llm::plugins
{
namespace
{

auto const& getCubins()
{
#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
    constexpr int32_t kSm = 89;
    constexpr int32_t kNumQHeads = 16;
    constexpr int32_t kHeadKDim = 128;
    constexpr int32_t kHeadVDim = 128;
    constexpr int32_t kSharedMemoryBytes = 16;
    constexpr char const* kKernelName = "fused_recurrent_gated_delta_rule_update_fwd_kernel";
    static std::array<GatedDeltaRuleDecodeCubin, 3> const cubins{{
        {kSm, kNumQHeads, 16, kHeadKDim, kHeadVDim,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv16_k128_v128_sm89_cubin,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv16_k128_v128_sm89_cubin_len,
            kKernelName, kSharedMemoryBytes},
        {kSm, kNumQHeads, 32, kHeadKDim, kHeadVDim,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv32_k128_v128_sm89_cubin,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv32_k128_v128_sm89_cubin_len,
            kKernelName, kSharedMemoryBytes},
        {kSm, kNumQHeads, 48, kHeadKDim, kHeadVDim,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv48_k128_v128_sm89_cubin,
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::gated_delta_rule_decode_bf16_h16_hv48_k128_v128_sm89_cubin_len,
            kKernelName, kSharedMemoryBytes},
    }};
    return cubins;
#else
    static std::array<GatedDeltaRuleDecodeCubin, 0> const cubins{};
    return cubins;
#endif
}

} // namespace

GatedDeltaRuleDecodeCubin const* findGatedDeltaRuleDecodeCubin(
    int32_t sm, int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim)
{
    for (auto const& cubin : getCubins())
    {
        if (cubin.sm == sm && cubin.numQHeads == numQHeads && cubin.numVHeads == numVHeads && cubin.headKDim == headKDim
            && cubin.headVDim == headVDim)
        {
            return &cubin;
        }
    }
    return nullptr;
}

} // namespace tensorrt_llm::plugins
