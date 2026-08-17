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

#include "gatedDeltaRulePrefillCubins.h"

#include "tensorrt_llm/common/config.h"

#include <array>

namespace tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels
{

#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
#define DECLARE_CUBIN(name)                                                                                            \
    extern unsigned char const name##_cubin[];                                                                         \
    extern unsigned int const name##_cubin_len

DECLARE_CUBIN(gated_delta_rule_prefill_l2norm_bf16_h16_k128_v128_sm89);
DECLARE_CUBIN(gated_delta_rule_prefill_init_chunks_bf16_h16_k128_v128_sm89);
DECLARE_CUBIN(gated_delta_rule_prefill_prepare_chunks_bf16_h16_k128_v128_sm89);
#define DECLARE_HV_CUBINS(hv)                                                                                          \
    DECLARE_CUBIN(gated_delta_rule_prefill_zero_state_bf16_h16_hv##hv##_k128_v128_sm89);                               \
    DECLARE_CUBIN(gated_delta_rule_prefill_gather_state_bf16_h16_hv##hv##_k128_v128_sm89);                             \
    DECLARE_CUBIN(gated_delta_rule_prefill_cumsum_bf16_h16_hv##hv##_k128_v128_sm89);                                   \
    DECLARE_CUBIN(gated_delta_rule_prefill_kkt_solve_bf16_h16_hv##hv##_k128_v128_sm89);                                \
    DECLARE_CUBIN(gated_delta_rule_prefill_recompute_bf16_h16_hv##hv##_k128_v128_sm89);                                \
    DECLARE_CUBIN(gated_delta_rule_prefill_state_bf16_h16_hv##hv##_k128_v128_sm89);                                    \
    DECLARE_CUBIN(gated_delta_rule_prefill_output_bf16_h16_hv##hv##_k128_v128_sm89)
DECLARE_HV_CUBINS(16);
DECLARE_HV_CUBINS(32);
DECLARE_HV_CUBINS(48);
#undef DECLARE_HV_CUBINS
#undef DECLARE_CUBIN
#endif

} // namespace tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels

namespace tensorrt_llm::plugins
{
namespace
{

#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
#define CUBIN_ENTRY(name, kind, hv, symbol, shared, warps)                                                             \
    {                                                                                                                  \
        89, GatedDeltaRulePrefillKernel::kind, hv, tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::name##_cubin,          \
            tensorrt_llm::TRTLLM_ABI_NAMESPACE::kernels::name##_cubin_len, symbol, shared, warps                       \
    }
#define HV_CUBIN_ENTRIES(hv)                                                                                           \
    CUBIN_ENTRY(gated_delta_rule_prefill_zero_state_bf16_h16_hv##hv##_k128_v128_sm89, kZeroState, hv,                  \
        "zero_missing_states_kernel", 0, 4),                                                                           \
        CUBIN_ENTRY(gated_delta_rule_prefill_gather_state_bf16_h16_hv##hv##_k128_v128_sm89, kGatherState, hv,          \
            "gather_states_kernel", 0, 4),                                                                             \
        CUBIN_ENTRY(gated_delta_rule_prefill_cumsum_bf16_h16_hv##hv##_k128_v128_sm89, kCumsum, hv,                     \
            "chunk_local_cumsum_scalar_kernel", 8, 8),                                                                 \
        CUBIN_ENTRY(gated_delta_rule_prefill_kkt_solve_bf16_h16_hv##hv##_k128_v128_sm89, kKktSolve, hv,                \
            "chunk_gated_delta_rule_fwd_kkt_solve_kernel", 6144, 4),                                                   \
        CUBIN_ENTRY(gated_delta_rule_prefill_recompute_bf16_h16_hv##hv##_k128_v128_sm89, kRecompute, hv,               \
            "recompute_w_u_fwd_kernel", 32768, 4),                                                                     \
        CUBIN_ENTRY(gated_delta_rule_prefill_state_bf16_h16_hv##hv##_k128_v128_sm89, kState, hv,                       \
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", 86536, 4),                                               \
        CUBIN_ENTRY(gated_delta_rule_prefill_output_bf16_h16_hv##hv##_k128_v128_sm89, kOutput, hv,                     \
            "chunk_fwd_kernel_o", 49152, 4)
#endif

auto const& getCubins()
{
#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
    static std::array<GatedDeltaRulePrefillCubin, 24> const cubins{{
        CUBIN_ENTRY(gated_delta_rule_prefill_l2norm_bf16_h16_k128_v128_sm89, kL2Norm, 0, "l2norm_fwd_kernel", 0, 8),
        CUBIN_ENTRY(gated_delta_rule_prefill_init_chunks_bf16_h16_k128_v128_sm89, kInitChunks, 0,
            "init_chunk_indices_kernel", 0, 1),
        CUBIN_ENTRY(gated_delta_rule_prefill_prepare_chunks_bf16_h16_k128_v128_sm89, kPrepareChunks, 0,
            "prepare_chunk_metadata_kernel", 4, 1),
        HV_CUBIN_ENTRIES(16),
        HV_CUBIN_ENTRIES(32),
        HV_CUBIN_ENTRIES(48),
    }};
    return cubins;
#else
    static std::array<GatedDeltaRulePrefillCubin, 0> const cubins{};
    return cubins;
#endif
}

#if !defined(_WIN32) && !defined(EXCLUDE_SM_89)
#undef HV_CUBIN_ENTRIES
#undef CUBIN_ENTRY
#endif

} // namespace

GatedDeltaRulePrefillCubin const* findGatedDeltaRulePrefillCubin(
    int32_t sm, GatedDeltaRulePrefillKernel kernel, int32_t numVHeads)
{
    for (auto const& cubin : getCubins())
    {
        if (cubin.sm == sm && cubin.kernel == kernel && cubin.numVHeads == numVHeads)
        {
            return &cubin;
        }
    }
    return nullptr;
}

} // namespace tensorrt_llm::plugins
