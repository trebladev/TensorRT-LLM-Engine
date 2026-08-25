/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/runtime/gptJsonConfig.h"

#include "tensorrt_llm/common/tllmException.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace tensorrt_llm::runtime
{
namespace
{

nlohmann::json makeAttentionLinearHybridConfig(
    SizeType32 tensorParallelism, SizeType32 numKeyHeads = 16, SizeType32 numValueHeads = 16)
{
    return {
        {"version", "1.0.0"},
        {"pretrained_config",
            {
                {"architecture", "Qwen35ForCausalLM"},
                {"dtype", "bfloat16"},
                {"vocab_size", 248320},
                {"hidden_size", 2048},
                {"num_hidden_layers", 2},
                {"num_attention_heads", 8},
                {"num_key_value_heads", 2},
                {"intermediate_size", 6144},
                {"head_size", 256},
                {"layer_types", {"linear", "attention"}},
                {"linear_conv_kernel_dim", 4},
                {"linear_key_head_dim", 128},
                {"linear_value_head_dim", 128},
                {"linear_num_key_heads", numKeyHeads},
                {"linear_num_value_heads", numValueHeads},
                {"state_dtype", "float32"},
                {"mapping",
                    {
                        {"world_size", tensorParallelism},
                        {"tp_size", tensorParallelism},
                        {"pp_size", 1},
                        {"cp_size", 1},
                    }},
                {"quantization",
                    {
                        {"quant_algo", nullptr},
                        {"kv_cache_quant_algo", nullptr},
                    }},
            }},
        {"build_config",
            {
                {"max_batch_size", 4},
                {"max_beam_width", 1},
                {"max_input_len", 128},
                {"max_seq_len", 256},
                {"max_num_tokens", 512},
                {"kv_cache_type", "paged"},
                {"lora_config",
                    {
                        {"max_lora_rank", 0},
                        {"lora_target_modules", nlohmann::json::array()},
                    }},
                {"plugin_config",
                    {
                        {"gpt_attention_plugin", "bfloat16"},
                        {"lora_plugin", nullptr},
                        {"mamba_conv1d_plugin", "bfloat16"},
                        {"remove_input_padding", true},
                        {"paged_kv_cache", true},
                        {"paged_state", true},
                        {"tokens_per_block", 32},
                        {"context_fmha", true},
                        {"use_paged_context_fmha", true},
                    }},
            }},
    };
}

TEST(GptJsonConfigTest, UsesTensorParallelLocalLinearAttentionHeads)
{
    auto const tp1Config = GptJsonConfig::parse(makeAttentionLinearHybridConfig(/*tensorParallelism=*/1).dump());
    auto const tp1LinearConfig = tp1Config.getModelConfig().getLinearAttentionConfig();
    ASSERT_TRUE(tp1LinearConfig.has_value());
    EXPECT_EQ(tp1LinearConfig->numKeyHeads, 16);
    EXPECT_EQ(tp1LinearConfig->numValueHeads, 16);
    EXPECT_EQ(tp1LinearConfig->getGatedDeltaStateBytes(), 1'048'576);
    EXPECT_EQ(tp1LinearConfig->getConvStateBytes(), 36'864);
    EXPECT_EQ(tp1LinearConfig->getStateSlotBytes(), 1'085'440);

    auto const tp2Config = GptJsonConfig::parse(makeAttentionLinearHybridConfig(/*tensorParallelism=*/2).dump());
    auto const tp2LinearConfig = tp2Config.getModelConfig().getLinearAttentionConfig();
    ASSERT_TRUE(tp2LinearConfig.has_value());
    EXPECT_EQ(tp2LinearConfig->numKeyHeads, 8);
    EXPECT_EQ(tp2LinearConfig->numValueHeads, 8);
    EXPECT_EQ(tp2LinearConfig->getGatedDeltaStateBytes(), 524'288);
    EXPECT_EQ(tp2LinearConfig->getConvStateBytes(), 18'432);
    EXPECT_EQ(tp2LinearConfig->getStateSlotBytes(), 542'720);
}

TEST(GptJsonConfigTest, RejectsLinearAttentionHeadsNotDivisibleByTensorParallelism)
{
    EXPECT_THROW(
        GptJsonConfig::parse(makeAttentionLinearHybridConfig(/*tensorParallelism=*/2, /*numKeyHeads=*/15).dump()),
        common::TllmException);
    EXPECT_THROW(GptJsonConfig::parse(makeAttentionLinearHybridConfig(/*tensorParallelism=*/2, /*numKeyHeads=*/16,
                     /*numValueHeads=*/15)
                                          .dump()),
        common::TllmException);
}

} // namespace
} // namespace tensorrt_llm::runtime
