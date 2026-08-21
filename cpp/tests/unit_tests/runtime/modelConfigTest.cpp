/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/modelConfig.h"

#include <gtest/gtest.h>

namespace tensorrt_llm::runtime
{
namespace
{

ModelConfig makeModelConfig(SizeType32 nbLayers, SizeType32 nbAttentionLayers)
{
    return ModelConfig{/*vocabSize=*/32, nbLayers, nbAttentionLayers, /*nbRnnLayers=*/0, /*nbHeads=*/4,
        /*hiddenSize=*/16, nvinfer1::DataType::kFLOAT};
}

TEST(ModelConfigTest, IdentifiesFullAttentionModel)
{
    auto modelConfig = makeModelConfig(/*nbLayers=*/4, /*nbAttentionLayers=*/4);
    modelConfig.setLayerTypes(std::vector(4, ModelConfig::LayerType::kATTENTION));

    EXPECT_TRUE(modelConfig.hasAttentionLayers());
    EXPECT_FALSE(modelConfig.hasLinearAttentionLayers());
    EXPECT_FALSE(modelConfig.isAttentionLinearHybrid());
    EXPECT_TRUE(modelConfig.isFullAttentionModel());
    EXPECT_EQ(modelConfig.getNbLinearLayers(), 0);
}

TEST(ModelConfigTest, IdentifiesAttentionLinearHybrid)
{
    auto modelConfig = makeModelConfig(/*nbLayers=*/4, /*nbAttentionLayers=*/2);
    modelConfig.setLayerTypes({ModelConfig::LayerType::kLINEAR, ModelConfig::LayerType::kATTENTION,
        ModelConfig::LayerType::kLINEAR, ModelConfig::LayerType::kATTENTION});

    EXPECT_TRUE(modelConfig.hasAttentionLayers());
    EXPECT_TRUE(modelConfig.hasLinearAttentionLayers());
    EXPECT_TRUE(modelConfig.isAttentionLinearHybrid());
    EXPECT_FALSE(modelConfig.isFullAttentionModel());
    EXPECT_EQ(modelConfig.getNbAttentionLayers(), 2);
    EXPECT_EQ(modelConfig.getNbLinearLayers(), 2);
}

TEST(ModelConfigTest, DoesNotClassifyLinearOnlyModelAsHybrid)
{
    auto modelConfig = makeModelConfig(/*nbLayers=*/4, /*nbAttentionLayers=*/0);
    modelConfig.setLayerTypes(std::vector(4, ModelConfig::LayerType::kLINEAR));

    EXPECT_FALSE(modelConfig.hasAttentionLayers());
    EXPECT_TRUE(modelConfig.hasLinearAttentionLayers());
    EXPECT_FALSE(modelConfig.isAttentionLinearHybrid());
    EXPECT_FALSE(modelConfig.isFullAttentionModel());
    EXPECT_EQ(modelConfig.getNbLinearLayers(), 4);
}

TEST(ModelConfigTest, KeepsRnnClassificationDistinct)
{
    auto modelConfig = makeModelConfig(/*nbLayers=*/4, /*nbAttentionLayers=*/4);
    modelConfig.setModelVariant(ModelConfig::ModelVariant::kMamba);
    modelConfig.setLayerTypes(std::vector(4, ModelConfig::LayerType::kATTENTION));

    EXPECT_TRUE(modelConfig.isRnnBased());
    EXPECT_FALSE(modelConfig.isFullAttentionModel());

    modelConfig.setLayerTypes({ModelConfig::LayerType::kLINEAR, ModelConfig::LayerType::kATTENTION,
        ModelConfig::LayerType::kLINEAR, ModelConfig::LayerType::kATTENTION});

    EXPECT_FALSE(modelConfig.isAttentionLinearHybrid());
}

TEST(ModelConfigTest, CalculatesLinearAttentionStateBytes)
{
    ModelConfig::LinearAttentionConfig config{};
    config.convKernel = 4;
    config.numKeyHeads = 16;
    config.numValueHeads = 32;
    config.keyHeadDim = 128;
    config.valueHeadDim = 128;
    config.stateDtype = nvinfer1::DataType::kFLOAT;
    config.convDtype = nvinfer1::DataType::kBF16;

    EXPECT_EQ(config.getGatedDeltaStateBytes(), 2'097'152);
    EXPECT_EQ(config.getConvStateBytes(), 49'152);
    EXPECT_EQ(config.getStateSlotBytes(), 2'146'304);

    auto modelConfig = makeModelConfig(/*nbLayers=*/4, /*nbAttentionLayers=*/1);
    modelConfig.setLinearAttentionConfig(config);

    ASSERT_TRUE(modelConfig.hasLinearAttentionConfig());
    EXPECT_EQ(modelConfig.getLinearAttentionConfig()->getStateSlotBytes(), 2'146'304);
}

} // namespace
} // namespace tensorrt_llm::runtime
