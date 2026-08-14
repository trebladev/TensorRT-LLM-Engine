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

#pragma once

#include "tensorrt_llm/plugins/common/plugin.h"

#include <cstdint>
#include <vector>

namespace tensorrt_llm::plugins
{

// Inputs:
//   0. query: [B, S, Hq, K] or [1, T, Hq, K] in packed mode.
//   1. key: [B, S, Hq, K] or [1, T, Hq, K] in packed mode.
//   2. value: [B, S, Hv, V] or [1, T, Hv, V] in packed mode.
//   3. log_decay: [B, S, Hv] or [1, T, Hv], float32.
//   4. beta: [B, S, Hv] or [1, T, Hv], float32.
//   5. state: [N, Hv, V, K], or host [1] containing a state-pool pointer when paged_state is enabled.
//   6. host_request_types: [N], int32 on the host. 0 is context and 1 is generation.
//   7. cu_seqlens: [N + 1], int32.
//   8. state_slot_mapping: [N], int32.
//   9. host_has_initial_state: [N], int8 on the host.
// Outputs:
//   0. output: same shape and type as value.
//   1. final_state: [N, Hv, V, K], float32. In paged mode it is valid for context requests and ignored for
//      generation requests; the state pool is updated in place in both phases.
class GatedDeltaRulePlugin : public BasePluginV3
{
public:
    GatedDeltaRulePlugin() = delete;
    GatedDeltaRulePlugin(int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim, int32_t chunkSize,
        nvinfer1::DataType type, nvinfer1::DataType stateType, bool removeInputPadding, bool pagedState,
        bool useQkL2norm);
    GatedDeltaRulePlugin(GatedDeltaRulePlugin const& plugin) = default;

    // IPluginV3 methods
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;

    // IPluginV3OneBuild methods
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs, nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs, nvinfer1::DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t getNbOutputs() const noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    int32_t getNbTactics() noexcept override;
    char const* getTimingCacheID() noexcept override;
    int32_t getFormatCombinationLimit() noexcept override;
    char const* getMetadataString() noexcept override;

    // IPluginV3OneRuntime methods
    int32_t setTactic(int32_t tactic) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    enum class InputIdx : int32_t
    {
        kQuery = 0,
        kKey,
        kValue,
        kLogDecay,
        kBeta,
        kState,
        kHostRequestTypes,
        kCuSeqLens,
        kStateSlotMapping,
        kHostHasInitialState,
        kNumInputs
    };

    enum class RequestType : int32_t
    {
        kContext = 0,
        kGeneration = 1
    };

    void initFieldsToSerialize();
    void validateConfig() const;
    int32_t enqueuePrefill(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;
    int32_t enqueueDecode(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept;

    int32_t mNumQHeads;
    int32_t mNumVHeads;
    int32_t mHeadKDim;
    int32_t mHeadVDim;
    int32_t mChunkSize;
    nvinfer1::DataType mType;
    nvinfer1::DataType mStateType;
    bool mRemoveInputPadding;
    bool mPagedState;
    bool mUseQkL2norm;

    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFieldsToSerialize{};
};

class GatedDeltaRulePluginCreator : public BaseCreatorV3
{
public:
    GatedDeltaRulePluginCreator();

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

} // namespace tensorrt_llm::plugins
