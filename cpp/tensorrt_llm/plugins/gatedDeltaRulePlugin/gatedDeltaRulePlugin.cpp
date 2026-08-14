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

#include "gatedDeltaRulePlugin.h"

#include "tensorrt_llm/common/assert.h"

#include <cstring>
#include <memory>

using namespace nvinfer1;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::GatedDeltaRulePlugin;
using tensorrt_llm::plugins::GatedDeltaRulePluginCreator;

namespace
{

char const* GATED_DELTA_RULE_PLUGIN_VERSION{"1"};
char const* GATED_DELTA_RULE_PLUGIN_NAME{"GatedDeltaRule"};

} // namespace

PluginFieldCollection GatedDeltaRulePluginCreator::mFC{};
std::vector<PluginField> GatedDeltaRulePluginCreator::mPluginAttributes;

GatedDeltaRulePlugin::GatedDeltaRulePlugin(int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim,
    int32_t chunkSize, DataType type, DataType stateType, bool removeInputPadding, bool pagedState, bool useQkL2norm)
    : mNumQHeads(numQHeads)
    , mNumVHeads(numVHeads)
    , mHeadKDim(headKDim)
    , mHeadVDim(headVDim)
    , mChunkSize(chunkSize)
    , mType(type)
    , mStateType(stateType)
    , mRemoveInputPadding(removeInputPadding)
    , mPagedState(pagedState)
    , mUseQkL2norm(useQkL2norm)
{
    validateConfig();
    initFieldsToSerialize();
}

void GatedDeltaRulePlugin::validateConfig() const
{
    TLLM_CHECK_WITH_INFO(mNumQHeads > 0, "num_q_heads must be positive");
    TLLM_CHECK_WITH_INFO(mNumVHeads > 0, "num_v_heads must be positive");
    TLLM_CHECK_WITH_INFO(mNumVHeads % mNumQHeads == 0, "num_v_heads must be divisible by num_q_heads");
    TLLM_CHECK_WITH_INFO(mHeadKDim > 0, "head_k_dim must be positive");
    TLLM_CHECK_WITH_INFO(mHeadVDim > 0, "head_v_dim must be positive");
    TLLM_CHECK_WITH_INFO(mChunkSize > 0, "chunk_size must be positive");
    TLLM_CHECK_WITH_INFO(mType == DataType::kHALF || mType == DataType::kBF16,
        "GatedDeltaRulePlugin only supports FP16 and BF16 activations");
    TLLM_CHECK_WITH_INFO(mStateType == DataType::kFLOAT, "GatedDeltaRulePlugin state must use FP32");
}

void GatedDeltaRulePlugin::initFieldsToSerialize()
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("num_q_heads", &mNumQHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("num_v_heads", &mNumVHeads, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_k_dim", &mHeadKDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("head_v_dim", &mHeadVDim, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("chunk_size", &mChunkSize, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("type_id", &mType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("state_type_id", &mStateType, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("remove_input_padding", &mRemoveInputPadding, PluginFieldType::kINT8, 1);
    mDataToSerialize.emplace_back("paged_state", &mPagedState, PluginFieldType::kINT8, 1);
    mDataToSerialize.emplace_back("use_qk_l2norm", &mUseQkL2norm, PluginFieldType::kINT8, 1);
    mFieldsToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFieldsToSerialize.fields = mDataToSerialize.data();
}

IPluginCapability* GatedDeltaRulePlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    switch (type)
    {
    case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
    case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
    case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
    }
    return nullptr;
}

IPluginV3* GatedDeltaRulePlugin::clone() noexcept
{
    try
    {
        auto plugin = std::make_unique<GatedDeltaRulePlugin>(mNumQHeads, mNumVHeads, mHeadKDim, mHeadVDim, mChunkSize,
            mType, mStateType, mRemoveInputPadding, mPagedState, mUseQkL2norm);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

char const* GatedDeltaRulePlugin::getPluginName() const noexcept
{
    return GATED_DELTA_RULE_PLUGIN_NAME;
}

char const* GatedDeltaRulePlugin::getPluginVersion() const noexcept
{
    return GATED_DELTA_RULE_PLUGIN_VERSION;
}

int32_t GatedDeltaRulePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == static_cast<int32_t>(InputIdx::kNumInputs));
        TLLM_CHECK(nbOutputs == getNbOutputs());
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

int32_t GatedDeltaRulePlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == static_cast<int32_t>(InputIdx::kNumInputs));
        TLLM_CHECK(nbOutputs == getNbOutputs());
        TLLM_CHECK(inputTypes[static_cast<int32_t>(InputIdx::kValue)] == mType);
        outputTypes[0] = mType;
        if (!mPagedState)
        {
            TLLM_CHECK(inputTypes[static_cast<int32_t>(InputIdx::kState)] == mStateType);
            outputTypes[1] = mStateType;
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

int32_t GatedDeltaRulePlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        TLLM_CHECK(nbInputs == static_cast<int32_t>(InputIdx::kNumInputs));
        TLLM_CHECK(nbShapeInputs == 0);
        TLLM_CHECK(nbOutputs == getNbOutputs());
        outputs[0] = inputs[static_cast<int32_t>(InputIdx::kValue)];
        if (!mPagedState)
        {
            outputs[1] = inputs[static_cast<int32_t>(InputIdx::kState)];
        }
    }
    catch (std::exception const& e)
    {
        caughtError(e);
        return -1;
    }
    return 0;
}

bool GatedDeltaRulePlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (nbInputs != static_cast<int32_t>(InputIdx::kNumInputs) || nbOutputs != getNbOutputs() || pos < 0
        || pos >= nbInputs + nbOutputs)
    {
        return false;
    }

    auto const type = inOut[pos].desc.type;
    auto const format = inOut[pos].desc.format;
    auto const isLinear = format == TensorFormat::kLINEAR;
    auto const stateIdx = static_cast<int32_t>(InputIdx::kState);

    if (pos == static_cast<int32_t>(InputIdx::kQuery) || pos == static_cast<int32_t>(InputIdx::kKey)
        || pos == static_cast<int32_t>(InputIdx::kValue) || pos == nbInputs)
    {
        return type == mType && isLinear;
    }
    if (pos == static_cast<int32_t>(InputIdx::kLogDecay) || pos == static_cast<int32_t>(InputIdx::kBeta))
    {
        return type == DataType::kFLOAT && isLinear;
    }
    if (pos == stateIdx)
    {
        return mPagedState ? type == DataType::kINT64 : type == mStateType && isLinear;
    }
    if (pos == static_cast<int32_t>(InputIdx::kHostRequestTypes) || pos == static_cast<int32_t>(InputIdx::kCuSeqLens)
        || pos == static_cast<int32_t>(InputIdx::kStateSlotMapping))
    {
        return type == DataType::kINT32 && isLinear;
    }
    if (pos == static_cast<int32_t>(InputIdx::kHostHasInitialState))
    {
        return type == DataType::kINT8 && isLinear;
    }
    if (!mPagedState && pos == nbInputs + 1)
    {
        return type == mStateType && isLinear;
    }
    return false;
}

int32_t GatedDeltaRulePlugin::getNbOutputs() const noexcept
{
    return mPagedState ? 1 : 2;
}

size_t GatedDeltaRulePlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t GatedDeltaRulePlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    return 0;
}

int32_t GatedDeltaRulePlugin::getNbTactics() noexcept
{
    return 0;
}

char const* GatedDeltaRulePlugin::getTimingCacheID() noexcept
{
    return nullptr;
}

int32_t GatedDeltaRulePlugin::getFormatCombinationLimit() noexcept
{
    return 1;
}

char const* GatedDeltaRulePlugin::getMetadataString() noexcept
{
    return nullptr;
}

int32_t GatedDeltaRulePlugin::setTactic(int32_t tactic) noexcept
{
    return 0;
}

int32_t GatedDeltaRulePlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

int32_t GatedDeltaRulePlugin::enqueuePrefill(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    TLLM_LOG_ERROR("GatedDeltaRulePlugin prefill kernel is not implemented");
    return -1;
}

int32_t GatedDeltaRulePlugin::enqueueDecode(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    TLLM_LOG_ERROR("GatedDeltaRulePlugin decode kernel is not implemented");
    return -1;
}

int32_t GatedDeltaRulePlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }

    auto const requestTypesIdx = static_cast<int32_t>(InputIdx::kHostRequestTypes);
    auto const numRequests = inputDesc[requestTypesIdx].dims.d[0];
    if (numRequests <= 0)
    {
        TLLM_LOG_ERROR("GatedDeltaRulePlugin requires at least one request");
        return -1;
    }

    auto const* requestTypes = static_cast<int32_t const*>(inputs[requestTypesIdx]);
    auto const firstRequestType = static_cast<RequestType>(requestTypes[0]);
    if (firstRequestType != RequestType::kContext && firstRequestType != RequestType::kGeneration)
    {
        TLLM_LOG_ERROR("GatedDeltaRulePlugin received an invalid request type: %d", requestTypes[0]);
        return -1;
    }

    for (int32_t requestIdx = 1; requestIdx < numRequests; ++requestIdx)
    {
        if (requestTypes[requestIdx] != requestTypes[0])
        {
            TLLM_LOG_ERROR("GatedDeltaRulePlugin does not support mixed prefill and decode batches");
            return -1;
        }
    }

    if (firstRequestType == RequestType::kContext)
    {
        return enqueuePrefill(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    return enqueueDecode(inputDesc, outputDesc, inputs, outputs, workspace, stream);
}

IPluginV3* GatedDeltaRulePlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* GatedDeltaRulePlugin::getFieldsToSerialize() noexcept
{
    return &mFieldsToSerialize;
}

GatedDeltaRulePluginCreator::GatedDeltaRulePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("num_q_heads", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("num_v_heads", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("head_k_dim", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("head_v_dim", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("chunk_size", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("type_id", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("state_type_id", nullptr, PluginFieldType::kINT32, 1);
    mPluginAttributes.emplace_back("remove_input_padding", nullptr, PluginFieldType::kINT8, 1);
    mPluginAttributes.emplace_back("paged_state", nullptr, PluginFieldType::kINT8, 1);
    mPluginAttributes.emplace_back("use_qk_l2norm", nullptr, PluginFieldType::kINT8, 1);
    mFC.nbFields = static_cast<int32_t>(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

char const* GatedDeltaRulePluginCreator::getPluginName() const noexcept
{
    return GATED_DELTA_RULE_PLUGIN_NAME;
}

char const* GatedDeltaRulePluginCreator::getPluginVersion() const noexcept
{
    return GATED_DELTA_RULE_PLUGIN_VERSION;
}

PluginFieldCollection const* GatedDeltaRulePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* GatedDeltaRulePluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        TLLM_CHECK(fc != nullptr);
        int32_t numQHeads{};
        int32_t numVHeads{};
        int32_t headKDim{};
        int32_t headVDim{};
        int32_t chunkSize{};
        DataType type{};
        DataType stateType{};
        bool removeInputPadding{};
        bool pagedState{};
        bool useQkL2norm{};
        bool hasNumQHeads{false};
        bool hasNumVHeads{false};
        bool hasHeadKDim{false};
        bool hasHeadVDim{false};
        bool hasChunkSize{false};
        bool hasType{false};
        bool hasStateType{false};
        bool hasRemoveInputPadding{false};
        bool hasPagedState{false};
        bool hasUseQkL2norm{false};

        for (int32_t fieldIdx = 0; fieldIdx < fc->nbFields; ++fieldIdx)
        {
            auto const& field = fc->fields[fieldIdx];
            TLLM_CHECK(field.name != nullptr);
            if (std::strcmp(field.name, "num_q_heads") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                numQHeads = *static_cast<int32_t const*>(field.data);
                hasNumQHeads = true;
            }
            else if (std::strcmp(field.name, "num_v_heads") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                numVHeads = *static_cast<int32_t const*>(field.data);
                hasNumVHeads = true;
            }
            else if (std::strcmp(field.name, "head_k_dim") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                headKDim = *static_cast<int32_t const*>(field.data);
                hasHeadKDim = true;
            }
            else if (std::strcmp(field.name, "head_v_dim") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                headVDim = *static_cast<int32_t const*>(field.data);
                hasHeadVDim = true;
            }
            else if (std::strcmp(field.name, "chunk_size") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                chunkSize = *static_cast<int32_t const*>(field.data);
                hasChunkSize = true;
            }
            else if (std::strcmp(field.name, "type_id") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                type = static_cast<DataType>(*static_cast<int32_t const*>(field.data));
                hasType = true;
            }
            else if (std::strcmp(field.name, "state_type_id") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT32 && field.length == 1);
                stateType = static_cast<DataType>(*static_cast<int32_t const*>(field.data));
                hasStateType = true;
            }
            else if (std::strcmp(field.name, "remove_input_padding") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT8 && field.length == 1);
                removeInputPadding = *static_cast<bool const*>(field.data);
                hasRemoveInputPadding = true;
            }
            else if (std::strcmp(field.name, "paged_state") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT8 && field.length == 1);
                pagedState = *static_cast<bool const*>(field.data);
                hasPagedState = true;
            }
            else if (std::strcmp(field.name, "use_qk_l2norm") == 0)
            {
                TLLM_CHECK(field.type == PluginFieldType::kINT8 && field.length == 1);
                useQkL2norm = *static_cast<bool const*>(field.data);
                hasUseQkL2norm = true;
            }
            else
            {
                TLLM_LOG_WARNING("%s: got an unexpected attribute: %s", __PRETTY_FUNCTION__, field.name);
            }
        }

        TLLM_CHECK(hasNumQHeads && hasNumVHeads && hasHeadKDim && hasHeadVDim && hasChunkSize && hasType && hasStateType
            && hasRemoveInputPadding && hasPagedState && hasUseQkL2norm);
        auto plugin = std::make_unique<GatedDeltaRulePlugin>(numQHeads, numVHeads, headKDim, headVDim, chunkSize, type,
            stateType, removeInputPadding, pagedState, useQkL2norm);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin.release();
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
