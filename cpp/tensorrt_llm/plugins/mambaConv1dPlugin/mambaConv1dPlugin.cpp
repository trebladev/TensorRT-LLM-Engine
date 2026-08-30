/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "mambaConv1dPlugin.h"
#include "tensorrt_llm/common/assert.h"
#include <algorithm>

using namespace nvinfer1;
using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;
using tensorrt_llm::plugins::MambaConv1dPluginCreator;
using tensorrt_llm::plugins::MambaConv1dPlugin;

static char const* MAMBA_CONV1D_PLUGIN_VERSION{"2"};
static char const* MAMBA_CONV1D_PLUGIN_NAME{"MambaConv1d"};

PluginFieldCollection MambaConv1dPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> MambaConv1dPluginCreator::mPluginAttributes;

MambaConv1dPlugin::MambaConv1dPlugin(int dim, int dconv, int preStride, int postStride, nvinfer1::DataType type,
    bool removePadding, bool pagedState, bool applySilu, int64_t stateSlotStrideBytes, int64_t stateChannelStrideBytes,
    int64_t stateHistoryStrideBytes, bool useInitialStateMask, bool useSeparateStateSlotMapping)
    : mDim(dim)
    , mDConv(dconv)
    , mPreStride(preStride)
    , mPostStride(postStride)
    , mType(type)
    , mRemovePadding(removePadding)
    , mPagedState(pagedState)
    , mApplySilu(applySilu)
    , mStateSlotStrideBytes(stateSlotStrideBytes)
    , mStateChannelStrideBytes(stateChannelStrideBytes)
    , mStateHistoryStrideBytes(stateHistoryStrideBytes)
    , mUseInitialStateMask(useInitialStateMask)
    , mUseSeparateStateSlotMapping(useSeparateStateSlotMapping)
{
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
    TLLM_CHECK_WITH_INFO(mStateSlotStrideBytes >= 0 && mStateChannelStrideBytes >= 0 && mStateHistoryStrideBytes >= 0,
        "State strides must be non-negative.");
    TLLM_CHECK_WITH_INFO(
        !mUseSeparateStateSlotMapping || mPagedState, "Separate state-slot mapping requires paged_state=true.");
}

// Parameterized constructor
MambaConv1dPlugin::MambaConv1dPlugin(void const* data, size_t length)
{
    char const *d = reinterpret_cast<char const*>(data), *a = d;
    read(d, mDim);
    read(d, mDConv);
    read(d, mPreStride);
    read(d, mPostStride);
    read(d, mType);
    read(d, mRemovePadding);
    read(d, mPagedState);
    read(d, mApplySilu);
    read(d, mStateSlotStrideBytes);
    read(d, mStateChannelStrideBytes);
    read(d, mStateHistoryStrideBytes);
    read(d, mUseInitialStateMask);
    if (d < a + length)
    {
        read(d, mUseSeparateStateSlotMapping);
    }
    TLLM_CHECK(d == a + length);
    TLLM_CHECK_WITH_INFO((mType == DataType::kBF16) || (mType == DataType::kFLOAT) || (mType == DataType::kHALF),
        "Only support float, half, and bfloat16.");
    TLLM_CHECK_WITH_INFO(mStateSlotStrideBytes >= 0 && mStateChannelStrideBytes >= 0 && mStateHistoryStrideBytes >= 0,
        "State strides must be non-negative.");
    TLLM_CHECK_WITH_INFO(
        !mUseSeparateStateSlotMapping || mPagedState, "Separate state-slot mapping requires paged_state=true.");
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* MambaConv1dPlugin::clone() const noexcept
{
    auto* plugin = new MambaConv1dPlugin(mDim, mDConv, mPreStride, mPostStride, mType, mRemovePadding, mPagedState,
        mApplySilu, mStateSlotStrideBytes, mStateChannelStrideBytes, mStateHistoryStrideBytes, mUseInitialStateMask,
        mUseSeparateStateSlotMapping);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

// Outputs
//     output_tensor: [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
//     state: [batch_size, dconv - 1, dim]
nvinfer1::DimsExprs MambaConv1dPlugin::getOutputDimensions(
    int outputIndex, nvinfer1::DimsExprs const* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    if (outputIndex == 0)
    {
        auto ret = inputs[getInputTensorIdx()];
        ret.d[mRemovePadding ? 1 : 2] = exprBuilder.constant(mDim);
        return ret;
    }
    return inputs[getConvStateIdx()];
}

bool MambaConv1dPlugin::supportsFormatCombination(
    int pos, nvinfer1::PluginTensorDesc const* inOut, int nbInputs, int nbOutputs) noexcept
{
    if (mUseInitialStateMask && pos == getHostHasInitialStateIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT8 || inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    if (pos == getHostRequestTypesIdx() || pos == getLastTokenIdsIdx()
        || (mRemovePadding && pos == getHostContextLengthIdx()) || (mPagedState && pos == getSourceSlotMappingIdx())
        || (mPagedState && mUseSeparateStateSlotMapping && pos == getTargetSlotMappingIdx()))
    {
        return inOut[pos].type == nvinfer1::DataType::kINT32;
    }
    else if (mPagedState && pos == getConvStateIdx())
    {
        return inOut[pos].type == nvinfer1::DataType::kINT64;
    }
    else
    {
        return (inOut[pos].type == mType) && (inOut[pos].format == TensorFormat::kLINEAR);
    }
}

void MambaConv1dPlugin::configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int nbInputs,
    nvinfer1::DynamicPluginTensorDesc const* out, int nbOutputs) noexcept
{
}

size_t MambaConv1dPlugin::getWorkspaceSize(nvinfer1::PluginTensorDesc const* inputs, int nbInputs,
    nvinfer1::PluginTensorDesc const* outputs, int nbOutputs) const noexcept
{
    return mUseInitialStateMask ? static_cast<size_t>(inputs[getHostRequestTypesIdx()].dims.d[0]) * sizeof(int8_t) : 0;
}

void MambaConv1dPlugin::setMambaConv1dParams(tensorrt_llm::kernels::MambaConv1dParamsBase& params, const size_t batch,
    const size_t dim, const size_t maxSeqLen, const size_t dconv, const size_t preStride, const size_t postStride,
    void const* inPtr, void const* stateInPtr, void* stateOutPtr, void const* convWeight, void const* convBias,
    void* outPtr, int const* lastTokenIds, int const* sourceStateSlotMapping, int const* targetStateSlotMapping,
    int8_t const* hasInitialState, int64_t stateSlotStride, int64_t stateChannelStride, int64_t stateHistoryStride,
    bool removePadding, bool applySilu)
{
    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.max_seqlen = maxSeqLen;
    params.dconv = dconv;
    params.pre_stride = preStride;
    params.post_stride = postStride;
    params.state_slot_stride = stateSlotStride;
    params.state_channel_stride = stateChannelStride;
    params.state_history_stride = stateHistoryStride;

    params.remove_padding = removePadding;
    params.apply_silu = applySilu;

    // Set the pointers and strides.
    params.in_ptr = const_cast<void*>(inPtr);
    params.state_in_ptr = const_cast<void*>(stateInPtr);
    params.state_out_ptr = stateOutPtr;
    params.weight_ptr = const_cast<void*>(convWeight);
    params.bias_ptr = const_cast<void*>(convBias);
    params.out_ptr = outPtr;
    params.last_token_ids_ptr = lastTokenIds;
    params.source_state_slot_mapping_ptr = sourceStateSlotMapping;
    params.target_state_slot_mapping_ptr = targetStateSlotMapping;
    params.has_initial_state_ptr = hasInitialState;
}

template <typename T>
int MambaConv1dPlugin::enqueueImpl(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream)
{
    // inputs
    //     0.  input_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //     1.  conv_state [batch_size, dconv - 1, dim] or host [1] containing only pointer for paged_state
    //     2.  weight [dim, 1, dconv]
    //     3.  bias [dim]
    //     4.  host_request_types [batch_size] int32. 0: context; 1: generation; 2: none.
    //     5.  last_token_ids [batch_size] int32
    //     6.  host_context_lengths [batch_size] int32, optional for remove_input_padding
    //     7.  source_state_slot_mapping [batch_size] int32, optional
    //     8.  target_state_slot_mapping [batch_size] int32, optional when separate mapping is enabled
    //     8/9.  host_has_initial_state [batch_size] int8 or int32, optional host input
    // outputs
    //     0. output_tensor [batch_size, seq_len, dim] or [num_tokens, dim] for remove_input_padding
    //     1. conv_state [batch_size, dconv - 1, dim]
    auto const batchSize = inputDesc[getHostRequestTypesIdx()].dims.d[0];
    int maxSeqLen;
    if (mRemovePadding)
    {
        int const* host_context_length = static_cast<int const*>(inputs[getHostContextLengthIdx()]);
        maxSeqLen = *std::max_element(host_context_length, host_context_length + batchSize);
    }
    else
    {
        maxSeqLen = inputDesc[getInputTensorIdx()].dims.d[1];
    }

    // only support context or generation, not for both of them
    RequestType const* reqTypes = static_cast<RequestType const*>(inputs[getHostRequestTypesIdx()]);

    MambaConv1dParamsBase mambaConv1dParams;

    int const* sourceSlotMapping = mPagedState ? static_cast<int const*>(inputs[getSourceSlotMappingIdx()]) : nullptr;
    int const* targetSlotMapping = mPagedState && mUseSeparateStateSlotMapping
        ? static_cast<int const*>(inputs[getTargetSlotMappingIdx()])
        : sourceSlotMapping;
    void* stateInPtr = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getConvStateIdx()]))
                                   : const_cast<void*>(inputs[getConvStateIdx()]);
    void* stateOutPtr
        = mPagedState ? *reinterpret_cast<void**>(const_cast<void*>(inputs[getConvStateIdx()])) : outputs[1];

    int8_t const* hasInitialState{};
    if (mUseInitialStateMask)
    {
        auto const maskIdx = getHostHasInitialStateIdx();
        auto const maskType = inputDesc[maskIdx].type;
        auto const* hostHasInitialState = inputs[maskIdx];
        if (reqTypes[0] == RequestType::kCONTEXT)
        {
            if (maskType == DataType::kINT32)
            {
                TLLM_CUDA_CHECK(cudaMemcpy2DAsync(workspace, sizeof(int8_t), hostHasInitialState, sizeof(int32_t),
                    sizeof(int8_t), batchSize, cudaMemcpyHostToDevice, stream));
            }
            else
            {
                TLLM_CUDA_CHECK(cudaMemcpyAsync(
                    workspace, hostHasInitialState, batchSize * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
            }
            hasInitialState = static_cast<int8_t const*>(workspace);
        }
        else if (reqTypes[0] == RequestType::kGENERATION)
        {
            for (int requestIdx = 0; requestIdx < batchSize; ++requestIdx)
            {
                auto const hasState = maskType == DataType::kINT32
                    ? static_cast<int32_t const*>(hostHasInitialState)[requestIdx]
                    : static_cast<int8_t const*>(hostHasInitialState)[requestIdx];
                TLLM_CHECK_WITH_INFO(
                    hasState == 1, "MambaConv1d generation request %d does not have an initial state", requestIdx);
            }
        }
    }

    auto const strideElements = [](int64_t strideBytes, int64_t defaultElements, char const* strideName)
    {
        TLLM_CHECK_WITH_INFO(
            strideBytes % sizeof(T) == 0, "%s must be divisible by the state element size", strideName);
        return strideBytes == 0 ? defaultElements : strideBytes / static_cast<int64_t>(sizeof(T));
    };
    int64_t const stateSlotStride
        = strideElements(mStateSlotStrideBytes, static_cast<int64_t>(mDConv - 1) * mDim, "state_slot_stride_bytes");
    int64_t const stateChannelStride = strideElements(mStateChannelStrideBytes, 1, "state_channel_stride_bytes");
    int64_t const stateHistoryStride = strideElements(mStateHistoryStrideBytes, mDim, "state_history_stride_bytes");
    int64_t const stateSpan = static_cast<int64_t>(mDim - 1) * stateChannelStride
        + static_cast<int64_t>(mDConv - 2) * stateHistoryStride + 1;
    TLLM_CHECK_WITH_INFO(
        stateSlotStride >= stateSpan, "state_slot_stride_bytes is too small for the configured Conv state layout");

    setMambaConv1dParams(mambaConv1dParams, batchSize, mDim, maxSeqLen, mDConv, mPreStride, mPostStride,
        inputs[getInputTensorIdx()], stateInPtr, stateOutPtr, inputs[getWeightIdx()], inputs[getBiasIdx()], outputs[0],
        static_cast<int const*>(inputs[getLastTokenIdsIdx()]), sourceSlotMapping, targetSlotMapping, hasInitialState,
        stateSlotStride, stateChannelStride, stateHistoryStride, mRemovePadding, mApplySilu);

    if (reqTypes[0] == RequestType::kCONTEXT)
    {
        invokeMambaConv1dContext<T>(mambaConv1dParams, stream);
    }
    else if (reqTypes[0] == RequestType::kGENERATION)
    {
        invokeMambaConv1dGeneration<T>(mambaConv1dParams, stream);
    }
    sync_check_cuda_error(stream);
    return 0;
}

int MambaConv1dPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    if (isBuilding())
    {
        return 0;
    }
    if (mType == DataType::kHALF)
    {
        return enqueueImpl<half>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
    else if (mType == DataType::kFLOAT)
    {
        return enqueueImpl<float>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#ifdef ENABLE_BF16
    else if (mType == DataType::kBF16)
    {
        return enqueueImpl<__nv_bfloat16>(inputDesc, outputDesc, inputs, outputs, workspace, stream);
    }
#endif
    return 0;
}

// IPluginV2Ext Methods
nvinfer1::DataType MambaConv1dPlugin::getOutputDataType(
    int index, nvinfer1::DataType const* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[getInputTensorIdx()];
}

// IPluginV2 Methods

char const* MambaConv1dPlugin::getPluginType() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_NAME;
}

char const* MambaConv1dPlugin::getPluginVersion() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_VERSION;
}

int MambaConv1dPlugin::getNbOutputs() const noexcept
{
    return 2;
}

int MambaConv1dPlugin::initialize() noexcept
{
    return 0;
}

void MambaConv1dPlugin::terminate() noexcept {}

size_t MambaConv1dPlugin::getSerializationSize() const noexcept
{
    return sizeof(mDim) + sizeof(mDConv) + sizeof(mPreStride) + sizeof(mPostStride) + sizeof(mType)
        + sizeof(mRemovePadding) + sizeof(mPagedState) + sizeof(mApplySilu) + sizeof(mStateSlotStrideBytes)
        + sizeof(mStateChannelStrideBytes) + sizeof(mStateHistoryStrideBytes) + sizeof(mUseInitialStateMask)
        + sizeof(mUseSeparateStateSlotMapping);
}

void MambaConv1dPlugin::serialize(void* buffer) const noexcept
{
    char *d = static_cast<char*>(buffer), *a = d;
    write(d, mDim);
    write(d, mDConv);
    write(d, mPreStride);
    write(d, mPostStride);
    write(d, mType);
    write(d, mRemovePadding);
    write(d, mPagedState);
    write(d, mApplySilu);
    write(d, mStateSlotStrideBytes);
    write(d, mStateChannelStrideBytes);
    write(d, mStateHistoryStrideBytes);
    write(d, mUseInitialStateMask);
    write(d, mUseSeparateStateSlotMapping);
    TLLM_CHECK(d == a + getSerializationSize());
}

void MambaConv1dPlugin::destroy() noexcept
{
    delete this;
}

///////////////

MambaConv1dPluginCreator::MambaConv1dPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("dim", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("dconv", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("pre_stride", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("post_stride", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("type_id", nullptr, PluginFieldType::kINT32));
    mPluginAttributes.emplace_back(PluginField("remove_input_padding", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("paged_state", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("apply_silu", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("state_slot_stride_bytes", nullptr, PluginFieldType::kINT64));
    mPluginAttributes.emplace_back(PluginField("state_channel_stride_bytes", nullptr, PluginFieldType::kINT64));
    mPluginAttributes.emplace_back(PluginField("state_history_stride_bytes", nullptr, PluginFieldType::kINT64));
    mPluginAttributes.emplace_back(PluginField("use_initial_state_mask", nullptr, PluginFieldType::kINT8));
    mPluginAttributes.emplace_back(PluginField("use_separate_state_slot_mapping", nullptr, PluginFieldType::kINT8));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

char const* MambaConv1dPluginCreator::getPluginName() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_NAME;
}

char const* MambaConv1dPluginCreator::getPluginVersion() const noexcept
{
    return MAMBA_CONV1D_PLUGIN_VERSION;
}

PluginFieldCollection const* MambaConv1dPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* MambaConv1dPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc) noexcept
{
    PluginField const* fields = fc->fields;
    int dim{};
    int dconv{};
    int pre_stride{};
    int post_stride{};
    bool removePadding{};
    bool pagedState{};
    bool applySilu{};
    int64_t stateSlotStrideBytes{};
    int64_t stateChannelStrideBytes{};
    int64_t stateHistoryStrideBytes{};
    bool useInitialStateMask{};
    bool useSeparateStateSlotMapping{};
    nvinfer1::DataType type{};
    // Read configurations from each fields
    for (int i = 0; i < fc->nbFields; ++i)
    {
        char const* attrName = fields[i].name;
        if (!strcmp(attrName, "dim"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dim = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "dconv"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            dconv = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "pre_stride"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            pre_stride = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "post_stride"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            post_stride = static_cast<int>(*(static_cast<int const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "type_id"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT32);
            type = static_cast<nvinfer1::DataType>(*(static_cast<nvinfer1::DataType const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "remove_input_padding"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            removePadding = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "paged_state"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            pagedState = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "apply_silu"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            applySilu = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "state_slot_stride_bytes"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT64);
            stateSlotStrideBytes = *(static_cast<int64_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "state_channel_stride_bytes"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT64);
            stateChannelStrideBytes = *(static_cast<int64_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "state_history_stride_bytes"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT64);
            stateHistoryStrideBytes = *(static_cast<int64_t const*>(fields[i].data));
        }
        else if (!strcmp(attrName, "use_initial_state_mask"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            useInitialStateMask = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
        else if (!strcmp(attrName, "use_separate_state_slot_mapping"))
        {
            TLLM_CHECK(fields[i].type == PluginFieldType::kINT8);
            useSeparateStateSlotMapping = static_cast<bool>(*(static_cast<bool const*>(fields[i].data)));
        }
    }
    try
    {
        auto* obj = new MambaConv1dPlugin(dim, dconv, pre_stride, post_stride, type, removePadding, pagedState,
            applySilu, stateSlotStrideBytes, stateChannelStrideBytes, stateHistoryStrideBytes, useInitialStateMask,
            useSeparateStateSlotMapping);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2* MambaConv1dPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    // This object will be deleted when the network is destroyed, which will
    // call MambaConv1dPlugin::destroy()
    try
    {
        auto* obj = new MambaConv1dPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
    catch (std::exception const& e)
    {
        caughtError(e);
    }
    return nullptr;
}
