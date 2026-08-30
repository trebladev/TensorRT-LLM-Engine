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

#include "tensorrt_llm/batch_manager/linearAttentionBuffers.h"

#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"

#include <cstdint>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

LinearAttentionBuffers::LinearAttentionBuffers(SizeType32 maxBatchSize, BufferManager const& manager)
{
    auto const maxBatchShape = ITensor::makeShape({maxBatchSize});
    auto const maxCuSeqlensShape = ITensor::makeShape({maxBatchSize + 1});

    sourceStateSlotMappingHost = BufferManager::cpu(maxBatchShape, nvinfer1::DataType::kINT32);
    sourceStateSlotMappingDevice = manager.gpu(maxBatchShape, nvinfer1::DataType::kINT32);
    targetStateSlotMappingHost = BufferManager::cpu(maxBatchShape, nvinfer1::DataType::kINT32);
    targetStateSlotMappingDevice = manager.gpu(maxBatchShape, nvinfer1::DataType::kINT32);
    cuSeqlensHost = BufferManager::cpu(maxCuSeqlensShape, nvinfer1::DataType::kINT32);
    cuSeqlensDevice = manager.gpu(maxCuSeqlensShape, nvinfer1::DataType::kINT32);
    hostHasInitialState = BufferManager::cpu(maxBatchShape, nvinfer1::DataType::kINT32);
}

void LinearAttentionBuffers::reshape(SizeType32 numSequences)
{
    auto const sequenceShape = ITensor::makeShape({numSequences});
    sourceStateSlotMappingHost->reshape(sequenceShape);
    sourceStateSlotMappingDevice->reshape(sequenceShape);
    targetStateSlotMappingHost->reshape(sequenceShape);
    targetStateSlotMappingDevice->reshape(sequenceShape);
    hostHasInitialState->reshape(sequenceShape);
    cuSeqlensHost->reshape(ITensor::makeShape({numSequences + 1}));
    cuSeqlensDevice->reshape(ITensor::makeShape({numSequences + 1}));
}

void LinearAttentionBuffers::fill(RequestVector const& contextRequests, RequestVector const& generationRequests,
    kv_cache_manager::KVCacheManager const& kvCacheManager)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(linearAttentionBuffersFill);

    auto* sourceStateSlotMapping = bufferCast<SizeType32>(*sourceStateSlotMappingHost);
    auto* targetStateSlotMapping = bufferCast<SizeType32>(*targetStateSlotMappingHost);
    auto* cuSeqlens = bufferCast<SizeType32>(*cuSeqlensHost);
    auto* hasInitialState = bufferCast<SizeType32>(*hostHasInitialState);

    SizeType32 sequenceIdx = 0;
    SizeType32 cumulativeLength = 0;
    cuSeqlens[0] = 0;

    for (auto const& request : contextRequests)
    {
        TLLM_CHECK_WITH_INFO(
            request->getNumDraftTokens() == 0, "Qwen3.5 linear attention does not support draft tokens.");
        auto const contextStart = request->getContextCurrentPosition();
        auto const contextEnd = contextStart + request->getContextChunkSize();
        auto const sourceTokenIdx = contextStart > 0 ? std::make_optional(contextStart - 1) : std::nullopt;
        auto const slots
            = kvCacheManager.getRecurrentStateSlotPair(request->mRequestId, sourceTokenIdx, contextEnd - 1);
        sourceStateSlotMapping[sequenceIdx] = slots.sourceSlot.value_or(slots.targetSlot);
        targetStateSlotMapping[sequenceIdx] = slots.targetSlot;
        cumulativeLength += request->getContextChunkSize();
        cuSeqlens[sequenceIdx + 1] = cumulativeLength;
        hasInitialState[sequenceIdx] = slots.sourceSlot.has_value() ? 1 : 0;
        ++sequenceIdx;
    }

    for (auto const& request : generationRequests)
    {
        auto const beamWidth = request->getBeamWidthByIter();
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Qwen3.5 linear attention only supports beam width 1.");
        TLLM_CHECK_WITH_INFO(
            request->getNumDraftTokens() == 0, "Qwen3.5 linear attention does not support draft tokens.");
        auto const lastTokenIdx = request->getNumTokens(/*beam=*/0) - 1;
        auto const slots = kvCacheManager.getRecurrentStateSlotPair(
            request->mRequestId, /*sourceTokenIdx=*/lastTokenIdx, /*targetTokenIdx=*/lastTokenIdx);
        sourceStateSlotMapping[sequenceIdx] = slots.sourceSlot.value();
        targetStateSlotMapping[sequenceIdx] = slots.targetSlot;
        ++cumulativeLength;
        cuSeqlens[sequenceIdx + 1] = cumulativeLength;
        hasInitialState[sequenceIdx] = 1;
        ++sequenceIdx;
    }

    TLLM_CHECK(sourceStateSlotMappingHost->getSize() == static_cast<std::size_t>(sequenceIdx));
    TLLM_CHECK(targetStateSlotMappingHost->getSize() == static_cast<std::size_t>(sequenceIdx));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LinearAttentionBuffers::copyToDevice(BufferManager const& manager)
{
    manager.copy(*sourceStateSlotMappingHost, *sourceStateSlotMappingDevice);
    manager.copy(*targetStateSlotMappingHost, *targetStateSlotMappingDevice);
    manager.copy(*cuSeqlensHost, *cuSeqlensDevice);
}

void LinearAttentionBuffers::getBuffers(TensorMap& inputBuffers) const
{
    inputBuffers.insert_or_assign("source_state_slot_mapping", sourceStateSlotMappingDevice);
    inputBuffers.insert_or_assign("target_state_slot_mapping", targetStateSlotMappingDevice);
    inputBuffers.insert_or_assign("gated_delta_cu_seqlens", cuSeqlensDevice);
    inputBuffers.insert_or_assign("host_has_initial_state", hostHasInitialState);
}

} // namespace tensorrt_llm::batch_manager
