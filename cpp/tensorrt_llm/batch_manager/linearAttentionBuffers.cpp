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

#include <algorithm>
#include <cstdint>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

LinearAttentionBuffers::LinearAttentionBuffers(
    SizeType32 maxBatchSize, BufferManager const& manager, SizeType32 maxSnapshotTokens, bool externalDraftVerification)
    : mExternalDraftVerification(externalDraftVerification)
{
    if (mExternalDraftVerification)
    {
        TLLM_CHECK(maxSnapshotTokens > 0);
        // Context candidates use causal attention; native MTP also enables
        // two-token speculative generation on later iterations.
        mSpecDecodingUse = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT32);
        bufferCast<SizeType32>(*mSpecDecodingUse)[0] = 0;
        mSpecDecodingLengths = manager.gpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
        mSpecDecodingOffsets = manager.gpu(ITensor::makeShape({maxBatchSize, 1}), nvinfer1::DataType::kINT32);
        mSpecDecodingMask = manager.gpu(ITensor::makeShape({maxBatchSize, 1}), nvinfer1::DataType::kINT32);
        mSpecDecodingLengthsHost = BufferManager::cpu(ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
        mSpecDecodingOffsetsHost
            = BufferManager::cpu(ITensor::makeShape({maxBatchSize, 2}), nvinfer1::DataType::kINT32);
        mSpecDecodingMaskHost
            = BufferManager::cpu(ITensor::makeShape({maxBatchSize * 2, 1}), nvinfer1::DataType::kINT32);
        manager.setZero(*mSpecDecodingLengths);
        manager.setZero(*mSpecDecodingOffsets);
        manager.setZero(*mSpecDecodingMask);
    }
    if (maxSnapshotTokens > 0)
    {
        auto const shape = ITensor::makeShape({maxSnapshotTokens});
        snapshotSlotMappingHost = BufferManager::cpu(shape, nvinfer1::DataType::kINT32);
        snapshotSlotMappingDevice = manager.gpu(shape, nvinfer1::DataType::kINT32);
    }
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
    if (mExternalDraftVerification)
    {
        mSpecDecodingLengths->reshape(sequenceShape);
        mSpecDecodingOffsets->reshape(ITensor::makeShape({numSequences, 1}));
        mSpecDecodingMask->reshape(ITensor::makeShape({numSequences, 1}));
    }
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

    auto const recordsPerBlock = mExternalDraftVerification ? kVerificationStateRecords : 1;
    TLLM_CHECK_WITH_INFO(!mExternalDraftVerification || !kvCacheManager.isEnableBlockReuse(),
        "Qwen3.5 external draft verification does not support prefix reuse.");
    SizeType32* snapshotSlots = nullptr;
    if (snapshotSlotMappingHost)
    {
        SizeType32 numTokens = 0;
        for (auto const& request : generationRequests)
        {
            numTokens += 1 + request->getNumDraftTokens();
        }
        for (auto const& request : contextRequests)
        {
            numTokens
                += request->getContextChunkSize() + (request->isLastContextChunk() ? request->getNumDraftTokens() : 0);
        }
        snapshotSlotMappingHost->reshape(ITensor::makeShape({numTokens}));
        snapshotSlotMappingDevice->reshape(ITensor::makeShape({numTokens}));
        snapshotSlots = bufferCast<SizeType32>(*snapshotSlotMappingHost);
        std::fill_n(snapshotSlots, numTokens, -1);
    }

    SizeType32 sequenceIdx = 0;
    SizeType32 cumulativeLength = 0;
    cuSeqlens[0] = 0;

    for (auto const& request : contextRequests)
    {
        auto const draftLength = request->getNumDraftTokens();
        TLLM_CHECK_WITH_INFO(draftLength == 0
                || (mExternalDraftVerification && draftLength == 1 && request->isLastContextChunk()
                    && request->getContextCurrentPosition() == 0),
            "Qwen3.5 external draft verification requires one candidate and unchunked context.");
        auto const contextStart = request->getContextCurrentPosition();
        auto const contextEnd = contextStart + request->getContextChunkSize();
        auto const sourceTokenIdx = contextStart > 0 ? std::make_optional(contextStart - 1) : std::nullopt;
        auto const slots
            = kvCacheManager.getRecurrentStateSlotPair(request->mRequestId, sourceTokenIdx, contextEnd - 1);
        sourceStateSlotMapping[sequenceIdx] = recordsPerBlock * slots.sourceSlot.value_or(slots.targetSlot);
        targetStateSlotMapping[sequenceIdx] = recordsPerBlock * slots.targetSlot;
        if (snapshotSlots && kvCacheManager.isEnableBlockReuse())
        {
            // Iterate the actual allocated blocks, including visual/last-full snapshots.
            // Placeholder positions have no GPU state and must never be published as snapshots.
            auto const window = kv_cache_manager::LinearAttentionMetadata::kRecurrentStates;
            auto const& ids = kvCacheManager.getCacheBlockIds(request->mRequestId, window).at(0);
            auto const tokensPerBlock = kvCacheManager.getTokensPerBlock();
            for (auto end = (contextStart / tokensPerBlock + 1) * tokensPerBlock; end <= contextEnd;
                 end += tokensPerBlock)
            {
                auto const block
                    = kvCacheManager.getBlockManager().getBlockById(ids.at(end / tokensPerBlock - 1), window);
                if (!block->isPlaceholder())
                {
                    snapshotSlots[cumulativeLength + end - contextStart - 1] = block->getMemoryPoolBlockIndex();
                }
            }
        }
        if (draftLength > 0)
        {
            snapshotSlots[cumulativeLength + request->getContextChunkSize() - 1]
                = recordsPerBlock * slots.targetSlot + 1;
            targetStateSlotMapping[sequenceIdx] += 2;
            // Fresh prefill zeroes the target record. Read that same zeroed record,
            // not the committed bank left by an earlier owner of this physical block.
            if (!slots.sourceSlot.has_value())
            {
                sourceStateSlotMapping[sequenceIdx] = targetStateSlotMapping[sequenceIdx];
            }
        }
        cumulativeLength += request->getContextChunkSize() + draftLength;
        cuSeqlens[sequenceIdx + 1] = cumulativeLength;
        hasInitialState[sequenceIdx] = slots.sourceSlot.has_value() ? 1 : 0;
        ++sequenceIdx;
    }

    for (auto const& request : generationRequests)
    {
        auto const beamWidth = request->getBeamWidthByIter();
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "Qwen3.5 linear attention only supports beam width 1.");
        auto const draftLength = request->getNumDraftTokens();
        TLLM_CHECK_WITH_INFO(draftLength == 0 || (mExternalDraftVerification && draftLength == 1),
            "Qwen3.5 linear attention only supports K=1 verification");
        auto const lastTokenIdx = request->getNumTokens(/*beam=*/0) - 1;
        auto const slots = kvCacheManager.getRecurrentStateSlotPair(
            request->mRequestId, /*sourceTokenIdx=*/lastTokenIdx, /*targetTokenIdx=*/lastTokenIdx);
        sourceStateSlotMapping[sequenceIdx] = recordsPerBlock * slots.sourceSlot.value();
        targetStateSlotMapping[sequenceIdx] = recordsPerBlock * slots.targetSlot;
        if (draftLength > 0)
        {
            snapshotSlots[cumulativeLength] = recordsPerBlock * slots.targetSlot + 1;
            targetStateSlotMapping[sequenceIdx] += 2;
        }
        cumulativeLength += 1 + draftLength;
        cuSeqlens[sequenceIdx + 1] = cumulativeLength;
        hasInitialState[sequenceIdx] = 1;
        ++sequenceIdx;
    }

    if (mExternalDraftVerification)
    {
        SizeType32 width = 1;
        SizeType32 numGenTokens = 0;
        for (auto const& request : generationRequests)
        {
            auto const count = request->getNumDraftTokens() + 1;
            width = std::max(width, count);
            numGenTokens += count;
        }
        // The attention plugin indexes speculative metadata from generation
        // sequence zero, even when context tokens precede it in the packed batch.
        auto const numGen = static_cast<SizeType32>(generationRequests.size());
        bufferCast<SizeType32>(*mSpecDecodingUse)[0] = width > 1 ? 1 : 0;
        // Keep one dummy row for older single-request engine profiles in context-only steps.
        auto const rows = std::max(1, numGen);
        mSpecDecodingLengthsHost->reshape(ITensor::makeShape({rows}));
        mSpecDecodingOffsetsHost->reshape(ITensor::makeShape({rows, width}));
        mSpecDecodingMaskHost->reshape(ITensor::makeShape({std::max(1, numGenTokens), 1}));
        std::fill_n(bufferCast<SizeType32>(*mSpecDecodingLengthsHost), rows, 1);
        std::fill_n(bufferCast<SizeType32>(*mSpecDecodingMaskHost), std::max(1, numGenTokens), 1);
        for (SizeType32 i = 0; i < rows * width; ++i)
        {
            bufferCast<SizeType32>(*mSpecDecodingOffsetsHost)[i] = i % width;
        }
        SizeType32 tokenOffset = 0;
        for (SizeType32 i = 0; i < numGen; ++i)
        {
            auto const count = generationRequests[i]->getNumDraftTokens() + 1;
            bufferCast<SizeType32>(*mSpecDecodingLengthsHost)[i] = count;
            for (SizeType32 token = 0; token < count; ++token)
            {
                bufferCast<SizeType32>(*mSpecDecodingMaskHost)[tokenOffset++] = (1 << (token + 1)) - 1;
            }
        }
    }

    TLLM_CHECK(sourceStateSlotMappingHost->getSize() == static_cast<std::size_t>(sequenceIdx));
    TLLM_CHECK(targetStateSlotMappingHost->getSize() == static_cast<std::size_t>(sequenceIdx));
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void LinearAttentionBuffers::copyToDevice(BufferManager const& manager)
{
    if (mExternalDraftVerification)
    {
        mSpecDecodingLengths->reshape(mSpecDecodingLengthsHost->getShape());
        mSpecDecodingOffsets->reshape(mSpecDecodingOffsetsHost->getShape());
        mSpecDecodingMask->reshape(mSpecDecodingMaskHost->getShape());
        manager.copy(*mSpecDecodingLengthsHost, *mSpecDecodingLengths);
        manager.copy(*mSpecDecodingOffsetsHost, *mSpecDecodingOffsets);
        manager.copy(*mSpecDecodingMaskHost, *mSpecDecodingMask);
    }
    manager.copy(*sourceStateSlotMappingHost, *sourceStateSlotMappingDevice);
    manager.copy(*targetStateSlotMappingHost, *targetStateSlotMappingDevice);
    manager.copy(*cuSeqlensHost, *cuSeqlensDevice);
    if (snapshotSlotMappingHost)
    {
        manager.copy(*snapshotSlotMappingHost, *snapshotSlotMappingDevice);
    }
}

void LinearAttentionBuffers::getBuffers(TensorMap& inputBuffers) const
{
    if (mExternalDraftVerification)
    {
        inputBuffers.insert_or_assign("spec_decoding_use", mSpecDecodingUse);
        inputBuffers.insert_or_assign("spec_decoding_generation_lengths", mSpecDecodingLengths);
        inputBuffers.insert_or_assign("spec_decoding_position_offsets", mSpecDecodingOffsets);
        inputBuffers.insert_or_assign("spec_decoding_packed_mask", mSpecDecodingMask);
    }
    if (snapshotSlotMappingDevice)
    {
        inputBuffers.insert_or_assign("state_snapshot_slot_mapping", snapshotSlotMappingDevice);
    }
    inputBuffers.insert_or_assign("source_state_slot_mapping", sourceStateSlotMappingDevice);
    inputBuffers.insert_or_assign("target_state_slot_mapping", targetStateSlotMappingDevice);
    inputBuffers.insert_or_assign("gated_delta_cu_seqlens", cuSeqlensDevice);
    inputBuffers.insert_or_assign("host_has_initial_state", hostHasInitialState);
}

} // namespace tensorrt_llm::batch_manager
