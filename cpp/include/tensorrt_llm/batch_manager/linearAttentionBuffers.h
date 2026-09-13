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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/runtime/iTensor.h"

namespace tensorrt_llm::runtime
{
class BufferManager;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class KVCacheManager;
} // namespace kv_cache_manager

class LinearAttentionBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorMap = runtime::ITensor::TensorMap;
    using TensorPtr = runtime::ITensor::SharedPtr;

    // Physical cache blocks contain committed, before-draft, and after-draft records.
    static constexpr SizeType32 kVerificationStateRecords = 3;

    TensorPtr sourceStateSlotMappingHost;
    TensorPtr sourceStateSlotMappingDevice;
    TensorPtr targetStateSlotMappingHost;
    TensorPtr targetStateSlotMappingDevice;
    TensorPtr cuSeqlensHost;
    TensorPtr cuSeqlensDevice;
    TensorPtr hostHasInitialState;
    // Packed-token mapping: slot to snapshot after this token, or -1.
    TensorPtr snapshotSlotMappingHost;
    TensorPtr snapshotSlotMappingDevice;

    LinearAttentionBuffers(SizeType32 maxBatchSize, runtime::BufferManager const& manager,
        SizeType32 maxSnapshotTokens = 0, bool externalDraftVerification = false);

    void reshape(SizeType32 numSequences);

    void fill(RequestVector const& contextRequests, RequestVector const& generationRequests,
        kv_cache_manager::KVCacheManager const& kvCacheManager);

    void copyToDevice(runtime::BufferManager const& manager);

    void getBuffers(TensorMap& inputBuffers) const;

private:
    bool mExternalDraftVerification;
    TensorPtr mSpecDecodingUse;
    TensorPtr mSpecDecodingLengths;
    TensorPtr mSpecDecodingOffsets;
    TensorPtr mSpecDecodingMask;
    TensorPtr mSpecDecodingLengthsHost;
    TensorPtr mSpecDecodingOffsetsHost;
    TensorPtr mSpecDecodingMaskHost;
};

} // namespace tensorrt_llm::batch_manager
