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

#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <map>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager
{
//! Synchronous native K=1 drafter with independent request histories and batched forwards.
class Qwen35MtpWorker
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using Tokens = std::vector<runtime::TokenIdType>;
    Qwen35MtpWorker(std::string const& enginePath, nvinfer1::ILogger* logger, runtime::SizeType32 maxSequenceLength,
        runtime::SizeType32 hiddenSize, runtime::SizeType32 vocabSize, runtime::SizeType32 maxBatchSize,
        runtime::SizeType32 rotaryDim);

    //! Retain this request's packed target outputs until acceptance selects the valid prefix.
    void capture(std::uint64_t requestId, bool context, TensorPtr const& hiddenStates, TensorPtr const& rotaryCache,
        runtime::SizeType32 positionDelta);

    //! Queue shifted prompt tokens or newly accepted tokens for the next batched forward.
    void queue(std::uint64_t requestId, Tokens tokens);
    std::map<std::uint64_t, runtime::TokenIdType> draft();

    [[nodiscard]] bool isContext(std::uint64_t requestId) const;
    //! Release history when its request completes, fails, is canceled, or pauses.
    void release(std::uint64_t requestId);

private:
    struct RequestState
    {
        runtime::SizeType32 length = 0;
        runtime::SizeType32 promptLength = 0;
        runtime::SizeType32 positionDelta = 0;
        bool context = false;
        TensorPtr kv;
        TensorPtr hiddenStates;
        TensorPtr rotaryCache;
        Tokens tokens;
    };

    std::vector<runtime::TokenIdType> draftBatch(std::vector<RequestState*> const& states);

    runtime::TllmRuntime mRuntime;
    runtime::SizeType32 mMaxSequenceLength;
    runtime::SizeType32 mHiddenSize;
    runtime::SizeType32 mVocabSize;
    runtime::SizeType32 mMaxBatchSize;
    runtime::SizeType32 mRotaryDim;
    std::map<std::uint64_t, RequestState> mRequests;
};
} // namespace tensorrt_llm::batch_manager
