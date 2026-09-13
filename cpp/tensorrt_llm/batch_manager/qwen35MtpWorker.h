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

#include <optional>
#include <string>
#include <vector>

namespace tensorrt_llm::batch_manager
{
//! Synchronous native K=1 drafter for the single-active-request executor baseline.
class Qwen35MtpWorker
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    Qwen35MtpWorker(std::string const& enginePath, nvinfer1::ILogger* logger, runtime::SizeType32 maxSequenceLength,
        runtime::SizeType32 hiddenSize, runtime::SizeType32 vocabSize);

    //! Retain the target outputs until acceptance selects the valid prefix.
    void capture(std::uint64_t requestId, bool context, TensorPtr const& hiddenStates, TensorPtr const& rotaryCache,
        TensorPtr const& positionDeltas);

    //! Append shifted prompt tokens or newly accepted tokens and predict the next candidate.
    runtime::TokenIdType draft(std::vector<runtime::TokenIdType> const& tokens);

    [[nodiscard]] bool isContext() const
    {
        return mContext;
    }

    //! Release history when its request completes, fails, is canceled, or pauses.
    void release(std::uint64_t requestId);

private:
    runtime::TllmRuntime mRuntime;
    runtime::SizeType32 mMaxSequenceLength;
    runtime::SizeType32 mHiddenSize;
    runtime::SizeType32 mVocabSize;
    runtime::SizeType32 mLength = 0;
    runtime::SizeType32 mPromptLength = 0;
    std::optional<std::uint64_t> mRequestId;
    bool mContext = false;
    TensorPtr mKv;
    TensorPtr mHiddenStates;
    TensorPtr mRotaryCache;
    TensorPtr mPositionDeltas;
};
} // namespace tensorrt_llm::batch_manager
