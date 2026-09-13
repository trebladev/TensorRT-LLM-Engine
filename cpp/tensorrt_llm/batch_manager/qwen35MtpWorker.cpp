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
#include "qwen35MtpWorker.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/rawEngine.h"

#include <algorithm>
#include <filesystem>

namespace tensorrt_llm::batch_manager
{
using namespace runtime;

Qwen35MtpWorker::Qwen35MtpWorker(std::string const& enginePath, nvinfer1::ILogger* logger, SizeType32 maxSequenceLength,
    SizeType32 hiddenSize, SizeType32 vocabSize)
    : mRuntime(RawEngine(std::filesystem::path(enginePath)), logger)
    , mMaxSequenceLength(maxSequenceLength)
    , mHiddenSize(hiddenSize)
    , mVocabSize(vocabSize)
{
    auto const& engine = mRuntime.getEngine();
    TLLM_CHECK_WITH_INFO(engine.getNbOptimizationProfiles() == 1, "Native MTP requires a single-profile draft engine");
    TLLM_CHECK_WITH_INFO(engine.getTensorIOMode("target_hidden_states") == nvinfer1::TensorIOMode::kINPUT
            && engine.getTensorIOMode("past_key_value_0") == nvinfer1::TensorIOMode::kINPUT,
        "Native MTP requires a continuous-KV draft engine");
    auto const hiddenShape = engine.getTensorShape("target_hidden_states");
    TLLM_CHECK_WITH_INFO(hiddenShape.nbDims == 2 && hiddenShape.d[1] == hiddenSize
            && engine.getTensorDataType("target_hidden_states") == nvinfer1::DataType::kBF16,
        "Native MTP draft hidden-state dimensions or dtype differ from target");
    auto const kvShape = engine.getProfileShape("past_key_value_0", 0, nvinfer1::OptProfileSelector::kMAX);
    TLLM_CHECK_WITH_INFO(kvShape.nbDims == 5 && kvShape.d[0] == 1 && kvShape.d[3] >= maxSequenceLength,
        "Native MTP draft cache must cover the target maximum sequence length, with batch one");
    auto const logitsShape = engine.getTensorShape("logits");
    TLLM_CHECK_WITH_INFO(logitsShape.nbDims == 2 && logitsShape.d[1] == vocabSize
            && engine.getTensorDataType("logits") == nvinfer1::DataType::kFLOAT,
        "Native MTP requires FP32 draft logits with the target vocabulary");
    mRuntime.addContext(0);
    mRuntime.setCurrentBeamWidths({1});
}

void Qwen35MtpWorker::release(std::uint64_t requestId)
{
    if (mRequestId == requestId)
    {
        mRuntime.getStream().synchronize();
        mRequestId.reset();
        mKv.reset();
        mHiddenStates.reset();
        mRotaryCache.reset();
        mPositionDeltas.reset();
        mLength = 0;
        mPromptLength = 0;
    }
}

void Qwen35MtpWorker::capture(std::uint64_t requestId, bool context, TensorPtr const& hiddenStates,
    TensorPtr const& rotaryCache, TensorPtr const& positionDeltas)
{
    if (context)
    {
        TLLM_CHECK_WITH_INFO(!mRequestId || mRequestId == requestId, "Native MTP supports one active request");
        release(requestId);
        mRequestId = requestId;
    }
    TLLM_CHECK_WITH_INFO(mRequestId == requestId, "Native MTP generation has no matching draft history");
    TLLM_CHECK(hiddenStates && rotaryCache && positionDeltas);
    mContext = context;
    mHiddenStates = hiddenStates;
    mRotaryCache = rotaryCache;
    mPositionDeltas = positionDeltas;
}

TokenIdType Qwen35MtpWorker::draft(std::vector<TokenIdType> const& tokens)
{
    auto const count = static_cast<SizeType32>(tokens.size());
    TLLM_CHECK_WITH_INFO(count > 0 && (mContext || count <= 2), "Invalid native MTP accepted-prefix length");
    TLLM_CHECK(mLength + count <= mMaxSequenceLength);
    TLLM_CHECK(mHiddenStates && mHiddenStates->getShape().d[0] >= count);
    auto& manager = mRuntime.getBufferManager();
    auto const& engine = mRuntime.getEngine();
    std::vector<TensorPtr> staging;
    auto ints = [&](std::vector<SizeType32> const& values, nvinfer1::Dims const& shape, bool gpu) -> TensorPtr
    {
        TensorPtr host = BufferManager::cpu(shape, nvinfer1::DataType::kINT32);
        TLLM_CHECK(host->getSize() == values.size());
        std::copy(values.begin(), values.end(), bufferCast<SizeType32>(*host));
        staging.push_back(host);
        return gpu ? TensorPtr(manager.copyFrom(*host, MemoryType::kGPU)) : host;
    };
    auto scalar = [&](SizeType32 value, bool gpu) { return ints({value}, ITensor::makeShape({1}), gpu); };
    if (mContext)
    {
        auto shape = engine.getProfileShape("past_key_value_0", 0, nvinfer1::OptProfileSelector::kMAX);
        shape.d[3] = mMaxSequenceLength;
        mKv = manager.gpu(shape, nvinfer1::DataType::kBF16);
        manager.setZero(*mKv);
    }
    auto const promptLength = mContext ? count : mPromptLength;
    auto const verification = !mContext && count == 2;
    auto const generationLength = verification ? 2 : 1;
    TensorPtr knobs = BufferManager::cpu(ITensor::makeShape({16}), nvinfer1::DataType::kINT64);
    std::fill_n(bufferCast<std::int64_t>(*knobs), 16, -1);
    TensorPtr progress = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT64);
    bufferCast<std::int64_t>(*progress)[0] = 0;
    TensorPtr indirection = manager.gpu(ITensor::makeShape({1, 1, mMaxSequenceLength}), nvinfer1::DataType::kINT32);
    manager.setZero(*indirection);
    TllmRuntime::TensorMap allInputs{{"input_ids", ints(tokens, ITensor::makeShape({count}), true)},
        {"target_hidden_states", ITensor::slice(mHiddenStates, 0, count)}, {"past_key_value_0", mKv},
        {"position_ids", scalar(mLength, true)}, {"last_token_ids", scalar(count, true)},
        {"context_lengths", scalar(promptLength, true)}, {"host_context_lengths", scalar(promptLength, false)},
        {"sequence_length", scalar(mLength + count, true)}, {"host_past_key_value_lengths", scalar(mLength, false)},
        {"host_request_types", scalar(mContext ? 0 : 1, false)},
        {"host_max_attention_window_sizes", scalar(mMaxSequenceLength, false)},
        {"host_sink_token_length", scalar(0, false)}, {"host_runtime_perf_knobs", knobs},
        {"host_context_progress", progress}, {"cache_indirection", indirection}, {"mrope_rotary_cos_sin", mRotaryCache},
        {"mrope_position_deltas", mPositionDeltas}, {"spec_decoding_use", scalar(verification ? 1 : 0, false)},
        {"spec_decoding_generation_lengths", scalar(generationLength, true)},
        {"spec_decoding_position_offsets",
            ints(verification ? std::vector<SizeType32>{0, 1} : std::vector<SizeType32>{0},
                ITensor::makeShape({1, generationLength}), true)},
        {"spec_decoding_packed_mask",
            ints(verification ? std::vector<SizeType32>{1, 3} : std::vector<SizeType32>{1},
                ITensor::makeShape({generationLength, 1}), true)}};
    TllmRuntime::TensorMap inputs;
    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        auto const* name = engine.getIOTensorName(i);
        if (engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            TLLM_CHECK_WITH_INFO(allInputs.count(name), "Unsupported native MTP input: %s", name);
            inputs.emplace(name, allInputs.at(name));
        }
    }
    mRuntime.setInputTensors(0, inputs);
    TllmRuntime::TensorMap outputs;
    mRuntime.setOutputTensors(0, outputs);
    TLLM_CHECK_WITH_INFO(mRuntime.executeContext(0), "Native MTP draft engine enqueue failed");
    auto const lastLogits = ITensor::slice(outputs.at("logits"), count - 1, 1);
    auto hostLogits = manager.copyFrom(*lastLogits, MemoryType::kCPU);
    manager.getStream().synchronize();
    auto const* logits = bufferCast<float const>(*hostLogits);
    auto const token = static_cast<TokenIdType>(std::max_element(logits, logits + mVocabSize) - logits);
    mKv = outputs.at("present_key_value_0");
    mLength += count;
    mPromptLength = promptLength;
    return token;
}
} // namespace tensorrt_llm::batch_manager
