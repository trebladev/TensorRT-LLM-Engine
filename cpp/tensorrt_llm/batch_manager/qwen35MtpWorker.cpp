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
#include "tensorrt_llm/kernels/speculativeDecoding/mtpKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/rawEngine.h"

#include <algorithm>
#include <filesystem>

namespace tensorrt_llm::batch_manager
{
using namespace runtime;

Qwen35MtpWorker::Qwen35MtpWorker(std::string const& enginePath, nvinfer1::ILogger* logger, SizeType32 maxSequenceLength,
    SizeType32 hiddenSize, SizeType32 vocabSize, SizeType32 maxBatchSize, SizeType32 rotaryDim)
    : mRuntime(RawEngine(std::filesystem::path(enginePath)), logger)
    , mMaxSequenceLength(maxSequenceLength)
    , mHiddenSize(hiddenSize)
    , mVocabSize(vocabSize)
    , mMaxBatchSize(maxBatchSize)
    , mRotaryDim(rotaryDim)
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
    TLLM_CHECK_WITH_INFO(kvShape.nbDims == 5 && kvShape.d[0] >= maxBatchSize && kvShape.d[3] >= maxSequenceLength,
        "Native MTP draft cache must cover the target maximum sequence length, and runtime batch size");
    auto const logitsShape = engine.getTensorShape("logits");
    TLLM_CHECK_WITH_INFO(logitsShape.nbDims == 2 && logitsShape.d[1] == vocabSize
            && engine.getTensorDataType("logits") == nvinfer1::DataType::kFLOAT,
        "Native MTP requires FP32 draft logits with the target vocabulary");
    mRuntime.addContext(0);
}

void Qwen35MtpWorker::release(std::uint64_t requestId)
{
    mRequests.erase(requestId);
}

bool Qwen35MtpWorker::isContext(std::uint64_t requestId) const
{
    return mRequests.at(requestId).context;
}

void Qwen35MtpWorker::capture(std::uint64_t requestId, bool context, TensorPtr const& hiddenStates,
    TensorPtr const& rotaryCache, SizeType32 positionDelta)
{
    if (context)
    {
        release(requestId);
        mRequests.emplace(requestId, RequestState{});
    }
    TLLM_CHECK_WITH_INFO(mRequests.count(requestId), "Native MTP generation has no matching draft history");
    TLLM_CHECK(hiddenStates && rotaryCache);
    auto& state = mRequests.at(requestId);
    TLLM_CHECK(state.tokens.empty());
    state.context = context;
    state.hiddenStates = hiddenStates;
    state.rotaryCache = rotaryCache;
    state.positionDelta = positionDelta;
}

void Qwen35MtpWorker::queue(std::uint64_t requestId, Tokens tokens)
{
    auto& state = mRequests.at(requestId);
    TLLM_CHECK(state.tokens.empty());
    TLLM_CHECK(!tokens.empty() && (state.context || tokens.size() <= 2));
    state.tokens = std::move(tokens);
}

std::map<std::uint64_t, TokenIdType> Qwen35MtpWorker::draft()
{
    std::map<std::uint64_t, TokenIdType> result;
    // Prefills may have different lengths. Generation groups append either one
    // correction pair or two accepted pairs, using causal verification attention.
    for (SizeType32 group = 0; group < 3; ++group)
    {
        std::vector<RequestState*> states;
        std::vector<std::uint64_t> ids;
        for (auto& [id, state] : mRequests)
        {
            if (!state.tokens.empty() && (state.context ? 0 : static_cast<SizeType32>(state.tokens.size())) == group)
            {
                states.push_back(&state);
                ids.push_back(id);
            }
        }
        if (!states.empty())
        {
            auto const tokens = draftBatch(states);
            for (std::size_t i = 0; i < ids.size(); ++i)
            {
                result.emplace(ids[i], tokens[i]);
            }
        }
    }
    return result;
}

std::vector<TokenIdType> Qwen35MtpWorker::draftBatch(std::vector<RequestState*> const& states)
{
    auto const batchSize = static_cast<SizeType32>(states.size());
    TLLM_CHECK(batchSize > 0 && batchSize <= mMaxBatchSize);
    auto const context = states.front()->context;
    auto const width = context ? 1 : static_cast<SizeType32>(states.front()->tokens.size());
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
    auto batchInts = [&](std::vector<SizeType32> const& values, bool gpu)
    { return ints(values, ITensor::makeShape({batchSize}), gpu); };

    Tokens tokens;
    std::vector<SizeType32> pastLengths, promptLengths, sequenceLengths, lastTokenIds, positionDeltas;
    for (auto const* state : states)
    {
        auto const count = static_cast<SizeType32>(state->tokens.size());
        TLLM_CHECK(state->context == context && (context || count == width));
        TLLM_CHECK(state->length + count <= mMaxSequenceLength);
        TLLM_CHECK(state->hiddenStates->getShape().d[0] >= count);
        tokens.insert(tokens.end(), state->tokens.begin(), state->tokens.end());
        pastLengths.push_back(state->length);
        promptLengths.push_back(context ? count : state->promptLength);
        sequenceLengths.push_back(state->length + count);
        lastTokenIds.push_back(static_cast<SizeType32>(tokens.size()));
        positionDeltas.push_back(state->positionDelta);
    }
    auto const numTokens = static_cast<SizeType32>(tokens.size());
    auto kvShape = engine.getProfileShape("past_key_value_0", 0, nvinfer1::OptProfileSelector::kMAX);
    kvShape.d[0] = batchSize;
    kvShape.d[3] = mMaxSequenceLength;
    TensorPtr kv = manager.gpu(kvShape, nvinfer1::DataType::kBF16);
    TensorPtr hidden = manager.gpu(ITensor::makeShape({numTokens, mHiddenSize}), nvinfer1::DataType::kBF16);
    auto ropeShape = engine.getProfileShape("mrope_rotary_cos_sin", 0, nvinfer1::OptProfileSelector::kMAX);
    ropeShape.d[0] = batchSize;
    TensorPtr rope = manager.gpu(ropeShape, nvinfer1::DataType::kFLOAT);
    if (context)
    {
        manager.setZero(*kv);
    }
    TensorPtr ropeFlat = ITensor::view(rope, ITensor::makeShape({static_cast<ITensor::DimType64>(rope->getSize())}));
    SizeType32 offset = 0;
    for (SizeType32 i = 0; i < batchSize; ++i)
    {
        auto const& state = *states[i];
        auto const count = static_cast<SizeType32>(state.tokens.size());
        manager.copy(*ITensor::slice(state.hiddenStates, 0, count), *ITensor::slice(hidden, offset, count));
        if (context)
        {
            // Context MRoPE coefficients follow packed token order, not padded batch rows.
            manager.copy(*ITensor::slice(state.rotaryCache, 0, count * mRotaryDim),
                *ITensor::slice(ropeFlat, offset * mRotaryDim, count * mRotaryDim));
        }
        else
        {
            TLLM_CHECK(state.kv);
            manager.copy(*state.kv, *ITensor::slice(kv, i, 1));
        }
        offset += count;
    }
    TensorPtr knobs = BufferManager::cpu(ITensor::makeShape({16}), nvinfer1::DataType::kINT64);
    std::fill_n(bufferCast<std::int64_t>(*knobs), 16, -1);
    TensorPtr progress = BufferManager::cpu(ITensor::makeShape({1}), nvinfer1::DataType::kINT64);
    bufferCast<std::int64_t>(*progress)[0] = 0;
    TensorPtr indirection
        = manager.gpu(ITensor::makeShape({batchSize, 1, mMaxSequenceLength}), nvinfer1::DataType::kINT32);
    manager.setZero(*indirection);
    std::vector<SizeType32> offsets(batchSize * width), mask(batchSize * width);
    for (SizeType32 i = 0; i < batchSize * width; ++i)
    {
        offsets[i] = i % width;
        mask[i] = (1 << (i % width + 1)) - 1;
    }
    TllmRuntime::TensorMap allInputs{{"input_ids", ints(tokens, ITensor::makeShape({numTokens}), true)},
        {"target_hidden_states", hidden}, {"past_key_value_0", kv}, {"position_ids", batchInts(pastLengths, true)},
        {"last_token_ids", batchInts(lastTokenIds, true)}, {"context_lengths", batchInts(promptLengths, true)},
        {"host_context_lengths", batchInts(promptLengths, false)},
        {"sequence_length", batchInts(sequenceLengths, true)},
        {"host_past_key_value_lengths", batchInts(pastLengths, false)},
        {"host_request_types", batchInts(std::vector<SizeType32>(batchSize, context ? 0 : 1), false)},
        {"host_max_attention_window_sizes", scalar(mMaxSequenceLength, false)},
        {"host_sink_token_length", scalar(0, false)}, {"host_runtime_perf_knobs", knobs},
        {"host_context_progress", progress}, {"cache_indirection", indirection}, {"mrope_rotary_cos_sin", rope},
        {"mrope_position_deltas", ints(positionDeltas, ITensor::makeShape({batchSize, 1}), true)},
        {"spec_decoding_use", scalar(width > 1 ? 1 : 0, false)},
        {"spec_decoding_generation_lengths", batchInts(std::vector<SizeType32>(batchSize, width), true)},
        {"spec_decoding_position_offsets", ints(offsets, ITensor::makeShape({batchSize, width}), true)},
        {"spec_decoding_packed_mask", ints(mask, ITensor::makeShape({batchSize * width, 1}), true)}};
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
    mRuntime.setCurrentBeamWidths(std::vector<SizeType32>(batchSize, 1));
    mRuntime.setInputTensors(0, inputs);
    TllmRuntime::TensorMap outputs;
    mRuntime.setOutputTensors(0, outputs);
    TLLM_CHECK_WITH_INFO(mRuntime.executeContext(0), "Native MTP draft engine enqueue failed");
    TensorPtr selectedTokens = manager.gpu(ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    kernels::invokeMTPPackedGreedySampling(bufferCast<float const>(*outputs.at("logits")),
        bufferCast<SizeType32 const>(*allInputs.at("last_token_ids")), bufferCast<TokenIdType>(*selectedTokens),
        batchSize, mVocabSize, manager.getStream().get());
    TensorPtr hostTokens = manager.copyFrom(*selectedTokens, MemoryType::kCPU);
    std::vector<TensorPtr> nextKv;
    for (SizeType32 i = 0; i < batchSize; ++i)
    {
        // Own only this request's KV so finishing peers do not keep an entire batch allocation alive.
        nextKv.emplace_back(
            manager.copyFrom(*ITensor::slice(outputs.at("present_key_value_0"), i, 1), MemoryType::kGPU));
    }
    manager.getStream().synchronize();
    std::vector<TokenIdType> result;
    for (SizeType32 i = 0; i < batchSize; ++i)
    {
        result.push_back(bufferCast<TokenIdType const>(*hostTokens)[i]);
        auto& state = *states[i];
        state.kv = std::move(nextKv[i]);
        state.length = sequenceLengths[i];
        state.promptLength = promptLengths[i];
        state.tokens.clear();
        state.hiddenStates.reset();
        state.rotaryCache.reset();
    }
    return result;
}
} // namespace tensorrt_llm::batch_manager
