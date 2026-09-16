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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/mtpKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/rawEngine.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numeric>

namespace tensorrt_llm::batch_manager
{
using namespace runtime;

Qwen35MtpWorker::Qwen35MtpWorker(std::string const& enginePath, nvinfer1::ILogger* logger, SizeType32 maxSequenceLength,
    SizeType32 hiddenSize, SizeType32 vocabSize, SizeType32 maxBatchSize, SizeType32 rotaryDim,
    SizeType32 maxDraftLength)
    : mWorkspaces((maxDraftLength + 2) * maxDraftLength)
    , mRuntime(RawEngine(std::filesystem::path(enginePath)), logger)
    , mMaxSequenceLength(maxSequenceLength)
    , mHiddenSize(hiddenSize)
    , mVocabSize(vocabSize)
    , mMaxBatchSize(maxBatchSize)
    , mRotaryDim(rotaryDim)
    , mMaxDraftLength(maxDraftLength)
    , mMergeDraftBatches(!common::getBoolEnv("TRTLLM_QWEN35_MTP_DISABLE_DRAFT_BATCHING"))
{
    auto const& engine = mRuntime.getEngine();
    TLLM_CHECK(maxDraftLength >= 1 && maxDraftLength <= 30);
    TLLM_CHECK_WITH_INFO(
        maxDraftLength == 1 || engine.getTensorIOMode("mtp_hidden_states") == nvinfer1::TensorIOMode::kOUTPUT,
        "Rebuild the MTP draft engine with hidden-state output for K > 1");
    TLLM_CHECK_WITH_INFO(engine.getNbOptimizationProfiles() == 1, "Native MTP requires a single-profile draft engine");
    auto const offsetsShape
        = engine.getProfileShape("spec_decoding_position_offsets", 0, nvinfer1::OptProfileSelector::kMAX);
    TLLM_CHECK_WITH_INFO(offsetsShape.nbDims == 2 && offsetsShape.d[1] >= maxDraftLength + 1,
        "Rebuild the MTP draft engine for the target maximum draft length");
    mPagedKv = engine.getTensorIOMode("kv_cache_block_offsets") == nvinfer1::TensorIOMode::kINPUT;
    TLLM_CHECK_WITH_INFO(engine.getTensorIOMode("target_hidden_states") == nvinfer1::TensorIOMode::kINPUT
            && (mPagedKv || engine.getTensorIOMode("past_key_value_0") == nvinfer1::TensorIOMode::kINPUT),
        "Native MTP requires a continuous or paged KV draft engine");
    auto const hiddenShape = engine.getTensorShape("target_hidden_states");
    TLLM_CHECK_WITH_INFO(hiddenShape.nbDims == 2 && hiddenShape.d[1] == hiddenSize
            && engine.getTensorDataType("target_hidden_states") == nvinfer1::DataType::kBF16,
        "Native MTP draft hidden-state dimensions or dtype differ from target");
    if (mPagedKv)
    {
        // Paged attention exposes pointers, not the allocation dimensions. Keep
        // its geometry alongside the serialized engine, like the target config.
        std::ifstream configFile(enginePath + ".json");
        TLLM_CHECK_WITH_INFO(configFile.good(), "Paged MTP requires %s.json", enginePath.c_str());
        auto const config = nlohmann::json::parse(configFile);
        TLLM_CHECK(config.at("version").get<int>() == 1 && config.at("dtype").get<std::string>() == "bfloat16");
        auto const blockSize = config.at("tokens_per_block").get<SizeType32>();
        auto const heads = config.at("num_kv_heads").get<SizeType32>();
        auto const headSize = config.at("head_size").get<SizeType32>();
        TLLM_CHECK(blockSize > 0 && (blockSize & (blockSize - 1)) == 0 && heads > 0 && headSize > 0);
        TLLM_CHECK(config.at("max_seq_len").get<SizeType32>() >= maxSequenceLength
            && config.at("max_batch_size").get<SizeType32>() >= maxBatchSize);
        mBlocksPerSlot = (maxSequenceLength + blockSize - 1) / blockSize;
        auto const shape = engine.getProfileShape("kv_cache_block_offsets", 0, nvinfer1::OptProfileSelector::kMAX);
        TLLM_CHECK(shape.nbDims == 4 && shape.d[0] == 1 && shape.d[1] >= maxBatchSize && shape.d[2] == 2
            && shape.d[3] >= mBlocksPerSlot);
        auto& manager = mRuntime.getBufferManager();
        mKvPool = manager.gpu(ITensor::makeShape({static_cast<ITensor::DimType64>(maxBatchSize) * mBlocksPerSlot, 2,
                                  heads, blockSize, headSize}),
            nvinfer1::DataType::kBF16);
        mPoolPointers = BufferManager::pinned(ITensor::makeShape({1, 2}), nvinfer1::DataType::kINT64);
        bufferCast<std::int64_t>(*mPoolPointers)[0] = reinterpret_cast<std::int64_t>(mKvPool->data());
        bufferCast<std::int64_t>(*mPoolPointers)[1] = 0;
        mPoolMapping = BufferManager::pinned(ITensor::makeShape({1, 2}), nvinfer1::DataType::kINT32);
        std::fill_n(bufferCast<SizeType32>(*mPoolMapping), 2, 0);
        for (SizeType32 slot = maxBatchSize; slot > 0; --slot)
        {
            mFreeSlots.push_back(slot - 1);
        }
    }
    else
    {
        auto const kvShape = engine.getProfileShape("past_key_value_0", 0, nvinfer1::OptProfileSelector::kMAX);
        TLLM_CHECK_WITH_INFO(kvShape.nbDims == 5 && kvShape.d[0] >= maxBatchSize && kvShape.d[3] >= maxSequenceLength,
            "Native MTP draft cache must cover the target maximum sequence length, and runtime batch size");
    }
    for (int i = 0; i < engine.getNbIOTensors(); ++i)
    {
        if (std::string(engine.getIOTensorName(i)) == "last_token_logits")
        {
            mLastTokenLogits = true;
        }
    }
    auto const* logitsName = mLastTokenLogits ? "last_token_logits" : "logits";
    auto const logitsShape = engine.getTensorShape(logitsName);
    TLLM_CHECK_WITH_INFO(logitsShape.nbDims == 2 && logitsShape.d[1] == vocabSize
            && engine.getTensorDataType(logitsName) == nvinfer1::DataType::kFLOAT,
        "Native MTP requires FP32 draft logits with the target vocabulary");
    mRuntime.addContext(0);
}

Qwen35MtpWorker::~Qwen35MtpWorker()
{
    if (mPending)
    {
        mReady.synchronize();
    }
}

void Qwen35MtpWorker::waitReady(CudaStream const& stream) const
{
    if (mPending)
    {
        stream.wait(mReady);
    }
}

Qwen35MtpWorker::TensorPtr const& Qwen35MtpWorker::candidate(std::uint64_t requestId) const
{
    return mRequests.at(requestId).candidate;
}

void Qwen35MtpWorker::release(std::uint64_t requestId)
{
    // Cancellation or context reinitialization can bypass the next target wait.
    if (mPending && mRequests.count(requestId))
    {
        mReady.synchronize();
    }
    auto const it = mRequests.find(requestId);
    if (mPagedKv && it != mRequests.end() && it->second.slot >= 0)
    {
        mFreeSlots.push_back(it->second.slot);
    }
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
        if (mPagedKv)
        {
            TLLM_CHECK_WITH_INFO(!mFreeSlots.empty(), "Native MTP draft KV slots exhausted");
        }
        auto& state = mRequests.emplace(requestId, RequestState{}).first->second;
        if (mPagedKv)
        {
            state.slot = mFreeSlots.back();
            mFreeSlots.pop_back();
        }
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

void Qwen35MtpWorker::queue(std::uint64_t requestId, Tokens tokens, SizeType32 draftLength)
{
    auto& state = mRequests.at(requestId);
    TLLM_CHECK(state.tokens.empty());
    TLLM_CHECK(!tokens.empty() && (state.context || static_cast<SizeType32>(tokens.size()) <= mMaxDraftLength + 1));
    TLLM_CHECK(draftLength >= 1 && draftLength <= mMaxDraftLength);
    state.draftLength = draftLength;
    state.tokens = std::move(tokens);
}

std::map<std::uint64_t, TokenIdType> Qwen35MtpWorker::draft()
{
    auto const ids = draftDevice();
    auto& manager = mRuntime.getBufferManager();
    std::vector<TensorPtr> tokens;
    for (auto id : ids)
    {
        tokens.emplace_back(manager.copyFrom(*candidate(id), MemoryType::kCPU));
    }
    manager.getStream().synchronize();
    std::map<std::uint64_t, TokenIdType> result;
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        result.emplace(ids[i], bufferCast<TokenIdType const>(*tokens[i])[0]);
    }
    return result;
}

std::vector<std::uint64_t> Qwen35MtpWorker::draftDevice()
{
    // The target normally consumes the previous candidates before reaching
    // here. Standalone callers must also finish reads of reusable host staging.
    if (mPending)
    {
        auto const status = cudaEventQuery(mReady.get());
        if (status == cudaErrorNotReady)
        {
            mReady.synchronize();
        }
        else
        {
            TLLM_CUDA_CHECK(status);
        }
    }
    std::vector<std::uint64_t> result;
    // Generation uses a common causal window, padding after each valid prefix.
    // Fall back to equal-length groups if padding would exceed any request's KV capacity.
    SizeType32 width = 1;
    for (auto const& [id, state] : mRequests)
    {
        if (!state.context)
        {
            width = std::max(width, static_cast<SizeType32>(state.tokens.size()));
        }
    }
    bool const merge = mMergeDraftBatches
        && std::all_of(mRequests.begin(), mRequests.end(),
            [this, width](auto const& entry)
            {
                auto const& state = entry.second;
                return state.context || state.tokens.empty() || state.length <= mMaxSequenceLength - width;
            });
    // Context stays packed-ragged and separate from generation.
    for (SizeType32 group = 0; group < mMaxDraftLength + 2; ++group)
    {
        std::vector<RequestState*> states;
        std::vector<std::uint64_t> ids;
        for (auto& [id, state] : mRequests)
        {
            auto const requestGroup = state.context ? 0 : (merge ? 1 : static_cast<SizeType32>(state.tokens.size()));
            if (!state.tokens.empty() && requestGroup == group)
            {
                states.push_back(&state);
                ids.push_back(id);
            }
        }
        if (!states.empty())
        {
            draftBatch(states, 0, group);
            std::vector<SizeType32> committedLengths;
            for (auto* state : states)
            {
                committedLengths.push_back(state->length);
            }
            for (SizeType32 depth = 1; depth < mMaxDraftLength; ++depth)
            {
                std::vector<RequestState*> active;
                for (auto* state : states)
                {
                    if (depth < state->draftLength)
                    {
                        state->context = false;
                        state->tokens = {0};
                        state->inputToken = ITensor::slice(state->candidate, depth - 1, 1);
                        active.push_back(state);
                    }
                }
                if (!active.empty())
                {
                    draftBatch(active, depth, group);
                }
            }
            for (std::size_t i = 0; i < states.size(); ++i)
            {
                // Only target-conditioned catch-up rows belong to the committed draft cache.
                states[i]->length = committedLengths[i];
                states[i]->hiddenStates.reset();
                states[i]->rotaryCache.reset();
                states[i]->inputToken.reset();
            }
            result.insert(result.end(), ids.begin(), ids.end());
        }
    }
    mRuntime.getStream().record(mReady);
    mPending = true;
    return result;
}

void Qwen35MtpWorker::draftBatch(std::vector<RequestState*> const& states, SizeType32 depth, SizeType32 group)
{
    auto const batchSize = static_cast<SizeType32>(states.size());
    TLLM_CHECK(batchSize > 0 && batchSize <= mMaxBatchSize);
    auto const context = states.front()->context;
    SizeType32 width = 1;
    if (!context)
    {
        for (auto const* state : states)
        {
            width = std::max(width, static_cast<SizeType32>(state->tokens.size()));
        }
    }
    auto& manager = mRuntime.getBufferManager();
    auto const& engine = mRuntime.getEngine();
    auto& workspace = mWorkspaces[group * mMaxDraftLength + depth];
    workspace.retained.clear();
    std::size_t hostIndex = 0, deviceIndex = 0;
    auto acquire = [&](bool gpu, nvinfer1::Dims const& shape, nvinfer1::DataType type) -> TensorPtr
    {
        auto& buffers = gpu ? workspace.device : workspace.host;
        auto& index = gpu ? deviceIndex : hostIndex;
        if (index == buffers.size())
        {
            buffers.emplace_back(gpu ? manager.gpu(shape, type) : BufferManager::pinned(shape, type));
        }
        auto const& buffer = buffers[index++];
        TLLM_CHECK(buffer->getDataType() == type);
        buffer->reshape(shape);
        return buffer;
    };
    auto ints = [&](std::vector<SizeType32> const& values, nvinfer1::Dims const& shape, bool gpu) -> TensorPtr
    {
        auto host = acquire(false, shape, nvinfer1::DataType::kINT32);
        TLLM_CHECK(host->getSize() == values.size());
        std::copy(values.begin(), values.end(), bufferCast<SizeType32>(*host));
        if (!gpu)
        {
            return host;
        }
        auto device = acquire(true, shape, nvinfer1::DataType::kINT32);
        manager.copy(*host, *device);
        return device;
    };
    auto scalar = [&](SizeType32 value, bool gpu) { return ints({value}, ITensor::makeShape({1}), gpu); };
    auto batchInts = [&](std::vector<SizeType32> const& values, bool gpu)
    { return ints(values, ITensor::makeShape({batchSize}), gpu); };

    Tokens tokens;
    std::vector<SizeType32> pastLengths, promptLengths, sequenceLengths, lastTokenIds, positionDeltas;
    for (auto const* state : states)
    {
        auto const count = static_cast<SizeType32>(state->tokens.size());
        TLLM_CHECK(state->context == context);
        auto const physicalCount = context ? count : width;
        TLLM_CHECK(state->length + physicalCount <= mMaxSequenceLength);
        TLLM_CHECK(state->hiddenStates->getShape().d[0] >= count);
        tokens.insert(tokens.end(), state->tokens.begin(), state->tokens.end());
        pastLengths.push_back(state->length);
        promptLengths.push_back(context ? count : state->promptLength);
        sequenceLengths.push_back(state->length + physicalCount);
        lastTokenIds.push_back(static_cast<SizeType32>(tokens.size()));
        // Sampling reads the last valid row, before this request's trailing padding.
        tokens.insert(tokens.end(), physicalCount - count, 0);
        positionDeltas.push_back(state->positionDelta);
    }
    auto const numTokens = static_cast<SizeType32>(tokens.size());
    TensorPtr kv;
    if (!mPagedKv)
    {
        auto kvShape = engine.getProfileShape("past_key_value_0", 0, nvinfer1::OptProfileSelector::kMAX);
        kvShape.d[0] = batchSize;
        kvShape.d[3] = mMaxSequenceLength;
        kv = acquire(true, kvShape, nvinfer1::DataType::kBF16);
    }
    TensorPtr hidden = acquire(true, ITensor::makeShape({numTokens, mHiddenSize}), nvinfer1::DataType::kBF16);
    if (!context)
    {
        manager.setZero(*hidden);
    }
    auto ropeShape = engine.getProfileShape("mrope_rotary_cos_sin", 0, nvinfer1::OptProfileSelector::kMAX);
    ropeShape.d[0] = batchSize;
    TensorPtr rope = acquire(true, ropeShape, nvinfer1::DataType::kFLOAT);
    if (context && !mPagedKv)
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
        else if (!mPagedKv)
        {
            TLLM_CHECK(state.kv);
            manager.copy(*state.kv, *ITensor::slice(kv, i, 1));
        }
        offset += context ? count : width;
    }
    TensorPtr knobs = acquire(false, ITensor::makeShape({16}), nvinfer1::DataType::kINT64);
    std::fill_n(bufferCast<std::int64_t>(*knobs), 16, -1);
    TensorPtr progress = acquire(false, ITensor::makeShape({1}), nvinfer1::DataType::kINT64);
    bufferCast<std::int64_t>(*progress)[0] = 0;
    TensorPtr indirection
        = acquire(true, ITensor::makeShape({batchSize, 1, mMaxSequenceLength}), nvinfer1::DataType::kINT32);
    manager.setZero(*indirection);
    std::vector<SizeType32> offsets(batchSize * width), mask(batchSize * width);
    for (SizeType32 i = 0; i < batchSize * width; ++i)
    {
        offsets[i] = i % width;
        mask[i] = (1U << (i % width + 1)) - 1;
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
    if (depth > 0)
    {
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            manager.copy(*states[i]->inputToken, *ITensor::slice(allInputs.at("input_ids"), i, 1));
        }
    }
    if (mPagedKv)
    {
        std::vector<SizeType32> blocks(batchSize * 2 * mBlocksPerSlot);
        for (SizeType32 i = 0; i < batchSize; ++i)
        {
            for (SizeType32 kvIndex = 0; kvIndex < 2; ++kvIndex)
            {
                for (SizeType32 block = 0; block < mBlocksPerSlot; ++block)
                {
                    blocks[(i * 2 + kvIndex) * mBlocksPerSlot + block]
                        = (states[i]->slot * mBlocksPerSlot + block) * 2 + kvIndex;
                }
            }
        }
        auto const shape = ITensor::makeShape({1, batchSize, 2, mBlocksPerSlot});
        allInputs.emplace("kv_cache_block_offsets", ints(blocks, shape, true));
        allInputs.emplace("host_kv_cache_block_offsets", ints(blocks, shape, false));
        allInputs.emplace("host_kv_cache_pool_pointers", mPoolPointers);
        allInputs.emplace("host_kv_cache_pool_mapping", mPoolMapping);
    }
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
    auto& outputs = workspace.outputs;
    mRuntime.setOutputTensors(0, outputs);
    TLLM_CHECK_WITH_INFO(mRuntime.executeContext(0), "Native MTP draft engine enqueue failed");
    TensorPtr selectedTokens = acquire(true, ITensor::makeShape({batchSize}), nvinfer1::DataType::kINT32);
    auto samplingRows = allInputs.at("last_token_ids");
    if (mLastTokenLogits)
    {
        auto const& logits = outputs.at("last_token_logits");
        TLLM_CHECK(logits->getShape().d[0] == batchSize);
        std::vector<SizeType32> rows(batchSize);
        std::iota(rows.begin(), rows.end(), 1);
        samplingRows = batchInts(rows, true);
    }
    kernels::invokeMTPPackedGreedySampling(
        bufferCast<float const>(*outputs.at(mLastTokenLogits ? "last_token_logits" : "logits")),
        bufferCast<SizeType32 const>(*samplingRows), bufferCast<TokenIdType>(*selectedTokens), batchSize, mVocabSize,
        manager.getStream().get());
    for (SizeType32 i = 0; i < batchSize; ++i)
    {
        auto& state = *states[i];
        if (!mPagedKv)
        {
            auto const outputKv = ITensor::slice(outputs.at("present_key_value_0"), i, 1);
            if (!state.kv)
            {
                state.kv = manager.gpu(outputKv->getShape(), outputKv->getDataType());
            }
            manager.copy(*outputKv, *state.kv);
        }
        if (!state.candidate)
        {
            state.candidate = manager.gpu(ITensor::makeShape({mMaxDraftLength}), nvinfer1::DataType::kINT32);
        }
        state.candidate->reshape(ITensor::makeShape({state.draftLength}));
        manager.copy(*ITensor::slice(selectedTokens, i, 1), *ITensor::slice(state.candidate, depth, 1));
        // Keep target outputs alive until the asynchronous draft has consumed them.
        workspace.retained.push_back(state.hiddenStates);
        workspace.retained.push_back(state.rotaryCache);
        // Padding KV is tentative: the next append overwrites it at the logical length.
        state.length += static_cast<SizeType32>(state.tokens.size());
        state.promptLength = promptLengths[i];
        state.tokens.clear();
        if (depth + 1 < state.draftLength)
        {
            auto nextHidden = manager.gpu(ITensor::makeShape({1, mHiddenSize}), nvinfer1::DataType::kBF16);
            manager.copy(*ITensor::slice(outputs.at("mtp_hidden_states"), lastTokenIds[i] - 1, 1), *nextHidden);
            state.hiddenStates = std::move(nextHidden);
        }
    }
}
} // namespace tensorrt_llm::batch_manager
