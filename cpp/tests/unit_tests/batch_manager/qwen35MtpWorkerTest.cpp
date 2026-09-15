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
#include "tensorrt_llm/batch_manager/qwen35MtpWorker.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/tllmLogger.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>

namespace tensorrt_llm::batch_manager
{
namespace
{
using namespace runtime;

TEST(Qwen35MtpWorkerTest, MixedExtensionsAtCapacity)
{
    auto const* directory = std::getenv("QWEN35_MTP_ENGINE_DIR");
    if (directory == nullptr)
    {
        GTEST_SKIP() << "Set QWEN35_MTP_ENGINE_DIR to a Qwen3.5-2B BF16 batch>=2 engine directory";
    }
    // Use the real draft engine with a deliberately small runtime KV allocation.
    // A valid one-token extension must succeed even when a peer needs two tokens.
    constexpr SizeType32 kCapacity = 4;
    constexpr SizeType32 kHiddenSize = 2048;
    constexpr SizeType32 kVocabSize = 248320;
    constexpr SizeType32 kRotaryDim = 64;
    TllmLogger logger;
    ASSERT_TRUE(initTrtLlmPlugins(&logger));
    Qwen35MtpWorker worker((std::filesystem::path(directory) / "mtp.engine").string(), &logger, kCapacity, kHiddenSize,
        kVocabSize, 2, kRotaryDim);
    BufferManager manager(std::make_shared<CudaStream>());
    auto queue = [&](std::uint64_t id, bool context, Qwen35MtpWorker::Tokens tokens)
    {
        auto const count = static_cast<SizeType32>(tokens.size());
        Qwen35MtpWorker::TensorPtr hidden
            = manager.gpu(ITensor::makeShape({count, kHiddenSize}), nvinfer1::DataType::kBF16);
        Qwen35MtpWorker::TensorPtr rope
            = manager.gpu(ITensor::makeShape({count * kRotaryDim}), nvinfer1::DataType::kFLOAT);
        manager.setZero(*hidden);
        manager.setZero(*rope);
        manager.getStream().synchronize();
        worker.capture(id, context, hidden, rope, 0);
        worker.queue(id, std::move(tokens));
    };
    queue(1, true, {11});
    queue(2, true, {17});
    ASSERT_EQ(worker.draft().size(), 2);
    queue(1, false, {12, 13});
    queue(2, false, {18});
    ASSERT_EQ(worker.draft().size(), 2);
    // Histories are now lengths 3 and 2. Padding both to width 2 would overflow
    // request 1, so this call must fall back to separate width-1/width-2 groups.
    queue(1, false, {14});
    queue(2, false, {19, 20});
    auto const result = worker.draft();
    ASSERT_EQ(result.size(), 2);
    for (auto const& [id, token] : result)
    {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, kVocabSize);
    }
    worker.release(1);
    worker.release(2);
    EXPECT_TRUE(worker.draft().empty());
    // Reusing an ID starts from context, without inheriting its full old cache.
    queue(1, true, {21});
    ASSERT_EQ(worker.draft().size(), 1);
}

// The two workers see identical packed inputs but allocate physical slots in
// opposite orders. Every candidate must survive page boundaries and slot reuse.
TEST(Qwen35MtpWorkerTest, SlotIsolationAcrossPages)
{
    auto const* directory = std::getenv("QWEN35_MTP_ENGINE_DIR");
    if (directory == nullptr)
    {
        GTEST_SKIP() << "Set QWEN35_MTP_ENGINE_DIR to a Qwen3.5-2B BF16 batch>=4 engine directory";
    }
    constexpr SizeType32 kCapacity = 129;
    constexpr SizeType32 kHiddenSize = 2048;
    constexpr SizeType32 kRotaryDim = 64;
    TllmLogger logger;
    ASSERT_TRUE(initTrtLlmPlugins(&logger));
    auto const path = (std::filesystem::path(directory) / "mtp.engine").string();
    Qwen35MtpWorker first(path, &logger, kCapacity, kHiddenSize, 248320, 4, kRotaryDim);
    Qwen35MtpWorker second(path, &logger, kCapacity, kHiddenSize, 248320, 4, kRotaryDim);
    BufferManager manager(std::make_shared<CudaStream>());
    auto queue = [&](Qwen35MtpWorker& worker, std::uint64_t id, bool context, SizeType32 count, SizeType32 step)
    {
        auto host = BufferManager::cpu(ITensor::makeShape({count, kHiddenSize}), nvinfer1::DataType::kBF16);
        // Deterministic BF16 values in [0.5, 1), varying by request and step.
        auto* bits = static_cast<std::uint16_t*>(host->data());
        for (std::size_t i = 0; i < host->getSize(); ++i)
        {
            bits[i] = static_cast<std::uint16_t>(0x3f00 + (i * 13 + id * 7 + step * 11) % 128);
        }
        Qwen35MtpWorker::TensorPtr hidden = manager.copyFrom(*host, MemoryType::kGPU);
        Qwen35MtpWorker::TensorPtr rope
            = manager.gpu(ITensor::makeShape({count * kRotaryDim}), nvinfer1::DataType::kFLOAT);
        manager.setZero(*rope);
        manager.getStream().synchronize();
        worker.capture(id, context, hidden, rope, 0);
        worker.queue(id, Qwen35MtpWorker::Tokens(count, static_cast<TokenIdType>(11 + id * 17 + step)));
    };
    for (std::uint64_t id : {1, 2, 3, 4})
    {
        queue(first, id, true, 30 + id % 3, 0);
    }
    for (std::uint64_t id : {4, 3, 2, 1})
    {
        queue(second, id, true, 30 + id % 3, 0);
    }
    EXPECT_EQ(first.draft(), second.draft());
    for (SizeType32 step = 1; step <= 65; ++step)
    {
        // Shrink the active batch, then reinitialize the same ID in a released
        // slot. Retained requests must keep their independent histories.
        if (step == 12)
        {
            first.release(2);
            second.release(2);
        }
        for (std::uint64_t id : {1, 2, 3, 4})
        {
            if (id == 2 && step >= 12 && step < 16)
            {
                continue;
            }
            auto const context = id == 2 && step == 16;
            auto const count = context ? 31 : 1 + (step + id) % 2;
            queue(first, id, context, count, step);
            queue(second, id, context, count, step);
        }
        EXPECT_EQ(first.draft(), second.draft()) << "step=" << step;
    }
}
} // namespace
} // namespace tensorrt_llm::batch_manager
