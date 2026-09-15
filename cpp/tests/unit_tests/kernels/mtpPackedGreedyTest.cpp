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
#include "tensorrt_llm/kernels/speculativeDecoding/mtpKernels.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <limits>
#include <random>

using namespace tensorrt_llm::runtime;

TEST(MtpPackedGreedy, MatchesFirstMaximumForPackedRows)
{
    auto stream = std::make_shared<CudaStream>();
    BufferManager manager(stream);
    constexpr int kRows = 9;
    constexpr int kBatchSize = 3;
    std::array<int, kBatchSize> const lastRows{2, 5, 9};
    auto hostRows = BufferManager::cpu(ITensor::makeShape({kBatchSize}), nvinfer1::DataType::kINT32);
    std::copy(lastRows.begin(), lastRows.end(), bufferCast<int>(*hostRows));
    auto deviceRows = manager.copyFrom(*hostRows, MemoryType::kGPU);
    auto output = manager.gpu(ITensor::makeShape({kBatchSize}), nvinfer1::DataType::kINT32);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-10.0F, 10.0F);
    for (int const vocabSize : {1, 3, 129, 256, 257, 248320})
    {
        auto logits = BufferManager::cpu(ITensor::makeShape({kRows, vocabSize}), nvinfer1::DataType::kFLOAT);
        auto* data = bufferCast<float>(*logits);
        for (int scenario = 0; scenario < 5; ++scenario)
        {
            std::generate_n(data, logits->getSize(), [&]() { return distribution(random); });
            std::array<int, kBatchSize> expected{};
            for (int request = 0; request < kBatchSize; ++request)
            {
                auto* row = data + (lastRows[request] - 1) * vocabSize;
                if (scenario == 1)
                {
                    // Equal maxima across reduction lanes must select the lowest token ID.
                    row[vocabSize / 3] = 20.0F;
                    row[vocabSize - 1] = 20.0F;
                }
                else if (scenario == 2)
                {
                    std::fill_n(row, vocabSize, -std::numeric_limits<float>::infinity());
                }
                else if (scenario == 3)
                {
                    row[0] = std::numeric_limits<float>::quiet_NaN();
                }
                else if (scenario == 4)
                {
                    row[vocabSize - 1] = std::numeric_limits<float>::quiet_NaN();
                }
                expected[request] = static_cast<int>(std::max_element(row, row + vocabSize) - row);
            }
            auto deviceLogits = manager.copyFrom(*logits, MemoryType::kGPU);
            tensorrt_llm::kernels::invokeMTPPackedGreedySampling(bufferCast<float>(*deviceLogits),
                bufferCast<int>(*deviceRows), bufferCast<int>(*output), kBatchSize, vocabSize, stream->get());
            auto result = manager.copyFrom(*output, MemoryType::kCPU);
            stream->synchronize();
            for (int request = 0; request < kBatchSize; ++request)
            {
                EXPECT_EQ(bufferCast<int>(*result)[request], expected[request])
                    << "vocab=" << vocabSize << " scenario=" << scenario << " request=" << request;
            }
        }
    }
}

TEST(MtpStateCommit, CopiesSelectedRecordsAndPreservesOtherBytes)
{
    auto stream = std::make_shared<CudaStream>();
    BufferManager manager(stream);
    constexpr int kRecords = 6;
    constexpr int kSlotStride = 8192;
    constexpr int kBytes = kRecords * 3 * kSlotStride;
    auto host = BufferManager::cpu(ITensor::makeShape({kBytes}), nvinfer1::DataType::kINT8);
    auto* initial = bufferCast<int8_t>(*host);
    for (int i = 0; i < kBytes; ++i)
    {
        initial[i] = static_cast<int8_t>((i * 17 + i / kSlotStride) % 127);
    }
    auto addresses = BufferManager::pinned(ITensor::makeShape({2 * kRecords}), nvinfer1::DataType::kINT64);
    // Include a partial vector and a full combined recurrent/convolution record.
    for (int const bytes : {1, 15, 16, 273, kSlotStride / 2})
    {
        auto pool = manager.copyFrom(*host, MemoryType::kGPU);
        std::vector<int8_t> expected(initial, initial + kBytes);
        auto* pointers = bufferCast<int64_t>(*addresses);
        for (int i = 0; i < kRecords; ++i)
        {
            int const dst = (kRecords - 1 - i) * 3 * kSlotStride + i % 2;
            int const src = dst + (1 + i % 2) * kSlotStride;
            pointers[2 * i] = reinterpret_cast<int64_t>(bufferCast<int8_t>(*pool) + src);
            pointers[2 * i + 1] = reinterpret_cast<int64_t>(bufferCast<int8_t>(*pool) + dst);
            std::copy_n(initial + src, bytes, expected.begin() + dst);
        }
        auto deviceAddresses = manager.copyFrom(*addresses, MemoryType::kGPU);
        tensorrt_llm::kernels::invokeMTPCommitStateRecords(
            bufferCast<int64_t>(*deviceAddresses), kRecords, bytes, stream->get());
        auto result = manager.copyFrom(*pool, MemoryType::kCPU);
        stream->synchronize();
        EXPECT_TRUE(std::equal(expected.begin(), expected.end(), bufferCast<int8_t>(*result))) << "bytes=" << bytes;
    }
    tensorrt_llm::kernels::invokeMTPCommitStateRecords(nullptr, 0, kSlotStride, stream->get());
    stream->synchronize();
}
