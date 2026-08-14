/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/cudaDriverWrapper.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace tensorrt_llm::plugins
{

struct GatedDeltaRuleDecodeParams
{
    void const* query;
    void const* key;
    void const* value;
    void const* logDecay;
    void const* beta;
    void* output;
    void* state;
    int32_t const* stateSlotMapping;
    int32_t const* cuSeqLens;
    int32_t batchSize;
};

class GatedDeltaRuleDecodeRunner
{
public:
    GatedDeltaRuleDecodeRunner(int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim);
    ~GatedDeltaRuleDecodeRunner() = default;

    GatedDeltaRuleDecodeRunner(GatedDeltaRuleDecodeRunner const&) = delete;
    GatedDeltaRuleDecodeRunner& operator=(GatedDeltaRuleDecodeRunner const&) = delete;
    GatedDeltaRuleDecodeRunner(GatedDeltaRuleDecodeRunner&&) = delete;
    GatedDeltaRuleDecodeRunner& operator=(GatedDeltaRuleDecodeRunner&&) = delete;

    void run(GatedDeltaRuleDecodeParams const& params, cudaStream_t stream) const;

private:
    int32_t mNumVHeads;
    int32_t mHeadKDim;
    int32_t mHeadVDim;
    int32_t mSharedMemoryBytes;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    CUfunction mFunction{};
};

} // namespace tensorrt_llm::plugins
