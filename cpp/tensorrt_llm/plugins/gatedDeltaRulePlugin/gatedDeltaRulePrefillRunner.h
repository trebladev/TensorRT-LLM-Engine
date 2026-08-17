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

#include "gatedDeltaRulePrefillCubins.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tensorrt_llm::plugins
{

struct GatedDeltaRulePrefillParams
{
    void const* query;
    void const* key;
    void const* value;
    void const* logDecay;
    void const* beta;
    void* output;
    void* state;
    void* finalState;
    int32_t const* stateSlotMapping;
    int32_t const* cuSeqLens;
    int8_t const* hostHasInitialState;
    void* workspace;
    int32_t totalTokens;
    int32_t numRequests;
    bool pagedState;
};

class GatedDeltaRulePrefillRunner
{
public:
    GatedDeltaRulePrefillRunner(int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim);
    ~GatedDeltaRulePrefillRunner() = default;

    GatedDeltaRulePrefillRunner(GatedDeltaRulePrefillRunner const&) = delete;
    GatedDeltaRulePrefillRunner& operator=(GatedDeltaRulePrefillRunner const&) = delete;
    GatedDeltaRulePrefillRunner(GatedDeltaRulePrefillRunner&&) = delete;
    GatedDeltaRulePrefillRunner& operator=(GatedDeltaRulePrefillRunner&&) = delete;

    static size_t getWorkspaceSize(int32_t totalTokens, int32_t numRequests, int32_t numQHeads, int32_t numVHeads,
        int32_t headKDim, int32_t headVDim);
    void run(GatedDeltaRulePrefillParams const& params, cudaStream_t stream) const;

private:
    struct Kernel
    {
        CUfunction function{};
        int32_t sharedMemoryBytes{};
        int32_t numWarps{};
    };

    Kernel load(GatedDeltaRulePrefillKernel kernel, int32_t numVHeads) const;

    int32_t mNumQHeads;
    int32_t mNumVHeads;
    int32_t mHeadKDim;
    int32_t mHeadVDim;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    Kernel mL2Norm;
    Kernel mInitChunks;
    Kernel mPrepareChunks;
    Kernel mZeroState;
    Kernel mGatherState;
    Kernel mCumsum;
    Kernel mKktSolve;
    Kernel mRecompute;
    Kernel mState;
    Kernel mOutput;
};

} // namespace tensorrt_llm::plugins
