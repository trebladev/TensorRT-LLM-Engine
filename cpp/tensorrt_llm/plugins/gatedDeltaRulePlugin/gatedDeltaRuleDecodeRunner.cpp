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

#include "gatedDeltaRuleDecodeRunner.h"

#include "gatedDeltaRuleDecodeCubins.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cmath>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace tensorrt_llm::plugins
{
namespace
{

CUfunction loadKernel(
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver, GatedDeltaRuleDecodeCubin const& cubin)
{
    static std::mutex mutex;
    static std::unordered_map<CUcontext, std::unordered_map<unsigned char const*, std::pair<CUmodule, CUfunction>>>
        kernels;
    std::lock_guard<std::mutex> const lock(mutex);
    CUcontext context{};
    TLLM_CU_CHECK(cuCtxGetCurrent(&context));
    TLLM_CHECK_WITH_INFO(context != nullptr, "GatedDeltaRule decode runner requires a current CUDA context");
    auto& contextKernels = kernels[context];
    auto const found = contextKernels.find(cubin.data);
    if (found != contextKernels.end())
    {
        return found->second.second;
    }

    CUmodule module{};
    TLLM_CU_CHECK(driver->cuModuleLoadData(&module, cubin.data));
    CUfunction function{};
    try
    {
        TLLM_CU_CHECK(driver->cuModuleGetFunction(&function, module, cubin.kernelName));
    }
    catch (...)
    {
        (void) cuModuleUnload(module);
        throw;
    }
    contextKernels.emplace(cubin.data, std::make_pair(module, function));
    return function;
}

} // namespace

GatedDeltaRuleDecodeRunner::GatedDeltaRuleDecodeRunner(
    int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim)
    : mNumVHeads(numVHeads)
    , mHeadKDim(headKDim)
    , mHeadVDim(headVDim)
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
{
#if defined(_WIN32)
    TLLM_THROW("GatedDeltaRule Triton decode runner is not supported on Windows");
#else
    auto const sm = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true);
    auto const* cubin = findGatedDeltaRuleDecodeCubin(sm, numQHeads, numVHeads, headKDim, headVDim);
    TLLM_CHECK_WITH_INFO(cubin != nullptr, "No GatedDeltaRule decode cubin for SM%d with H=%d, HV=%d, K=%d, V=%d", sm,
        numQHeads, numVHeads, headKDim, headVDim);

    mFunction = loadKernel(mDriver, *cubin);
    mSharedMemoryBytes = cubin->sharedMemoryBytes;
#endif
}

void GatedDeltaRuleDecodeRunner::run(GatedDeltaRuleDecodeParams const& params, cudaStream_t stream) const
{
    TLLM_CHECK_WITH_INFO(params.batchSize > 0, "GatedDeltaRule decode batch size must be positive");

    CUdeviceptr query = reinterpret_cast<CUdeviceptr>(params.query);
    CUdeviceptr key = reinterpret_cast<CUdeviceptr>(params.key);
    CUdeviceptr value = reinterpret_cast<CUdeviceptr>(params.value);
    CUdeviceptr logDecay = reinterpret_cast<CUdeviceptr>(params.logDecay);
    CUdeviceptr beta = reinterpret_cast<CUdeviceptr>(params.beta);
    CUdeviceptr output = reinterpret_cast<CUdeviceptr>(params.output);
    CUdeviceptr state = reinterpret_cast<CUdeviceptr>(params.state);
    CUdeviceptr stateSlotMapping = reinterpret_cast<CUdeviceptr>(params.stateSlotMapping);
    CUdeviceptr cuSeqLens = reinterpret_cast<CUdeviceptr>(params.cuSeqLens);
    float scale = 1.0F / std::sqrt(static_cast<float>(mHeadKDim));
    CUdeviceptr intermediateStatesBuffer{};
    int32_t cacheSteps{};
    int32_t totalTokens = params.batchSize;
    CUdeviceptr globalScratch{};
    CUdeviceptr profileScratch{};

    void* kernelParams[] = {&query, &key, &value, &logDecay, &beta, &output, &state, &stateSlotMapping, &cuSeqLens,
        &scale, &intermediateStatesBuffer, &cacheSteps, &totalTokens, &globalScratch, &profileScratch};

    constexpr unsigned int kGridX = 1U;
    constexpr unsigned int kBlockX = 32U;
    constexpr unsigned int kBlockY = 1U;
    constexpr unsigned int kBlockZ = 1U;
    auto const gridY = static_cast<unsigned int>((mHeadVDim + 7) / 8);
    auto const gridZ = static_cast<unsigned int>(params.batchSize * mNumVHeads);
    TLLM_CU_CHECK(mDriver->cuLaunchKernel(mFunction, kGridX, gridY, gridZ, kBlockX, kBlockY, kBlockZ,
        static_cast<unsigned int>(mSharedMemoryBytes), stream, kernelParams, nullptr));
}

} // namespace tensorrt_llm::plugins
