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

#include "gatedDeltaRulePrefillRunner.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/workspace.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace tensorrt_llm::plugins
{
namespace
{

constexpr int32_t kChunkSize = 64;
constexpr int32_t kL2NormBlockTokens = 16;
constexpr int32_t kInitChunkBlockSize = 256;
constexpr int32_t kStateBlockV = 32;
constexpr int32_t kIoBlockV = 64;
constexpr int32_t kOutputBlockV = 64;
constexpr int32_t kDefaultMaxDynamicSharedMemory = 48 * 1024;
constexpr unsigned int kMaxGridYZ = 65'535;

CUfunction loadKernel(
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> const& driver, GatedDeltaRulePrefillCubin const& cubin)
{
    static std::mutex mutex;
    static std::unordered_map<CUcontext, std::unordered_map<unsigned char const*, std::pair<CUmodule, CUfunction>>>
        kernels;
    std::lock_guard<std::mutex> const lock(mutex);
    CUcontext context{};
    TLLM_CU_CHECK(cuCtxGetCurrent(&context));
    TLLM_CHECK_WITH_INFO(context != nullptr, "GatedDeltaRule prefill runner requires a current CUDA context");
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
        if (cubin.sharedMemoryBytes > kDefaultMaxDynamicSharedMemory)
        {
            TLLM_CU_CHECK(driver->cuFuncSetAttribute(
                function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, cubin.sharedMemoryBytes));
        }
    }
    catch (...)
    {
        (void) cuModuleUnload(module);
        throw;
    }
    contextKernels.emplace(cubin.data, std::make_pair(module, function));
    return function;
}

size_t ceilDiv(size_t numerator, size_t denominator)
{
    return numerator / denominator + static_cast<size_t>(numerator % denominator != 0);
}

} // namespace

GatedDeltaRulePrefillRunner::GatedDeltaRulePrefillRunner(
    int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim)
    : mNumQHeads(numQHeads)
    , mNumVHeads(numVHeads)
    , mHeadKDim(headKDim)
    , mHeadVDim(headVDim)
    , mDriver(tensorrt_llm::common::CUDADriverWrapper::getInstance())
{
#if defined(_WIN32)
    TLLM_THROW("GatedDeltaRule Triton prefill runner is not supported on Windows");
#else
    TLLM_CHECK_WITH_INFO(numQHeads == 16 && (numVHeads == 16 || numVHeads == 32 || numVHeads == 48) && headKDim == 128
            && headVDim == 128,
        "GatedDeltaRule Triton prefill supports H=16, HV in {16, 32, 48}, K=128, and V=128");
    mL2Norm = load(GatedDeltaRulePrefillKernel::kL2Norm, 0);
    mInitChunks = load(GatedDeltaRulePrefillKernel::kInitChunks, 0);
    mPrepareChunks = load(GatedDeltaRulePrefillKernel::kPrepareChunks, 0);
    mZeroState = load(GatedDeltaRulePrefillKernel::kZeroState, numVHeads);
    mGatherState = load(GatedDeltaRulePrefillKernel::kGatherState, numVHeads);
    mCumsum = load(GatedDeltaRulePrefillKernel::kCumsum, numVHeads);
    mKktSolve = load(GatedDeltaRulePrefillKernel::kKktSolve, numVHeads);
    mRecompute = load(GatedDeltaRulePrefillKernel::kRecompute, numVHeads);
    mState = load(GatedDeltaRulePrefillKernel::kState, numVHeads);
    mOutput = load(GatedDeltaRulePrefillKernel::kOutput, numVHeads);
#endif
}

GatedDeltaRulePrefillRunner::Kernel GatedDeltaRulePrefillRunner::load(
    GatedDeltaRulePrefillKernel kernel, int32_t numVHeads) const
{
    auto const sm = tensorrt_llm::common::getSMVersion(/*queryRealSmArch=*/true);
    auto const* cubin = findGatedDeltaRulePrefillCubin(sm, kernel, numVHeads);
    TLLM_CHECK_WITH_INFO(cubin != nullptr, "No GatedDeltaRule prefill cubin for SM%d, kernel=%d, HV=%d", sm,
        static_cast<int32_t>(kernel), numVHeads);
    return {loadKernel(mDriver, *cubin), cubin->sharedMemoryBytes, cubin->numWarps};
}

size_t GatedDeltaRulePrefillRunner::getWorkspaceSize(
    int32_t totalTokens, int32_t numRequests, int32_t numQHeads, int32_t numVHeads, int32_t headKDim, int32_t headVDim)
{
    TLLM_CHECK_WITH_INFO(totalTokens > 0, "GatedDeltaRule prefill total token count must be positive");
    TLLM_CHECK_WITH_INFO(numRequests > 0, "GatedDeltaRule prefill request count must be positive");
    TLLM_CHECK_WITH_INFO(numQHeads == 16 && (numVHeads == 16 || numVHeads == 32 || numVHeads == 48) && headKDim == 128
            && headVDim == 128,
        "GatedDeltaRule Triton prefill supports H=16, HV in {16, 32, 48}, K=128, and V=128");
    TLLM_CHECK_WITH_INFO(totalTokens <= std::numeric_limits<int32_t>::max() / numQHeads,
        "GatedDeltaRule prefill normalization row count exceeds INT32_MAX");

    auto const tokens = static_cast<size_t>(totalTokens);
    auto const requests = static_cast<size_t>(numRequests);
    auto const maxChunks = requests + ceilDiv(tokens, static_cast<size_t>(kChunkSize));
    auto const requestHeads = requests * static_cast<size_t>(numVHeads);
    TLLM_CHECK_WITH_INFO(maxChunks <= kMaxGridYZ, "GatedDeltaRule prefill chunk grid exceeds CUDA grid.y limit");
    TLLM_CHECK_WITH_INFO(
        requestHeads <= kMaxGridYZ, "GatedDeltaRule prefill request-head grid exceeds CUDA grid.y limit");
    constexpr size_t kActivationBytes = sizeof(uint16_t);
    constexpr size_t kStateBytes = sizeof(float);
    constexpr size_t kIndexBytes = sizeof(int32_t);
    std::array<size_t, 11> const workspaces{
        tokens * numQHeads * headKDim * kActivationBytes,
        tokens * numQHeads * headKDim * kActivationBytes,
        tokens * numVHeads * kStateBytes,
        tokens * numVHeads * kChunkSize * kActivationBytes,
        tokens * numVHeads * headKDim * kActivationBytes,
        tokens * numVHeads * headVDim * kActivationBytes,
        maxChunks * numVHeads * headKDim * headVDim * kActivationBytes,
        maxChunks * 2 * kIndexBytes,
        requests * kIndexBytes,
        kIndexBytes,
        requests * sizeof(int8_t),
    };
    return tensorrt_llm::common::calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
}

void GatedDeltaRulePrefillRunner::run(GatedDeltaRulePrefillParams const& params, cudaStream_t stream) const
{
    TLLM_CHECK_WITH_INFO(params.totalTokens > 0, "GatedDeltaRule prefill total token count must be positive");
    TLLM_CHECK_WITH_INFO(params.numRequests > 0, "GatedDeltaRule prefill request count must be positive");
    TLLM_CHECK(params.workspace != nullptr);
    TLLM_CHECK(params.state != nullptr);

    (void) getWorkspaceSize(params.totalTokens, params.numRequests, mNumQHeads, mNumVHeads, mHeadKDim, mHeadVDim);

    auto const tokens = static_cast<size_t>(params.totalTokens);
    auto const requests = static_cast<size_t>(params.numRequests);
    auto const maxChunks = requests + ceilDiv(tokens, static_cast<size_t>(kChunkSize));
    constexpr size_t kActivationBytes = sizeof(uint16_t);
    constexpr size_t kStateBytes = sizeof(float);
    constexpr size_t kIndexBytes = sizeof(int32_t);
    std::array<size_t, 11> const workspaceSizes{
        tokens * mNumQHeads * mHeadKDim * kActivationBytes,
        tokens * mNumQHeads * mHeadKDim * kActivationBytes,
        tokens * mNumVHeads * kStateBytes,
        tokens * mNumVHeads * kChunkSize * kActivationBytes,
        tokens * mNumVHeads * mHeadKDim * kActivationBytes,
        tokens * mNumVHeads * mHeadVDim * kActivationBytes,
        maxChunks * mNumVHeads * mHeadKDim * mHeadVDim * kActivationBytes,
        maxChunks * 2 * kIndexBytes,
        requests * kIndexBytes,
        kIndexBytes,
        requests * sizeof(int8_t),
    };

    auto* workspace = static_cast<int8_t*>(params.workspace);
    uintptr_t offset{};
    auto next = [&](size_t size) { return tensorrt_llm::common::nextWorkspacePtr(workspace, offset, size); };
    CUdeviceptr normalizedQuery = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[0]));
    CUdeviceptr normalizedKey = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[1]));
    CUdeviceptr gCumsum = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[2]));
    CUdeviceptr a = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[3]));
    CUdeviceptr w = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[4]));
    CUdeviceptr u = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[5]));
    CUdeviceptr h = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[6]));
    CUdeviceptr chunkIndices = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[7]));
    CUdeviceptr chunkOffsets = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[8]));
    CUdeviceptr chunkCounter = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[9]));
    CUdeviceptr hasInitialState = reinterpret_cast<CUdeviceptr>(next(workspaceSizes[10]));

    CUdeviceptr query = reinterpret_cast<CUdeviceptr>(params.query);
    CUdeviceptr key = reinterpret_cast<CUdeviceptr>(params.key);
    CUdeviceptr value = reinterpret_cast<CUdeviceptr>(params.value);
    CUdeviceptr logDecay = reinterpret_cast<CUdeviceptr>(params.logDecay);
    CUdeviceptr beta = reinterpret_cast<CUdeviceptr>(params.beta);
    CUdeviceptr output = reinterpret_cast<CUdeviceptr>(params.output);
    CUdeviceptr state = reinterpret_cast<CUdeviceptr>(params.state);
    CUdeviceptr finalState = reinterpret_cast<CUdeviceptr>(params.finalState);
    CUdeviceptr stateSlotMapping = reinterpret_cast<CUdeviceptr>(params.stateSlotMapping);
    CUdeviceptr cuSeqLens = reinterpret_cast<CUdeviceptr>(params.cuSeqLens);
    int32_t totalTokens = params.totalTokens;
    int32_t numRequests = params.numRequests;
    int32_t maxChunksValue = static_cast<int32_t>(maxChunks);
    int64_t stateStride = static_cast<int64_t>(mNumVHeads) * mHeadVDim * mHeadKDim;
    float epsilon = 1e-6F;
    float scale = 1.0F / std::sqrt(static_cast<float>(mHeadKDim));
    CUdeviceptr globalScratch{};
    CUdeviceptr profileScratch{};

    TLLM_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(hasInitialState), params.hostHasInitialState,
        workspaceSizes[10], cudaMemcpyHostToDevice, stream));
    TLLM_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(chunkCounter), 0, workspaceSizes[9], stream));
    TLLM_CUDA_CHECK(cudaMemsetAsync(reinterpret_cast<void*>(a), 0, workspaceSizes[3], stream));

    auto launch
        = [&](Kernel const& kernel, unsigned int gridX, unsigned int gridY, unsigned int gridZ, void** kernelParams)
    {
        TLLM_CU_CHECK(mDriver->cuLaunchKernel(kernel.function, gridX, gridY, gridZ,
            static_cast<unsigned int>(kernel.numWarps * 32), 1, 1, static_cast<unsigned int>(kernel.sharedMemoryBytes),
            stream, kernelParams, nullptr));
    };

    void* initChunksParams[]{&chunkIndices, &maxChunksValue, &globalScratch, &profileScratch};
    launch(mInitChunks, static_cast<unsigned int>(ceilDiv(maxChunks, kInitChunkBlockSize)), 1, 1, initChunksParams);

    void* prepareChunksParams[]{
        &cuSeqLens, &chunkIndices, &chunkOffsets, &chunkCounter, &numRequests, &globalScratch, &profileScratch};
    launch(mPrepareChunks, static_cast<unsigned int>(numRequests), 1, 1, prepareChunksParams);

    void* zeroStateParams[]{&state, &stateSlotMapping, &hasInitialState, &numRequests, &globalScratch, &profileScratch};
    launch(mZeroState, static_cast<unsigned int>(ceilDiv(mHeadVDim, kIoBlockV)),
        static_cast<unsigned int>(numRequests * mNumVHeads), 1, zeroStateParams);

    int32_t normRows = static_cast<int32_t>(static_cast<size_t>(totalTokens) * mNumQHeads);
    void* normQueryParams[]{&query, &normalizedQuery, &epsilon, &normRows, &globalScratch, &profileScratch};
    launch(mL2Norm, static_cast<unsigned int>(ceilDiv(normRows, kL2NormBlockTokens)), 1, 1, normQueryParams);
    void* normKeyParams[]{&key, &normalizedKey, &epsilon, &normRows, &globalScratch, &profileScratch};
    launch(mL2Norm, static_cast<unsigned int>(ceilDiv(normRows, kL2NormBlockTokens)), 1, 1, normKeyParams);

    void* cumsumParams[]{&logDecay, &gCumsum, &cuSeqLens, &chunkIndices, &totalTokens, &globalScratch, &profileScratch};
    launch(mCumsum, static_cast<unsigned int>(maxChunks), static_cast<unsigned int>(mNumVHeads), 1, cumsumParams);

    void* kktSolveParams[]{
        &normalizedKey, &gCumsum, &beta, &a, &cuSeqLens, &chunkIndices, &totalTokens, &globalScratch, &profileScratch};
    launch(mKktSolve, static_cast<unsigned int>(maxChunks), static_cast<unsigned int>(mNumVHeads), 1, kktSolveParams);

    void* recomputeParams[]{&normalizedKey, &value, &beta, &w, &u, &a, &gCumsum, &cuSeqLens, &chunkIndices,
        &totalTokens, &globalScratch, &profileScratch};
    launch(mRecompute, static_cast<unsigned int>(maxChunks), static_cast<unsigned int>(mNumVHeads), 1, recomputeParams);

    void* stateParams[]{&normalizedKey, &u, &w, &u, &gCumsum, &h, &state, &stateSlotMapping, &cuSeqLens, &chunkOffsets,
        &totalTokens, &stateStride, &globalScratch, &profileScratch};
    launch(mState, static_cast<unsigned int>(ceilDiv(mHeadVDim, kStateBlockV)),
        static_cast<unsigned int>(numRequests * mNumVHeads), 1, stateParams);

    void* outputParams[]{&normalizedQuery, &normalizedKey, &u, &h, &gCumsum, &output, &cuSeqLens, &chunkIndices, &scale,
        &totalTokens, &globalScratch, &profileScratch};
    launch(mOutput, static_cast<unsigned int>(ceilDiv(mHeadVDim, kOutputBlockV)), static_cast<unsigned int>(maxChunks),
        static_cast<unsigned int>(mNumVHeads), outputParams);

    if (params.pagedState)
    {
        void* gatherStateParams[]{
            &state, &finalState, &stateSlotMapping, &numRequests, &globalScratch, &profileScratch};
        launch(mGatherState, static_cast<unsigned int>(ceilDiv(mHeadVDim, kIoBlockV)),
            static_cast<unsigned int>(numRequests * mNumVHeads), 1, gatherStateParams);
    }
}

} // namespace tensorrt_llm::plugins
