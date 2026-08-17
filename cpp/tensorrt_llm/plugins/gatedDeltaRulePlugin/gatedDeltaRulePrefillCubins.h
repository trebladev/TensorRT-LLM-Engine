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

#include <cstdint>

namespace tensorrt_llm::plugins
{

enum class GatedDeltaRulePrefillKernel : int32_t
{
    kL2Norm,
    kInitChunks,
    kPrepareChunks,
    kZeroState,
    kGatherState,
    kCumsum,
    kKktSolve,
    kRecompute,
    kState,
    kOutput
};

struct GatedDeltaRulePrefillCubin
{
    int32_t sm;
    GatedDeltaRulePrefillKernel kernel;
    int32_t numVHeads;
    unsigned char const* data;
    unsigned int size;
    char const* kernelName;
    int32_t sharedMemoryBytes;
    int32_t numWarps;
};

GatedDeltaRulePrefillCubin const* findGatedDeltaRulePrefillCubin(
    int32_t sm, GatedDeltaRulePrefillKernel kernel, int32_t numVHeads);

} // namespace tensorrt_llm::plugins
