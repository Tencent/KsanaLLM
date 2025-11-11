/*
 * Copyright 2025 vLLM Team
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
 *
 * Adapted from
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/9798b2fb0052092a6420172e41c0c8a307eedfa6/csrc/quantization/cutlass_w8a8/c3x/cutlass_gemm_caller.cuh
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace llm_kernels {
namespace nvidia {

// Fast TopK selection kernel
// Parameters:
//   logits: [numRows, stride0] input logits (float*)
//   rowStarts: [numRows] start index for each row (int32_t*)
//   rowEnds: [numRows] end index for each row (int32_t*)
//   indices: [numRows, TopK] output indices (int32_t*)
//   values: [numRows, TopK] output values (float*)
//   numRows: number of rows
//   stride0: stride of input tensor dimension 0
//   stride1: stride of input tensor dimension 1
//   stream: CUDA stream
void InvokeFastTopK(const float* logits, const int32_t* rowStarts, const int32_t* rowEnds,
                    int32_t* indices, int64_t numRows,
                    int64_t stride0, int64_t stride1, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels