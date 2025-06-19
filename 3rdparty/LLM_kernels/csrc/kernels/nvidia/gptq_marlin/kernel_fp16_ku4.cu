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
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/tree/65334ef3b9e4fd32ebc5c4e512debc25d5025488/csrc/quantization/gptq_marlin
 */

#include "csrc/kernels/nvidia/gptq_marlin/kernel.h"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_template.h"

namespace llm_kernels {
namespace nvidia {
namespace marlin {

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, true, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, true, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, true, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 2, 16, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 8, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 4, 8, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 3, 16, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 8, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 4, 8, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 4, 16, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 8, 4, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 4, 8, false, pipe_stages, -1,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, true, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, true, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, true, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 2, 16, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 8, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 4, 8, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 3, 16, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 8, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 4, 8, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 4, 16, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 8, 4, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 4, 8, false, pipe_stages, 2,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, true, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, true, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, true, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, true, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, true, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, true, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 2, 16, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 2, 16, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 8, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 8, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 4, 8, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 4, 8, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 3, 16, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 3, 16, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 8, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 8, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 4, 8, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 4, 8, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 4, 16, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 4, 16, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 8, 4, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 8, 4, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 4, 8, false, pipe_stages, 4,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 4, 8, false, pipe_stages, 4,
                                true>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, true, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, true, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, true, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 1, 8, 8, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 8, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 1, 4, 8, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 2, 16, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 8, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 2, 4, 8, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 3, 16, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 8, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 3, 4, 8, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 256, 4, 16, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 8, 4, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

template __global__ void Marlin<half, llm_kernels::nvidia::vllm_dtype::kU4.id(), 128, 4, 4, 8, false, pipe_stages, 8,
                                false>(MARLIN_KERNEL_PARAMS);

}  // namespace marlin
}  // namespace nvidia
}  // namespace llm_kernels