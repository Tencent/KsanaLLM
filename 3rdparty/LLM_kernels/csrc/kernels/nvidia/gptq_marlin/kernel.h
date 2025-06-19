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

#pragma once

#include "csrc/kernels/nvidia/gptq_marlin/marlin.cuh"
#include "csrc/kernels/nvidia/gptq_marlin/marlin_dtypes.cuh"
#include "csrc/utils/nvidia/scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin {

#define MARLIN_KERNEL_PARAMS                                                                                         \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C, int4 *__restrict__ C_tmp,            \
      const int4 *__restrict__ scales_ptr, const uint16_t *__restrict__ scale2_ptr, const int4 *__restrict__ zp_ptr, \
      const int *__restrict__ g_idx, int num_groups, int prob_m, int prob_n, int prob_k, int lda, int *locks,        \
      bool use_atomic_add, bool use_fp32_reduce, int max_shared_mem

template <typename scalar_t,                                              // compute dtype, half or nv_float16
          const llm_kernels::nvidia::vllm_dtype::ScalarTypeId w_type_id,  // weight ScalarType id
          const int threads,                                              // number of threads in a threadblock
          const int thread_m_blocks,                                      // number of 16x16 blocks in the m
                                                                          // dimension (batchsize) of the
                                                                          // threadblock
          const int thread_n_blocks,                                      // same for n dimension (output)
          const int thread_k_blocks,                                      // same for k dimension (reduction)
          const bool m_block_size_8,                                      // whether m_block_size == 8
                                                                          // only works when thread_m_blocks == 1
          const int stages,        // number of stages for the async global->shared
                                   // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}  // namespace marlin
}  // namespace nvidia
}  // namespace llm_kernels
