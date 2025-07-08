/*
 * Modified by Neural Magic
 * Copyright (C) Marlin.2024 Elias Frantar
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/vllm-project/vllm/tree/v0.6.4.post1
 */

#pragma once
#include "csrc/utils/nvidia/scalar_type.hpp"

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

void marlin_gemm_moe(void* output, const void* a, const void* b_q_weights, const void* sorted_ids,
                     const void* topk_weights, const void* topk_ids, const void* b_scales, const void* b_zeros,
                     const void* g_idx, const void* perm, void* workspace, void* expert_offsets, void* a_tmp,
                     const vllm_dtype::ScalarTypeId b_q_type_id, int64_t size_m, int64_t size_n, int64_t size_k,
                     bool is_k_full, int64_t num_experts, int64_t topk, int num_groups, int64_t moe_block_size,
                     bool replicate_input, bool apply_weights, cudaStream_t& stream);

}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
