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

#include "marlin_moe_kernel.h"

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

bool call_marlin_moe_kernel_ku8b128(vllm_dtype::ScalarType const& q_type, int thread_n_blocks, int thread_k_blocks,
                                    bool has_act_order, int group_blocks, int num_threads, int blocks,
                                    int max_shared_mem, cudaStream_t stream, const int4* A_ptr, const int4* B_ptr,
                                    int4* C_ptr, const int* sorted_ids_ptr, const float* topk_weights_ptr,
                                    const int4* s_ptr, const int4* zp_ptr, const int* g_idx_ptr,
                                    int* expert_offsets_ptr, int num_groups, int expert_idx, int num_experts, int topk,
                                    int prob_m, int prob_n, int prob_k, int tot_m, int* locks, bool replicate_input,
                                    bool apply_weights, int m_block, int max_par, int cfg_max_m_blocks);

}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
