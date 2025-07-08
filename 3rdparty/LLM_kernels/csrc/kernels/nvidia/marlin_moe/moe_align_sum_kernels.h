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

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

void moe_align_block_size(
    int* topk_ids,          // int[total_tokens, topk] representing the top-k expert indices for each token.
    int* sorted_token_ids,  // int[max_num_tokens_padded], containing the sorted token indices according to allocated
                            // expert. max_num_tokens_padded = M * topk + E * (block_size - 1);
    int* experts_ids, int* num_tokens_post_pad, int num_experts, int num_tokens, int topk, int max_num_tokens_padded,
    int block_size,  // The block size used in block matrix multiplication.
    cudaStream_t& stream);

template <typename T>
void moe_sum(T* output,       // [num_tokens, hidden_size]
             const T* input,  // [num_tokens, topk, hidden_size]
             int num_tokens, int topk, int hidden_size, cudaStream_t& stream);

}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
