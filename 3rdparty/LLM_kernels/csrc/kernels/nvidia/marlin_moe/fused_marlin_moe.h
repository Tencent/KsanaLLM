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

#include "csrc/kernels/nvidia/mixture_of_experts/moe_norm_config.h"
namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

size_t get_fused_marlin_moe_workspace_size(int M,     // num_tokens
                                           int N,     // inter_size
                                           int K,     // hidden_size
                                           int E,     // num_experts
                                           int topk,  // top-k
                                           size_t data_type_size);

// modify from
// https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/gptq_marlin.py#L525
// modify from
// https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/fused_moe/fused_marlin_moe.py#L146
// This function computes a Mixture of Experts (MoE) layer using two sets of
// weights, w1 and w2, and top-k gating mechanism.
// The input must currently be float16
void fused_marlin_moe(
    half* output,                // The output tensor after applying the MoE layer.
    const half* input,           // float16[M, K], The input tensor to the MoE layer, should be float16
    const float* gating_output,  // float[num_rows, E] The output of the gating operation (before softmax).
    const void* w1,              // int, [E, K / 16, 2 * N * (num_bits / 2)], The first set of expert weights.
    const void* w2,              // int, [E, N / 16, K * (num_bits / 2)] The second set of expert weights.
    const void* w1_scale,        // Scale to be used for w1.
    const void* w2_scale,        // Scale to be used for w2.
    void* workspace,             // The workspace buffer
    size_t workspace_size,       // The bytes of workspace
    int M,                       // num_tokens
    int N,                       // inter_size
    int K,                       // hidden_size
    int E,                       // num_experts
    int topk,                    // The topk
    cudaStream_t& stream,        // The stream
    MOEExpertScaleNormalizationMode norm_mode,  // The nomalize mode
    const int* g_idx1 = nullptr,                // The first set of act_order indices.
    const int* g_idx2 = nullptr,                // The second set of act_order indices.
    const void* sort_indices1 = nullptr,        // The first act_order input permutation.
    const void* sort_indices2 = nullptr,        // The second act_order input permutation.
    const void* w1_zeros = nullptr,             // Optional zero points to be used for w1.
    const void* w2_zeros = nullptr,             // Optional zero points to be used for w2.
    int num_bits = 8,                           // The number of bits in expert weights quantization.
    int group_size = 128                        // The group_size in expert weights quantization.
);

}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
