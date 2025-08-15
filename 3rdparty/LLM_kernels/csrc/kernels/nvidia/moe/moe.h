/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 * Copyright 2025 vLLM Team
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/09e56f92620908d3cf1c3020336460f0db8beead/csrc/moe/moe_ops.h
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Vectorization size used in the sgl_moe_align_block_size kernel
#define SGL_MOE_ALIGN_BLOCK_VEC_SIZE 4

namespace llm_kernels {
namespace nvidia {

template <typename T, typename TOKEN_CNTS_T, bool UseExpertParallel>
void InvokeMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* experts_ids,
                             int32_t* total_tokens_post_pad, const int32_t* expert_map, const int32_t topk,
                             const int32_t num_experts, const int32_t expert_para_size, const int32_t block_size,
                             const size_t numel, const int32_t rank, const cudaStream_t& stream);

template <typename T>
void InvokeSglMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
                                int32_t* total_tokens_post_pad, const int32_t max_num_tokens_padded,
                                const int32_t num_experts, const int32_t block_size, const size_t numel,
                                int32_t* cumsum, const cudaStream_t& stream);

template <typename T>
void InvokeDynamicPerTokenScaledFP8Quant(const T* input, const float* scale, const void* scale_ub, void* output,
                                         const int hidden_size, const int num_tokens, const cudaStream_t& stream);

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream);

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream);

template <typename T, bool UseExpertParallel>
void InvokeMoeSum(void* input,      // [num_tokens, topk, hidden_size]
                  void* output,     // [num_tokens, hidden_size]
                  void* topk_ids,   // [..., topk]
                  int* expert_map,  // [expert_nums]
                  int num_tokens, const int num_experts, int topk, int hidden_size, const cudaStream_t& stream);

template <typename T, bool UseExpertParallel>
void SiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                size_t elements_num, size_t inter_size, const cudaStream_t& stream);
template <typename T>
void FlashinferSiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                          size_t elements_num, size_t inter_size, const cudaStream_t& stream);
template <typename T, bool UseExpertParallel>
void InvokeSiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                      size_t elements_num, size_t inter_size, const cudaStream_t& stream);

template <typename T>
void InvokeWeightDequant(const uint8_t* x, const float* s, T* output, int M, int N, int block_size,
                         const cudaStream_t& stream);

void InvokeMapExpertIds(const int* expert_map, int* expert_ids, int total_num);

void InvokeFillIntToBuffer(int* output, void* fill_info, int* fill_info_on_host, int fill_info_length,
                           const cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
