/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <cuda_runtime.h>
#include <string>

namespace llm_kernels {
namespace nvidia {

/**
 * @brief Performs grouped top-k operation
 *
 * @tparam T Input data type (float, half, __nv_bfloat16)
 * @param gating_output Input gating output with shape [num_rows, num_experts]
 * @param e_bias Expert bias with shape [num_experts]
 * @param routed_scaling_factor scaling factor for routing
 * @param topk_weights_ptr Output top-k weights with shape [num_rows, topk]
 * @param topk_ids_ptr Output top-k indices with shape [num_rows, topk]
 * @param num_rows Number of input rows
 * @param num_experts Number of experts
 * @param topk Number of top-k elements to select
 * @param num_expert_group Number of expert groups
 * @param topk_group Number of groups to select for each sample
 * @param stream CUDA stream
 */
template <typename T>
void InvokeDeepSeekV3GroupedTopk(void* gating_output, void* e_bias, float routed_scaling_factor, void* topk_weights_ptr, void* topk_ids_ptr,
                                 int tokens_num, int num_experts, int topk, int num_expert_group, int topk_group,
                                 cudaStream_t stream);

/**
 * @brief Performs basic softmax + top-k operation without grouping
 *
 * @tparam T Input data type (float, half, __nv_bfloat16)
 * @param gating_output Input gating output with shape [num_rows, num_experts]
 * @param topk_weights_ptr Output top-k weights with shape [num_rows, topk]
 * @param topk_ids_ptr Output top-k indices with shape [num_rows, topk]
 * @param num_rows Number of input rows
 * @param num_experts Number of experts
 * @param topk Number of top-k elements to select
 * @param routed_scaling_factor scaling factor for routing
 * @param stream CUDA stream
 */
template <typename T>
void InvokeBasicSoftmaxTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows,
                            int num_experts, int topk, float routed_scaling_factor, cudaStream_t stream);
}  // namespace nvidia
}  // namespace llm_kernels