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

#include "csrc/kernels/nvidia/mixture_of_experts/moe_kernels.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "marlin_moe_ops.h"
#include "moe_align_sum_kernels.h"

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

size_t get_fused_marlin_moe_workspace_size(int M,     // num_tokens
                                           int N,     // inter_size
                                           int K,     // hidden_size
                                           int E,     // num_experts
                                           int topk,  // top-k
                                           size_t data_type_size) {
  int block_size_m = ((M <= E || M <= 32) ? 16 : 64);
  int max_num_tokens_padded = M * topk + E * (block_size_m - 1);

  size_t size = 0;
  // gemm1 output
  size_t intermediate_cache1_size = data_type_size * M * topk * 2 * N;
  size += intermediate_cache1_size;
  // activation output
  size_t intermediate_cache2_size = data_type_size * M * topk * N;
  size += intermediate_cache2_size;
  // gemm2 output
  size_t intermediate_cache3_size = data_type_size * M * topk * K;
  size += intermediate_cache3_size;
  // gemm input permute buffer
  size_t a_tmp_size = data_type_size * std::max(M * K, M * topk * N);
  size += a_tmp_size;
  // gemm1 output
  size_t expert_offsets_size = sizeof(int) * (E + 1);
  size += expert_offsets_size;
  // moe_align_block_size output
  size_t sorted_token_ids_size = sizeof(int) * max_num_tokens_padded;
  size += sorted_token_ids_size;
  size_t softmax_temp_output_size = sizeof(float) * ((!((E != 0) && ((E & (E - 1)) == 0)) || E > 256) ? M * E : 0);
  size += softmax_temp_output_size;
  // token_expert_indices Not used. Will be used in the future.
  size_t token_expert_indices_size = sizeof(int) * M * topk;
  size += token_expert_indices_size;
  size_t topk_ids_size = sizeof(int) * M * topk;
  size += topk_ids_size;
  size_t topk_weights = sizeof(float) * M * topk;
  size += topk_weights;
  // gemm workspace:
  size_t gemm_workspace_size = sizeof(int) * (max(2 * N, K) / 64) * 16;
  size += gemm_workspace_size;
  return size;
}

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
    const int* g_idx1,                          // The first set of act_order indices.
    const int* g_idx2,                          // The second set of act_order indices.
    const void* sort_indices1,                  // The first act_order input permutation.
    const void* sort_indices2,                  // The second act_order input permutation.
    const void* w1_zeros,                       // Optional zero points to be used for w1.
    const void* w2_zeros,                       // Optional zero points to be used for w2.
    int num_bits,                               // The number of bits in expert weights quantization.
    int group_size                              // The group_size in expert weights quantization.
) {
  typedef half T;
  size_t data_type_size = sizeof(T);
  bool is_k_full = true;
  KLLM_KERNEL_CHECK_WITH_INFO(get_fused_marlin_moe_workspace_size(M, N, K, E, topk, data_type_size) <= workspace_size,
                              "workspace_size is not enough for moe.");

  KLLM_KERNEL_CHECK_WITH_INFO(num_bits == 4 || num_bits == 8, "num_bits should be 4 or 8.");
  bool has_no_act_order =
      (g_idx1 == nullptr && g_idx2 == nullptr && sort_indices1 == nullptr && sort_indices2 == nullptr);
  bool has_all_act_order =
      (g_idx1 != nullptr && g_idx2 != nullptr && sort_indices1 != nullptr && sort_indices2 != nullptr);

  KLLM_KERNEL_CHECK_WITH_INFO(has_no_act_order || has_all_act_order,
                              "g_idx and sorted_indices must be all not None or must be all None.");

  bool has_no_zp = (w1_zeros == nullptr and w2_zeros == nullptr);
  bool has_all_zp = (w1_zeros != nullptr and w2_zeros != nullptr);
  KLLM_KERNEL_CHECK_WITH_INFO(has_no_zp || has_all_zp, "zero points must be both not None or must be both None");
  int block_size_m = ((M <= E || M <= 32) ? 16 : 64);

  int max_num_tokens_padded = M * topk + E * (block_size_m - 1);

  // gemm workspace:
  char* ws_ptr = reinterpret_cast<char*>(workspace);
  // gemm1 output
  T* intermediate_cache1 = reinterpret_cast<T*>(ws_ptr);
  size_t intermediate_cache1_size = data_type_size * M * topk * 2 * N;
  ws_ptr += intermediate_cache1_size;
  // activation output
  T* intermediate_cache2 = reinterpret_cast<T*>(ws_ptr);
  size_t intermediate_cache2_size = data_type_size * M * topk * N;
  ws_ptr += intermediate_cache2_size;
  // gemm2 output
  T* intermediate_cache3 = reinterpret_cast<T*>(ws_ptr);
  size_t intermediate_cache3_size = data_type_size * M * topk * K;
  ws_ptr += intermediate_cache3_size;
  // gemm input permute buffer
  T* a_tmp = reinterpret_cast<T*>(ws_ptr);
  size_t a_tmp_size = data_type_size * std::max(M * K, M * topk * N);
  ws_ptr += a_tmp_size;
  // gemm1 output
  int* expert_offsets = reinterpret_cast<int*>(ws_ptr);
  size_t expert_offsets_size = sizeof(int) * (E + 1);
  ws_ptr += expert_offsets_size;
  // moe_align_block_size output
  int* sorted_token_ids = reinterpret_cast<int*>(ws_ptr);
  size_t sorted_token_ids_size = sizeof(int) * max_num_tokens_padded;
  ws_ptr += sorted_token_ids_size;
  float* softmax_temp_output = reinterpret_cast<float*>(ws_ptr);
  size_t softmax_temp_output_size = sizeof(float) * ((!((E != 0) && ((E & (E - 1)) == 0)) || E > 256) ? M * E : 0);
  ws_ptr += softmax_temp_output_size;
  // token_expert_indices Not used. Will be used in the future.
  int* token_expert_indices = reinterpret_cast<int*>(ws_ptr);
  size_t token_expert_indices_size = sizeof(int) * M * topk;
  ws_ptr += token_expert_indices_size;
  // Indices of topk-k elements.
  int* topk_ids = reinterpret_cast<int*>(ws_ptr);
  size_t topk_ids_size = sizeof(int) * M * topk;
  ws_ptr += topk_ids_size;
  float* topk_weights = reinterpret_cast<float*>(ws_ptr);
  size_t topk_weights_size = sizeof(float) * M * topk;
  ws_ptr += topk_weights_size;
  size_t gemm_workspace_size = sizeof(int) * (max(2 * N, K) / 64) * 16;

  topkGatingSoftmaxKernelLauncher(gating_output, /*finished*/ nullptr, topk_weights, softmax_temp_output, topk_ids,
                                  token_expert_indices, M, E, topk, 0, E, norm_mode, stream);

  moe_align_block_size(topk_ids, sorted_token_ids, nullptr, nullptr, E, M, topk, max_num_tokens_padded, block_size_m,
                       stream);

  vllm_dtype::ScalarTypeId scalar_type_id;
  if (has_all_zp) {
    KLLM_KERNEL_CHECK_WITH_INFO(num_bits == 4, "when has_all_zp is true, num_bits should be 4.");
    scalar_type_id = vllm_dtype::kUint4.id();
  } else {
    if (num_bits == 4) {
      scalar_type_id = vllm_dtype::kUint4b8.id();
    } else {
      scalar_type_id = vllm_dtype::kUint8b128.id();
    }
  }

  cudaMemsetAsync(intermediate_cache1, 0, intermediate_cache1_size, stream);
  cudaMemsetAsync(ws_ptr, 0, gemm_workspace_size, stream);
  cudaMemsetAsync(a_tmp, 0, a_tmp_size, stream);
  marlin_gemm_moe(intermediate_cache1, input, w1, sorted_token_ids, topk_weights, topk_ids, w1_scale, w1_zeros, g_idx1,
                  sort_indices1, ws_ptr, expert_offsets, a_tmp, scalar_type_id, M, 2 * N, K, is_k_full, E, topk,
                  K / group_size, block_size_m, true, false, stream);

  doGatedActivation(intermediate_cache2, intermediate_cache1, nullptr, N, M, ActivationType::Swiglu, stream, false);

  T* gemm2_output = nullptr;
  if (topk > 1) {
    gemm2_output = intermediate_cache3;
  } else {
    gemm2_output = output;
  }
  cudaMemsetAsync(gemm2_output, 0, intermediate_cache3_size, stream);
  cudaMemsetAsync(ws_ptr, 0, gemm_workspace_size, stream);
  cudaMemsetAsync(a_tmp, 0, a_tmp_size, stream);
  marlin_gemm_moe(gemm2_output, intermediate_cache2, w2, sorted_token_ids, topk_weights, topk_ids, w2_scale, w2_zeros,
                  g_idx2, sort_indices2, ws_ptr, expert_offsets, a_tmp, scalar_type_id, M, K, N, is_k_full, E, topk,
                  N / group_size, block_size_m, false, true, stream);
  // reduce topk dim
  if (topk > 1) {
    moe_sum<T>(output, intermediate_cache3, M, topk, N, stream);
  }
}
}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
