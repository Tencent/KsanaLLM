/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llm_kernels {
namespace nvidia {
namespace moe_wna16 {

template <typename T>
void moe_wna16_gemm(cudaStream_t stream, void* output, const void* input, const void* b_qweight, const void* b_scales,
                    const void* b_qzeros, const void* topk_weights, const void* sorted_token_ids,
                    const void* expert_ids, const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M,
                    int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts, int size_m, int size_n, int size_k,
                    int group_size, int num_token_blocks);

}  // namespace moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels
