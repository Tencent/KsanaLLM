/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#pragma once

#include "csrc/utils/nvidia/cuda_bf16_wrapper.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "csrc/utils/nvidia/cuda_utils.h"

namespace llm_kernels {
namespace nvidia {


/**
 * @brief Extracts the hidden states of the some tokens for each sample in a batch
 *
 * This function efficiently extracts the hidden states of the some token for each sample in a batch
 * from the input tensor containing all token hidden states. It uses shared memory and vectorized
 * memory operations to optimize performance. The function selects the appropriate kernel based on
 * the size of hidden_units_num to maximize efficiency.
 *
 * @tparam T Data type (float, half, or __nv_bfloat16)
 * @param[in] input Pointer to the input tensor containing all token hidden states
 *                  Shape: [total_tokens, hidden_units_num]
 * @param[in] accepted_tokens_idx Array of indices pointing to the input token of each sample in the batch
 *                       Shape: [accepted_tokens_size]
 * @param[in] accepted_tokens_size Number of samples in the batch
 * @param[in] hidden_units_num Dimension of the hidden state vectors
 * @param[out] output Pointer to the output tensor where the extracted hidden states will be stored
 *                    Shape: [accepted_tokens_size, hidden_units_num]
 * @param[in] stream CUDA stream to execute the kernel on
 */
////// for example :
// input prompt_token_ids={{1,2},{3,4,5,6},{7,8,9}}
// input tensor shape =[2+4+3, 4096]
// input accepted_tokens_num={1,2,2}
// prompt 1 tensor shape [2, 4096]:
//  [[emb1],
//   [emb2]]
// prompt 2 tensor shape [4, 4096]:
//  [[emb3],
//   [emb4],
//   [emb5],
//   [emb6]]
// prompt 3 tensor shape [3, 4096]:
//  [[emb7],
//   [emb8],
//   [emb9]]
// of [embX] is values like [0.1, 0.2, ..., 409.6]
////// then the input params are:
// accepted_tokens_idx is {0, 2+0, 2+1, 6+0, 6+1}
// baccepted_tokens_size is 5
////// then the output tensor shape [1+2+2, 4096], and values should be:
// prompt 1 tensor shape [1, 4096]:
//  [[emb1]]
// prompt 2 tensor shape [2, 4096]:
//  [[emb3],
//   [emb4]]
// prompt 3 tensor shape [2, 4096]:
//  [[emb7],
//   [emb8]]
template <typename T>
void AssembleTokensHidden(const T* input, const size_t* accepted_tokens_idx, const int32_t accepted_tokens_size,
                          const int32_t hidden_units_num, T* output, cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
