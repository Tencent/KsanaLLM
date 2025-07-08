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

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#include "csrc/utils/nvidia/cuda_utils.h"
using namespace llm_kernels::utils;

namespace llm_kernels {
namespace nvidia {
namespace marlin_moe {

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];

  int32_t* tokens_cnts = shared_mem;                              // 2d tensor with shape (blockDim.x + 1, num_experts)
  int32_t* cumsum = shared_mem + (blockDim.x + 1) * num_experts;  // 1d tensor with shape (num_experts + 1)

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    ++tokens_cnts[index(num_experts, threadIdx.x + 1, topk_ids[i])];
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  if (threadIdx.x < num_experts) {
    tokens_cnts[index(num_experts, 0, threadIdx.x)] = 0;
    for (int i = 1; i <= blockDim.x; ++i) {
      tokens_cnts[index(num_experts, i, threadIdx.x)] += tokens_cnts[index(num_experts, i - 1, threadIdx.x)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] + CEILDIV(tokens_cnts[index(num_experts, blockDim.x, i - 1)], block_size) * block_size;
    }
    if (total_tokens_post_pad) {
      *total_tokens_post_pad = cumsum[num_experts];
    }
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (expert_ids && threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }

  /**
   * Each thread processes a token shard, calculating the index of each token
   * after sorting by expert number. Given the example topk_ids =
   * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6, *,
   * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
   * padding value(preset in python).
   */
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    /** The cumsum[expert_id] stores the starting index of the tokens that the
     * expert with expert_id needs to process, and
     * tokens_cnts[threadIdx.x][expert_id] stores the indices of the tokens
     * processed by the expert with expert_id within the current thread's token
     * shard.
     */
    int32_t rank_post_pad = tokens_cnts[index(num_experts, threadIdx.x, expert_id)] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[index(num_experts, threadIdx.x, expert_id)];
  }
}

template <typename scalar_t>
__global__ void fill_kernel(scalar_t* data, scalar_t value, int num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < num) {
    data[idx] = value;
  }
}

// https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/fused_moe/fused_moe.py#L177
/*
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.
    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and E = 4:
    - We initially have 12 tokens (after repeating 'topk' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
*/
void moe_align_block_size(
    int* topk_ids,          // int[total_tokens, topk] representing the top-k expert indices for each token.
    int* sorted_token_ids,  // int[max_num_tokens_padded], containing the sorted token indices according to allocated
                            // expert. max_num_tokens_padded = M * topk + E * (block_size - 1);
    int* experts_ids, int* num_tokens_post_pad, int num_experts, int num_tokens, int topk, int max_num_tokens_padded,
    int block_size,  // The block size used in block matrix multiplication.
    cudaStream_t& stream) {
  int grid = ((max_num_tokens_padded + DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM - 1) / DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);
  fill_kernel<int><<<grid, DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM, 0, stream>>>(sorted_token_ids, num_tokens * topk,
                                                                             max_num_tokens_padded);
  // calc needed amount of shared mem for `tokens_cnts` and `cumsum`
  // tensors
  const int32_t num_thread = max((int32_t)num_experts, DEFAULT_CUDA_WARP_SIZE);
  const int32_t shared_mem = ((num_thread + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);

  // set dynamic shared mem
  auto kernel = moe_align_block_size_kernel<int>;
  CHECK_NVIDIA_CUDA_ERROR(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem));
  kernel<<<1, num_thread, shared_mem, stream>>>(topk_ids, sorted_token_ids, experts_ids, num_tokens_post_pad,
                                                num_experts, block_size, num_tokens * topk);
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(scalar_t* __restrict__ out,          // [..., d]
                               const scalar_t* __restrict__ input,  // [..., topk, d]
                               const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += __ldg(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename T>
void moe_sum(T* output,       // [num_tokens, hidden_size]
             const T* input,  // [num_tokens, topk, hidden_size]
             int num_tokens, int topk, int hidden_size, cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  switch (topk) {
    case 2:
      moe_sum_kernel<T, 2><<<grid, block, 0, stream>>>(output, input, hidden_size);
      break;

    case 3:
      moe_sum_kernel<T, 3><<<grid, block, 0, stream>>>(output, input, hidden_size);
      break;

    case 4:
      moe_sum_kernel<T, 4><<<grid, block, 0, stream>>>(output, input, hidden_size);
      break;

    default:
      KLLM_KERNEL_CHECK_WITH_INFO(topk <= 4, "moe topk > 4 is not implemented.");
      break;
  }
}

template void moe_sum(half* output, const half* input, int num_tokens, int topk, int hidden_size, cudaStream_t& stream);

}  // namespace marlin_moe
}  // namespace nvidia
}  // namespace llm_kernels
