/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 * Copyright 2025 vLLM Team
 * Copyright (c) 2023 DeepSeek
 * Copyright 2023-2024 SGLang Team
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
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/09e56f92620908d3cf1c3020336460f0db8beead/csrc/moe/moe_align_sum_kernels.cu
 * [vLLM Project]
 * https://github.com/vllm-project/vllm/blob/v0.7.1/vllm/model_executor/layers/quantization/utils/quant_utils.py#L63
 * [DeepSeek-V3 Project] https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/kernel.py#L84
 * [SGLang Project] https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
 */
#include "moe.h"

#include <cub/cub.cuh>
#include "csrc/utils/nvidia/cuda_utils.h"
#include <flashinfer/activation.cuh>


#define CEILDIV(x, y) (((x) + (y)-1) / (y))

namespace llm_kernels {
namespace nvidia {

using FP8_TYPE = uint8_t;
constexpr float FP8_E4M3_MAX = 448.0f;

namespace {
__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row, int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}
}  // namespace

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
  float old;
  old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
                     : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
  return old;
}

// Vectorization containers
template <typename scalar_t>
struct __align__(8) vec4_t {
  scalar_t x;
  scalar_t y;
  scalar_t z;
  scalar_t w;
};

template <typename quant_type_t>
struct __align__(4) q8x4_t {
  static_assert(std::is_same_v<quant_type_t, int8_t> || std::is_same_v<quant_type_t, uint8_t>);
  quant_type_t x;
  quant_type_t y;
  quant_type_t z;
  quant_type_t w;
};

template <bool is_scale_inverted>
__device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val, float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
  return static_cast<FP8_TYPE>(r);
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(void* __restrict__ out, scalar_t const* __restrict__ input, float const scale,
                                          int64_t const num_elems, int const tid, int const step);

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                            int32_t* expert_ids, int32_t* total_tokens_post_pad, int32_t num_experts,
                                            int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts + 1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

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
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
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

template <typename scalar_t, typename token_cnts_t>
__global__ void moe_align_block_size_kernel_expert_parallel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                                            int32_t* expert_ids, int32_t* total_tokens_post_pad,
                                                            const int32_t* expert_map, int32_t topk,
                                                            int32_t num_experts, int32_t block_size, size_t numel) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;  // 1d tensor with shape (num_experts + 1)
  token_cnts_t* tokens_cnts =
      (token_cnts_t*)(shared_mem + num_experts + 1);  // 2d tensor with shape (blockDim.x + 1, num_experts)

  // 把共享内存从 threadIdx.x * num_experts 开始， expert_num 个数字置 0
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  /**
   * In the first step we compute token_cnts[thread_index + 1][expert_index],
   * which counts how many tokens in the token shard of thread_index are
   * assigned to expert expert_index.
   */
  // 统计所有的 topk_ids，将对应的位置计数 +n
  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = expert_map[topk_ids[i]];
    if (expert_id < num_experts) {
      ++tokens_cnts[index(num_experts, threadIdx.x + 1, expert_id)];
    }
  }

  __syncthreads();

  // For each expert we accumulate the token counts from the different threads.
  // 我们用不同的 thread，来累加不同的专家
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
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
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
    int32_t expert_id = expert_map[topk_ids[i]];
    if (expert_id >= num_experts) {
      continue;
    }
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

// TODO(simon): this is temporarily adapted from
// https://github.com/sgl-project/sglang/commit/31548116a8dc8c6df7e146e0587335a59fc5b9d7
// we did this to unblock Deepseek V3 but there should be a better
// implementation to manage shared memory.
template <typename scalar_t>
__global__ void moe_align_block_size_global_mem_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                                       int32_t* expert_ids, int32_t* total_tokens_post_pad,
                                                       int32_t num_experts, int32_t block_size, size_t numel,
                                                       int32_t* tokens_cnts, int32_t* cumsum) {
  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

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
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  if (threadIdx.x < num_experts) {
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

// taken from
// https://github.com/sgl-project/sglang/pull/3347/files
template <typename scalar_t>
__global__ void sgl_moe_token_sort_kernel(scalar_t* __restrict__ topk_ids, int* sorted_token_ids, int* cumsum,
                                          size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i];
    int32_t rank_post_pad = atomicAdd(&cumsum[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

// taken from
// https://github.com/sgl-project/sglang/commit/ded9fcd09a43d5e7d5bb31a2bc3e9fc21bf65d2a
template <typename scalar_t>
__global__ void sgl_moe_align_block_size_kernel(scalar_t* __restrict__ topk_ids, int32_t* sorted_token_ids,
                                                int32_t* expert_ids, int32_t* total_tokens_post_pad,
                                                int32_t num_experts, int32_t block_size, size_t numel,
                                                int32_t* cumsum) {
  __shared__ int32_t shared_counts[32][8];

  const int warp_id = threadIdx.x / 32;
  const int experts_per_warp = 8;
  const int my_expert_start = warp_id * experts_per_warp;

  for (int i = 0; i < experts_per_warp; ++i) {
    if (my_expert_start + i < num_experts) {
      shared_counts[warp_id][i] = 0;
    }
  }

  const size_t tokens_per_thread = CEILDIV(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int expert_id = topk_ids[i];
    int warp_idx = expert_id / experts_per_warp;
    int expert_offset = expert_id % experts_per_warp;
    atomicAdd(&shared_counts[warp_idx][expert_offset], 1);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      int expert_count = 0;
      int warp_idx = (i - 1) / experts_per_warp;
      int expert_offset = (i - 1) % experts_per_warp;
      expert_count = shared_counts[warp_idx][expert_offset];

      cumsum[i] = cumsum[i - 1] + CEILDIV(expert_count, block_size) * block_size;
    }
    *total_tokens_post_pad = cumsum[num_experts];
  }

  __syncthreads();

  if (threadIdx.x < num_experts) {
    for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1]; i += block_size) {
      expert_ids[i / block_size] = threadIdx.x;
    }
  }
}

template <typename scalar_t>
__global__ void dynamic_per_token_scaled_fp8_quant_kernel(FP8_TYPE* __restrict__ out, float* __restrict__ scale,
                                                          scalar_t const* __restrict__ input,
                                                          float const* __restrict__ scale_ub, const int hidden_size) {
  float const min_scaling_factor = 1.0f / (FP8_E4M3_MAX * 512.f);

  int const tid = threadIdx.x;
  int const token_idx = blockIdx.x;

  // Use int64 to avoid overflowing an int32 when calculating this offset
  int64_t offset = static_cast<int64_t>(token_idx) * hidden_size;
  scalar_t const* __restrict__ token_input = &input[offset];
  FP8_TYPE* __restrict__ token_output = &out[offset];

  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  float absmax_val = 0.0f;
  if (can_vectorize) {
    absmax_val = thread_max_vec(token_input, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float const x = static_cast<float>(token_input[i]);
      absmax_val = max(absmax_val, fabs(x));
    }
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe = BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float token_scale;
  if (tid == 0) {
    if (scale_ub) {
      token_scale = min(block_absmax_val_maybe, *scale_ub);
    } else {
      token_scale = block_absmax_val_maybe;
    }
    // token scale computation
    token_scale = max(token_scale / FP8_E4M3_MAX, min_scaling_factor);
    scale[token_idx] = token_scale;
  }
  __syncthreads();

  // Note that we don't use inverted scales so we can match FBGemm impl.
  if (can_vectorize) {
    scaled_fp8_conversion_vec<scalar_t, false>(token_output, token_input, token_scale, hidden_size, tid, blockDim.x);
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      token_output[i] = scaled_fp8_conversion<false>(static_cast<float>(token_input[i]), token_scale);
    }
  }
}

template <typename scalar_t>
__global__ void segmented_max_reduction(float* __restrict__ scale, const scalar_t* __restrict__ input,
                                        int64_t num_elems) {
  __shared__ float cache[1024];
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;

  // First store maximum for all values processes by
  // the current thread in cache[threadIdx.x]
  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }
  cache[threadIdx.x] = tmp;

  __syncthreads();

  // Now perform parallel reduction within the thread block
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (threadIdx.x < ib && cache[threadIdx.x + ib] > cache[threadIdx.x]) {
      cache[threadIdx.x] = cache[threadIdx.x + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // Finally, since cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (threadIdx.x == 0) {
    atomicMaxFloat(scale, cache[0] / FP8_E4M3_MAX);
  }
}

template <typename scalar_t, bool is_scale_inverted>
__device__ void scaled_fp8_conversion_vec(FP8_TYPE* __restrict__ out, scalar_t const* __restrict__ input,
                                          float const scale, int64_t const num_elems, int const tid, int const step) {
  using float8x4_t = q8x4_t<FP8_TYPE>;
  // Vectorized input/output to better utilize memory bandwidth.
  auto const* vectorized_in = reinterpret_cast<vec4_t<scalar_t> const*>(input);
  auto* vectorized_out = reinterpret_cast<float8x4_t*>(out);

  int64_t const num_vec_elems = num_elems >> 2;

#pragma unroll 4
  for (int64_t i = tid; i < num_vec_elems; i += step) {
    vec4_t<scalar_t> in_vec = vectorized_in[i];
    float8x4_t out_vec;

    out_vec.x = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.x), scale);
    out_vec.y = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.y), scale);
    out_vec.z = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.z), scale);
    out_vec.w = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(in_vec.w), scale);
    vectorized_out[i] = out_vec;
  }

  // Handle the remaining elements if num_elems is not divisible by 4
  for (int64_t i = num_vec_elems * 4 + tid; i < num_elems; i += step) {
    out[i] = scaled_fp8_conversion<is_scale_inverted>(static_cast<float>(input[i]), scale);
  }
}

template <typename scalar_t>
__global__ void scaled_fp8_quant_kernel(FP8_TYPE* __restrict__ out, const scalar_t* __restrict__ input,
                                        const float* __restrict__ scale, int64_t num_elems) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  // Invert the scale so that we can use multiplications to avoid expensive
  // division.
  const float inverted_scale = 1.0f / (*scale);
  scaled_fp8_conversion_vec<scalar_t, true>(out, input, inverted_scale, num_elems, tid, blockDim.x * gridDim.x);
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel(scalar_t* __restrict__ out,          // [..., d]
                               const scalar_t* __restrict__ input,  // [..., topk, d]
                               const int num_experts, const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      x += *(&input[token_idx * TOPK * d + k * d + idx]);
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_kernel_expert_parallel(scalar_t* __restrict__ out,          // [..., d]
                                               const scalar_t* __restrict__ input,  // [..., topk, d]
                                               const int* __restrict__ topk_ids,    // [..., topk]
                                               const int* __restrict__ expert_map,  // [expert_nums]
                                               const int num_experts, const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    scalar_t x = 0.0;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      if (expert_map[topk_ids[token_idx * TOPK + k]] < num_experts) {
        x += *(&input[token_idx * TOPK * d + k * d + idx]);
      }
    }
    out[token_idx * d + idx] = x;
  }
}

template <typename scalar_t>
__global__ void sum_out_dim1_kernel(scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
                                    const int num_tokens, const int topk, const int hidden_size) {
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < num_tokens && col < hidden_size) {
    scalar_t sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
      sum += input[row * topk * hidden_size + k * hidden_size + col];
    }
    out[row * hidden_size + col] = sum;
  }
}

template <typename T>
__global__ void silu_and_mul_kernel(const T* x, T* out, size_t size, size_t inter_size) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= size) return;
  for (int64_t idx = threadIdx.x; idx < inter_size; idx += blockDim.x) {
    float a = *(&x[token_idx * 2 * inter_size + idx]);
    float b = *(&x[token_idx * 2 * inter_size + inter_size + idx]);
    out[token_idx * inter_size + idx] = a / (1.0f + expf(-a)) * b;
  }
}

template <typename T>
__global__ void silu_and_mul_kernel_expert_parallel(const T* x, T* out, const int* topk_ids, const int* expert_map,
                                                    const int num_experts, size_t size, size_t inter_size) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= size) return;
  if (expert_map[topk_ids[token_idx]] >= num_experts) return;
  for (int64_t idx = threadIdx.x; idx < inter_size; idx += blockDim.x) {
    float a = *(&x[token_idx * 2 * inter_size + idx]);
    float b = *(&x[token_idx * 2 * inter_size + inter_size + idx]);
    out[token_idx * inter_size + idx] = a / (1.0f + expf(-a)) * b;
  }
}

__device__ float fp8e4m3_to_float(uint8_t fp8) {
  uint8_t sign = (fp8 >> 7) & 0x1;
  uint8_t exponent = (fp8 >> 3) & 0xF;
  uint8_t mantissa = fp8 & 0x7;

  // 计算符号
  float sign_value = sign ? -1.0f : 1.0f;

  // 计算指数值
  int exponent_value = exponent - 7;  // FP8E4M3的指数偏移量为7

  // 计算尾数值
  float mantissa_value = 1.0f + (mantissa / 8.0f);

  // 组合符号、指数和尾数得到最终的float值
  // printf("%f * %f * 2^ %d\n", sign_value, mantissa_value, exponent_value);
  float result = sign_value * mantissa_value * std::pow(2.0f, exponent_value);

  return result;
}

template <typename T>
__global__ void weight_dequant_kernel(const uint8_t* data, const float* scale, T* output, int M, int N,
                                      int BLOCK_SIZE) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < static_cast<size_t>(M) * N) {
    size_t m = idx / N;
    size_t n = idx % N;
    float s = scale[m / BLOCK_SIZE * N / BLOCK_SIZE + n / BLOCK_SIZE];
    float w = fp8e4m3_to_float(data[idx]);
    float result = w * s;
    if constexpr (std::is_same<T, half>::value) {
      output[idx] = __float2half(result);
    } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      output[idx] = __float2bfloat16(result);
    }
  }
}

__global__ void map_expert_ids_kernel(const int* expert_map, int* expert_idx, int total_nums) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < total_nums) {
    int idx = expert_idx[tid];
    expert_idx[tid] = expert_map[idx];
  }
}

template <typename T>
void InvokeMoeAlignBlockSizeGlobalMem(T* topk_ids, int32_t* sorted_token_ids, int32_t* experts_ids,
                                      int32_t* total_tokens_post_pad, const int32_t num_thread,
                                      const int32_t num_experts, const int32_t block_size, const size_t numel,
                                      int32_t* tokens_cnts, int32_t* cumsum, const cudaStream_t& stream) {
  moe_align_block_size_global_mem_kernel<T><<<1, num_thread, 0, stream>>>(topk_ids, sorted_token_ids, experts_ids,
                                                                          total_tokens_post_pad, num_experts,
                                                                          block_size, numel, tokens_cnts, cumsum);
}
#define INVOKE_MOE_ALIGN_BLOCK_SIZE_GLOBAL_MEM(T)                                                        \
  template void InvokeMoeAlignBlockSizeGlobalMem<T>(                                                     \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * experts_ids, int32_t * total_tokens_post_pad,  \
      const int32_t num_thread, const int32_t num_experts, const int32_t block_size, const size_t numel, \
      int32_t* tokens_cnts, int32_t* cumsum, const cudaStream_t& stream)
INVOKE_MOE_ALIGN_BLOCK_SIZE_GLOBAL_MEM(int32_t);
#undef INVOKE_MOE_ALIGN_BLOCK_SIZE_GLOBAL_MEM

template <typename T, typename TOKEN_CNTS_T, bool UseExpertParallel>
void InvokeMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* experts_ids,
                             int32_t* total_tokens_post_pad, const int32_t* expert_map, const int32_t topk,
                             const int32_t num_experts, const int32_t expert_para_size, const int32_t block_size,
                             const size_t numel, const int32_t rank, const cudaStream_t& stream) {
  static int device_count = llm_kernels::utils::getDeviceCount();
  static std::vector<bool> initialized(device_count, false);
  static std::vector<int> num_threads(device_count, 0);
  static std::vector<int> shared_mems(device_count, 0);
  static std::vector<int> num_experts_list(device_count, 0);
  static std::mutex init_mutex;
  if (!initialized[rank] || num_experts_list[rank] != num_experts) {
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!initialized[rank] || num_experts_list[rank] != num_experts) {
      cudaSetDevice(rank);

      int device_max_shared_mem = llm_kernels::utils::getMaxSharedMemoryPerBlockOptin();
      int device_max_threads_per_block = llm_kernels::utils::getMaxThreadPerBlock();

      // 为该rank设置内核属性
      cudaError_t err = cudaFuncSetAttribute(moe_align_block_size_kernel_expert_parallel<T, TOKEN_CNTS_T>,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, device_max_shared_mem);
      if (err != cudaSuccess) {
        printf(
            "CUDA error in InvokeMoeAlignBlockSize: cudaFuncSetAttribute(moe_align_block_size_kernel_expert_parallel) "
            "error in rank %d - %s\n",
            rank, cudaGetErrorString(err));
      }

      err = cudaFuncSetAttribute(moe_align_block_size_kernel<T, TOKEN_CNTS_T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, device_max_shared_mem);
      if (err != cudaSuccess) {
        printf(
            "CUDA error in InvokeMoeAlignBlockSize: cudaFuncSetAttribute(moe_align_block_size_kernel) error in rank %d "
            "- %s\n",
            rank, cudaGetErrorString(err));
      }

      int target_num_thread = (device_max_shared_mem - (num_experts * expert_para_size + 1) * sizeof(int32_t)) /
                                  (num_experts * expert_para_size * sizeof(uint16_t)) -
                              2;
      target_num_thread = std::min(target_num_thread, device_max_threads_per_block);

      num_threads[rank] = target_num_thread;
      shared_mems[rank] = device_max_shared_mem;
      num_experts_list[rank] = num_experts;
      initialized[rank] = true;
    }
  }

  int num_thread = num_threads[rank];
  int shared_mem = shared_mems[rank];
  if constexpr (UseExpertParallel) {
    moe_align_block_size_kernel_expert_parallel<T, TOKEN_CNTS_T>
        <<<1, num_thread, shared_mem, stream>>>(topk_ids, sorted_token_ids, experts_ids, total_tokens_post_pad,
                                                expert_map, topk, num_experts, block_size, numel);
  } else {
    moe_align_block_size_kernel<T, TOKEN_CNTS_T><<<1, num_thread, shared_mem, stream>>>(
        topk_ids, sorted_token_ids, experts_ids, total_tokens_post_pad, num_experts, block_size, numel);
  }
}
#define INVOKE_MOE_ALIGN_BLOCK_SIZE(T, TOKEN_CNTS_T)                                                            \
  template void InvokeMoeAlignBlockSize<T, TOKEN_CNTS_T, true>(                                                 \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * experts_ids, int32_t * total_tokens_post_pad,         \
      const int32_t* expert_map, const int32_t topk, const int32_t expert_para_size, const int32_t num_experts, \
      const int32_t block_size, const size_t numel, const int32_t rank, const cudaStream_t& stream);            \
  template void InvokeMoeAlignBlockSize<T, TOKEN_CNTS_T, false>(                                                \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * experts_ids, int32_t * total_tokens_post_pad,         \
      const int32_t* expert_map, const int32_t topk, const int32_t expert_para_size, const int32_t num_experts, \
      const int32_t block_size, const size_t numel, const int32_t rank, const cudaStream_t& stream);
INVOKE_MOE_ALIGN_BLOCK_SIZE(int32_t, uint16_t);
INVOKE_MOE_ALIGN_BLOCK_SIZE(int32_t, int32_t);
#undef INVOKE_MOE_ALIGN_BLOCK_SIZE

template <typename T>
void InvokeSglMoeAlignBlockSize(T* topk_ids, int32_t* sorted_token_ids, int32_t* expert_ids,
                                int32_t* total_tokens_post_pad, int32_t num_experts, int32_t block_size, size_t numel,
                                int32_t* cumsum, const cudaStream_t& stream) {
  sgl_moe_align_block_size_kernel<T><<<1, 1024, 0, stream>>>(
      topk_ids, sorted_token_ids, expert_ids, total_tokens_post_pad, num_experts, block_size, numel, cumsum);

  const int block_threads = 256;
  const int num_blocks = (numel + block_threads - 1) / block_threads;
  const int max_blocks = 65535;
  const int actual_blocks = std::min(num_blocks, max_blocks);
  sgl_moe_token_sort_kernel<T><<<actual_blocks, block_threads, 0, stream>>>(topk_ids, sorted_token_ids, cumsum, numel);
}
#define INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE(T)                                                             \
  template void InvokeSglMoeAlignBlockSize<T>(                                                         \
      T * topk_ids, int32_t * sorted_token_ids, int32_t * expert_ids, int32_t * total_tokens_post_pad, \
      int32_t num_experts, int32_t block_size, size_t numel, int32_t * cumsum, const cudaStream_t& stream)
INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE(int32_t);
#undef INVOKE_SGL_MOE_ALIGN_BLOCK_SIZE

template <typename T>
void InvokeDynamicPerTokenScaledFP8Quant(const T* input, const float* scale, const void* scale_ub, void* output,
                                         const int hidden_size, const int num_tokens, const cudaStream_t& stream) {
  dim3 const grid(num_tokens);
  dim3 const block(std::min(hidden_size, 1024));
  dynamic_per_token_scaled_fp8_quant_kernel<T><<<grid, block, 0, stream>>>(output, scale, input, scale_ub, hidden_size);
}

template <typename T>
void InvokeDynamicScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                 const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(1024);
  segmented_max_reduction<T><<<grid, block, 0, stream>>>(scale, input, num_elems);
  scaled_fp8_quant_kernel<T><<<grid, block, 0, stream>>>(output, input, scale, num_elems);
}

template <typename T>
void InvokeStaticScaledFP8Quant(const T* input, float* scale, void* output, const int num_tokens, const int num_elems,
                                const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(1024);
  scaled_fp8_quant_kernel<T><<<grid, block, 0, stream>>>(output, input, scale, num_elems);
}

template <typename T, bool UseExpertParallel>
void InvokeMoeSum(void* input,      // [num_tokens, topk, hidden_size]
                  void* output,     // [num_tokens, hidden_size]
                  void* topk_ids,   // [..., topk]
                  int* expert_map,  // [expert_nums]
                  int num_tokens, const int num_experts, int topk, int hidden_size, const cudaStream_t& stream) {
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  switch (topk) {
#define CASE_LAUNCH_MOE_SUM_KERNEL(TOPK)                                                                           \
  case TOPK:                                                                                                       \
    if constexpr (UseExpertParallel) {                                                                             \
      moe_sum_kernel_expert_parallel<T, TOPK><<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input), reinterpret_cast<const int*>(topk_ids), \
          reinterpret_cast<const int*>(expert_map), num_experts, hidden_size);                                     \
    } else {                                                                                                       \
      moe_sum_kernel<T, TOPK><<<grid, block, 0, stream>>>(                                                         \
          reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input), num_experts, hidden_size);              \
    }                                                                                                              \
    break;
    CASE_LAUNCH_MOE_SUM_KERNEL(1);
    CASE_LAUNCH_MOE_SUM_KERNEL(2);
    CASE_LAUNCH_MOE_SUM_KERNEL(3);
    CASE_LAUNCH_MOE_SUM_KERNEL(4);
    CASE_LAUNCH_MOE_SUM_KERNEL(5);
    CASE_LAUNCH_MOE_SUM_KERNEL(6);
    CASE_LAUNCH_MOE_SUM_KERNEL(7);
    CASE_LAUNCH_MOE_SUM_KERNEL(8);
    CASE_LAUNCH_MOE_SUM_KERNEL(9);
    CASE_LAUNCH_MOE_SUM_KERNEL(10);
    CASE_LAUNCH_MOE_SUM_KERNEL(11);
    CASE_LAUNCH_MOE_SUM_KERNEL(12);
    CASE_LAUNCH_MOE_SUM_KERNEL(13);
    CASE_LAUNCH_MOE_SUM_KERNEL(14);
    CASE_LAUNCH_MOE_SUM_KERNEL(15);
    CASE_LAUNCH_MOE_SUM_KERNEL(16);
    CASE_LAUNCH_MOE_SUM_KERNEL(32);
    CASE_LAUNCH_MOE_SUM_KERNEL(64);
#undef CASE_LAUNCH_MOE_SUM_KERNEL
    default:
      dim3 block_size(16, 16);
      dim3 grid_size((num_tokens + block_size.x - 1) / block_size.x, (hidden_size + block_size.y - 1) / block_size.y);

      sum_out_dim1_kernel<T><<<grid_size, block_size>>>(reinterpret_cast<T*>(output), reinterpret_cast<const T*>(input),
                                                        num_tokens, topk, hidden_size);
      break;
  }
}
#define INVOKE_MOE_SUM(T)                                                                                            \
  template void InvokeMoeSum<T, true>(void* input, void* output, void* topk_ids, int* expert_map, int num_tokens,    \
                                      const int num_experts, int topk, int hidden_size, const cudaStream_t& stream); \
  template void InvokeMoeSum<T, false>(void* input, void* output, void* topk_ids, int* expert_map, int num_tokens,   \
                                       const int num_experts, int topk, int hidden_size, const cudaStream_t& stream)
INVOKE_MOE_SUM(float);
INVOKE_MOE_SUM(half);
INVOKE_MOE_SUM(__nv_bfloat16);
#undef INVOKE_MOE_SUM

template <typename T, bool UseExpertParallel>
void SiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                size_t elements_num, size_t inter_size, const cudaStream_t& stream) {
  size_t num_tokens = elements_num / 2 / inter_size;
  // dim3 grid(std::min(elements_num / 2 / inter_size, static_cast<size_t>(65535)));
  dim3 grid(num_tokens);
  dim3 block(std::min(inter_size, static_cast<size_t>(1024)));
  if constexpr (UseExpertParallel) {
    silu_and_mul_kernel_expert_parallel<T>
        <<<grid, block, 0, stream>>>(input, output, topk_ids, expert_map, num_experts, num_tokens, inter_size);
  } else {
    silu_and_mul_kernel<T><<<grid, block, 0, stream>>>(input, output, num_tokens, inter_size);
  }
}

#define SILU_AND_MUL(T)                                                                                     \
  template void SiluAndMul<T, true>(const T* input, T* output, const int* topk_ids, const int* expert_map,  \
                                    int num_experts, size_t elements_num, size_t inter_size,                \
                                    const cudaStream_t& stream);                                            \
  template void SiluAndMul<T, false>(const T* input, T* output, const int* topk_ids, const int* expert_map, \
                                     int num_experts, size_t elements_num, size_t inter_size,               \
                                     const cudaStream_t& stream)
SILU_AND_MUL(float);
SILU_AND_MUL(half);
SILU_AND_MUL(__nv_bfloat16);
#undef SILU_AND_MUL

__device__ __forceinline__ float silu(const float& val) { return val / (1.0f + __expf(-val)); }

template <typename T>
void FlashinferSiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                          size_t elements_num, size_t inter_size, const cudaStream_t& stream) {
  size_t num_tokens = elements_num / 2 / inter_size;
  dim3 grid(num_tokens);
  uint32_t vec_size = 16 / sizeof(T);
  dim3 block(std::min(inter_size / vec_size, static_cast<size_t>(1024)));
  flashinfer::activation::act_and_mul_kernel<T, silu><<<grid, block, 0, stream>>>(output, input, inter_size);
}

#define FLASHINFER_SILU_AND_MUL(T)                                                                             \
  template void FlashinferSiluAndMul<T>(const T* input, T* output, const int* topk_ids, const int* expert_map, \
                                        int num_experts, size_t elements_num, size_t inter_size,               \
                                        const cudaStream_t& stream)
FLASHINFER_SILU_AND_MUL(float);
FLASHINFER_SILU_AND_MUL(half);
FLASHINFER_SILU_AND_MUL(__nv_bfloat16);
#undef FLASHINFER_SILU_AND_MUL

template <typename T, bool UseExpertParallel>
void InvokeSiluAndMul(const T* input, T* output, const int* topk_ids, const int* expert_map, int num_experts,
                      size_t elements_num, size_t inter_size, const cudaStream_t& stream) {
  if constexpr (UseExpertParallel) {
    SiluAndMul<T, true>(input, output, topk_ids, expert_map, num_experts, elements_num, inter_size, stream);
  } else {
#if defined(ENABLE_FLASHINFER)
    FlashinferSiluAndMul<T>(input, output, topk_ids, expert_map, num_experts, elements_num, inter_size, stream);
#else
    SiluAndMul<T, false>(input, output, topk_ids, expert_map, num_experts, elements_num, inter_size, stream);
#endif
  }
}

#define INVOKE_SILU_AND_MUL(T)                                                                                    \
  template void InvokeSiluAndMul<T, true>(const T* input, T* output, const int* topk_ids, const int* expert_map,  \
                                          int num_experts, size_t elements_num, size_t inter_size,                \
                                          const cudaStream_t& stream);                                            \
  template void InvokeSiluAndMul<T, false>(const T* input, T* output, const int* topk_ids, const int* expert_map, \
                                           int num_experts, size_t elements_num, size_t inter_size,               \
                                           const cudaStream_t& stream)
INVOKE_SILU_AND_MUL(float);
INVOKE_SILU_AND_MUL(half);
INVOKE_SILU_AND_MUL(__nv_bfloat16);
#undef INVOKE_SILU_AND_MUL

template <typename T>
void InvokeWeightDequant(const uint8_t* x, const float* s, T* output, int M, int N, int block_size,
                         const cudaStream_t& stream) {
  dim3 block(256);
  dim3 grid((M * N + 256 - 1) / 256);
  weight_dequant_kernel<T><<<grid, block, 0, stream>>>(x, s, output, M, N, block_size);
}

#define INVOKE_WEIGHT_DEQUANT(T)                                                                                  \
  template void InvokeWeightDequant<T>(const uint8_t* x, const float* s, T* output, int M, int N, int block_size, \
                                       const cudaStream_t& stream)
INVOKE_WEIGHT_DEQUANT(half);
INVOKE_WEIGHT_DEQUANT(float);
INVOKE_WEIGHT_DEQUANT(__nv_bfloat16);
#undef INVOKE_WEIGHT_DEQUANT

void InvokeMapExpertIds(const int* expert_map, int* expert_ids, int total_num) {
  const int BLOCK_SIZE = 256;
  int num_blocks = (total_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
  map_expert_ids_kernel<<<num_blocks, BLOCK_SIZE>>>(expert_map, expert_ids, total_num);
}

__global__ void fill_buffer_kernel(int* buffer, const int* fill_info, int fill_info_length) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int32_t i = 0; i < fill_info_length; i += 3) {
    int32_t start_offset = fill_info[i];
    int32_t length = fill_info[i + 1];
    if (idx >= start_offset && idx < start_offset + length) {
      buffer[idx] = fill_info[i + 2];
      break;
    }
  }
}

void InvokeFillIntToBuffer(int* output, void* fill_info, int* fill_info_on_host, int fill_info_length,
                           const cudaStream_t& stream) {
  cudaMemcpyAsync(fill_info, reinterpret_cast<void*>(fill_info_on_host), fill_info_length * sizeof(int),
                  cudaMemcpyHostToDevice, stream);
  // [start_offset0, length0, fill_value0, ... , start_offsetN, lengthN, fill_valueN]
  const int total_elems = fill_info_on_host[fill_info_length - 2] + fill_info_on_host[fill_info_length - 3];

  const int BLOCK_SIZE = 256;
  int num_blocks = ceil(static_cast<float>(total_elems) / BLOCK_SIZE);
  fill_buffer_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(output, static_cast<int*>(fill_info), fill_info_length);
}

}  // namespace nvidia
}  // namespace llm_kernels
