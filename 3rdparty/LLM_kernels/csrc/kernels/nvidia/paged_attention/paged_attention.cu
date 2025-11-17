/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * and
 * https://github.com/vllm-project/vllm/blob/v0.2.3/csrc/attention/attention_kernels.cu
 * Copyright (c) 2023, Tencent Inc.
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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
 */

#include "paged_attention.h"
#include "paged_attention_dtypes.h"
#include "paged_attention_utils.cuh"
#include "quant_utils.cuh"

#include "csrc/utils/nvidia/cuda_utils.h"

#include <algorithm>

namespace llm_kernels {
namespace nvidia {

#define _PARTITION_SIZE 512
#define DIVIDE_ROUND_UP(a, b) (((a) + (b)-1) / (b))
#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// TODO(woosuk): Merge the last two dimensions of the grid.
// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, llm_kernels::utils::KVCacheType KV_DTYPE, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,      // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,  // [num_seqs, num_heads, head_size]
    cache_t** __restrict__ k_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size/x, block_size, x]
    cache_t** __restrict__ v_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size, block_size]
    const int num_head_repeats,      // num_heads / num_kv_heads
    const float scale,
    const int* __restrict__ cache_offsets,   // [num_seqs]
    const int* __restrict__ context_lens,    // [num_seqs]
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_head_stride, const float k_scale, const float v_scale) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / num_head_repeats;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  cache_t** k_ptrs = k_cache + static_cast<int64_t>(cache_offsets[seq_idx]);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const cache_t* k_block_ptr = k_ptrs[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];
#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr = k_block_ptr + kv_head_idx * kv_head_stride + physical_block_offset * x;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        } else {
          // Vector conversion from Quant_vec to K_vec.
          Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
          k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(k_vec_quant, k_scale);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr =
        exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using V_quant_vec = typename Vec<cache_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  cache_t** v_ptrs = v_cache + static_cast<int64_t>(cache_offsets[seq_idx]);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    const cache_t* v_block_ptr = v_ptrs[block_idx];
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const cache_t* v_ptr = v_block_ptr + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        } else {
          V_quant_vec v_quant_vec = *reinterpret_cast<const V_quant_vec*>(v_ptr + offset);
          // Vector conversion from V_quant_vec to V_vec.
          v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec, v_scale);
        }
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
          }
        }
        accs[i] += dot(logits_vec, v_vec);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, 1).
template <typename scalar_t, typename cache_t, llm_kernels::utils::KVCacheType KV_DTYPE, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,      // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,  // [num_seqs, num_heads, head_size]
    cache_t** __restrict__ k_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size/x, block_size, x]
    cache_t** __restrict__ v_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size, block_size]
    const int num_head_repeats,      // num_heads / num_kv_heads
    const float scale,
    const int* __restrict__ cache_offsets,   // [num_seqs]
    const int* __restrict__ context_lens,    // [num_seqs]
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_head_stride, const float k_scale, const float v_scale) {
  paged_attention_kernel<scalar_t, cache_t, KV_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
      /* exp_sums */ nullptr, /* max_logits */ nullptr, out, q, k_cache, v_cache, num_head_repeats, scale,
      cache_offsets, context_lens, alibi_slopes, q_stride, kv_head_stride, k_scale, v_scale);
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, typename cache_t, llm_kernels::utils::KVCacheType KV_DTYPE, int HEAD_SIZE, int BLOCK_SIZE,
          int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,  // [num_seqs, num_heads, head_size]
    cache_t** __restrict__ k_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size/x, block_size, x]
    cache_t** __restrict__ v_cache,  // num_seqs x [seq_blocks, num_kv_heads, head_size, block_size]
    const int num_head_repeats,      // num_heads / num_kv_heads
    const float scale,
    const int* __restrict__ cache_offsets,   // [num_seqs]
    const int* __restrict__ context_lens,    // [num_seqs]
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_head_stride, const float k_scale, const float v_scale) {
  paged_attention_kernel<scalar_t, cache_t, KV_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_head_repeats, scale, cache_offsets, context_lens,
      alibi_slopes, q_stride, kv_head_stride, k_scale, v_scale);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel(
    scalar_t* __restrict__ out,            // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor_sync(uint32_t(-1), max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, __shfl_xor_sync(uint32_t(-1), max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = __shfl_sync(uint32_t(-1), max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                                                                       \
  cudaFuncSetAttribute(paged_attention_v1_kernel<SCALAR_T, CACHE_T, KV_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>, \
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);                              \
  paged_attention_v1_kernel<SCALAR_T, CACHE_T, KV_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>                       \
      <<<grid, block, shared_mem_size, params.stream_>>>(                                                          \
          params.out_, params.query_, params.key_caches_, params.value_caches_, params.num_head_repeats_,          \
          params.scale_, params.cache_offsets_, params.context_lens_, params.alibi_slopes_, params.q_stride_,      \
          params.kv_head_stride_, params.k_scale_, params.v_scale_);

// TODO(woosuk): Tune NUM_THREADS.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE, int BLOCK_SIZE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(const PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>& params) {
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);

  if (params.head_size_ % thread_group_size != 0) {
    throw std::runtime_error("head_size % thread_group_size != 0");
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = DIVIDE_ROUND_UP(params.max_context_len_, BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * params.head_size_ * sizeof(float);
  // Python-side check in vllm.worker.worker._check_if_can_support_max_seq_len
  // Keep that in sync with the logic here!
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(params.num_heads_, params.num_seqs_, 1);
  dim3 block(NUM_THREADS);
  switch (params.head_size_) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      throw std::runtime_error("Unsupported head size: ");
      break;
  }
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                                                          \
  paged_attention_v2_kernel<SCALAR_T, CACHE_T, KV_DTYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>          \
      <<<grid, block, shared_mem_size, params.stream_>>>(                                                             \
          params.exp_sums_, params.max_logits_, params.tmp_out_, params.query_, params.key_caches_,                   \
          params.value_caches_, params.num_head_repeats_, params.scale_, params.cache_offsets_, params.context_lens_, \
          params.alibi_slopes_, params.q_stride_, params.kv_head_stride_, params.k_scale_, params.v_scale_);          \
  paged_attention_v2_reduce_kernel<SCALAR_T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>                                  \
      <<<reduce_grid, block, reduce_shared_mem_size, params.stream_>>>(params.out_, params.exp_sums_,                 \
                                                                       params.max_logits_, params.tmp_out_,           \
                                                                       params.context_lens_, max_num_partitions);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE, int BLOCK_SIZE,
          int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(const PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>& params) {
  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);

  if (params.head_size_ % thread_group_size != 0) {
    throw std::runtime_error("head_size % thread_group_size != 0");
  }
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int max_num_partitions = DIVIDE_ROUND_UP(params.max_context_len_, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * params.head_size_ * sizeof(float);

  // For paged attention v2 kernel.
  dim3 grid(params.num_heads_, params.num_seqs_, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  // For paged attention v2 reduce kernel.
  dim3 reduce_grid(params.num_heads_, params.num_seqs_);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 block(NUM_THREADS);
  switch (params.head_size_) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      throw std::runtime_error("Unsupported head size");
      break;
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void paged_attention_v1(const PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>& params) {
  switch (params.block_size_) {
    case 8:
      paged_attention_v1_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 8>(params);
      break;
    case 16:
      paged_attention_v1_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 16>(params);
      break;
    case 32:
      paged_attention_v1_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 32>(params);
      break;
    case 64:
      paged_attention_v1_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 64>(params);
      break;
    default:
      throw std::runtime_error("Unsupported block size:" + std::to_string(params.block_size_));
      break;
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void paged_attention_v2(const PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>& params) {
  // NOTE(woosuk): To reduce the compilation time, we omitted block sizes
  // 1, 2, 4, 64, 128, 256.
  switch (params.block_size_) {
    case 8:
      paged_attention_v2_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 8>(params);
      break;
    case 16:
      paged_attention_v2_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 16>(params);
      break;
    case 32:
      paged_attention_v2_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 32>(params);
      break;
    case 64:
      paged_attention_v2_launcher<SCALAR_T, CACHE_T, KV_DTYPE, 64>(params);
      break;
    default:
      throw std::runtime_error("Unsupported block size:" + std::to_string(params.block_size_));
      break;
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void paged_attention_impl(const PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>& params) {
  if (params.use_v1_) {
    // v1
    paged_attention_v1<SCALAR_T, CACHE_T, KV_DTYPE>(params);
  } else {
    // v2
    paged_attention_v2<SCALAR_T, CACHE_T, KV_DTYPE>(params);
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
int PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::GetTmpOutNumel() const {
  return num_seqs_ * num_heads_ * max_num_partitions_ * head_size_;
}
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
int PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::GetExpSumsNumel() const {
  return num_seqs_ * num_heads_ * max_num_partitions_;
}
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
int PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::GetMaxLogitsNumel() const {
  return num_seqs_ * num_heads_ * max_num_partitions_;
}
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
int PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::GetMaxNumPartitions() const {
  return DIVIDE_ROUND_UP(max_context_len_, _PARTITION_SIZE);
}
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
bool PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::IsUseV1() const {
  int max_num_partitions = max_num_partitions_;
  // NOTE(woosuk): We use a simple heuristic to decide whether to use
  // PagedAttention V1 or V2. If the number of partitions is 1, we use
  // V1 to avoid the overhead of reduction. Also, if the number of
  // sequences or heads is large, we use V1 since there is enough work
  // to parallelize.
  // TODO(woosuk): Tune this heuristic.
  // For context len > 8192, use V2 kernel to avoid shared memory shortage.
  bool use_v1 = max_context_len_ <= 8192 and (max_num_partitions == 1 or num_seqs_ * num_heads_ > _PARTITION_SIZE);
  return use_v1;
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
size_t PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::GetWorkSize() const {
  size_t work_size = 0;
  if (!use_v1_) {
    size_t exp_sums_size = GetExpSumsNumel() * sizeof(float);
    work_size += exp_sums_size;
    size_t max_logits_size = GetMaxLogitsNumel() * sizeof(float);
    work_size += max_logits_size;
    size_t tmp_out_size = GetTmpOutNumel() * sizeof(SCALAR_T);
    work_size += tmp_out_size;
  }
  return work_size;
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE>::SetWorkSpace(void* workspace, size_t work_size) {
  char* workspace_ptr = static_cast<char*>(workspace);
  if (GetWorkSize() > work_size) {
    throw std::runtime_error("workspace less than needed");
  }
  if (!use_v1_) {
    exp_sums_ = reinterpret_cast<float*>(workspace_ptr);
    workspace_ptr += GetExpSumsNumel() * sizeof(float);
    max_logits_ = reinterpret_cast<float*>(workspace_ptr);
    workspace_ptr += GetMaxLogitsNumel() * sizeof(float);
    tmp_out_ = reinterpret_cast<SCALAR_T*>(workspace_ptr);
    workspace_ptr += GetTmpOutNumel() * sizeof(SCALAR_T);
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttentionCuda<SCALAR_T, CACHE_T, KV_DTYPE>::SetConfig(const int num_kv_heads, int num_heads, int head_size,
                                                                int block_size, int stride_size, float k_scale,
                                                                float v_scale) {
  params_.num_head_repeats_ = num_heads / num_kv_heads;
  params_.num_heads_ = num_heads;
  params_.head_size_ = head_size;
  params_.q_stride_ = stride_size;
  params_.kv_head_stride_ = head_size * block_size;
  params_.k_scale_ = k_scale;
  params_.v_scale_ = v_scale;
  params_.scale_ = rsqrt(static_cast<float>(head_size));
  params_.block_size_ = block_size;
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
size_t PagedAttentionCuda<SCALAR_T, CACHE_T, KV_DTYPE>::GetWorkspaceSize(int num_seqs, int max_context_len) {
  params_.num_seqs_ = num_seqs;
  params_.max_context_len_ = max_context_len;
  params_.max_num_partitions_ = params_.GetMaxNumPartitions();
  params_.use_v1_ = params_.IsUseV1();
  return params_.GetWorkSize();
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttentionCuda<SCALAR_T, CACHE_T, KV_DTYPE>::SetInput(SCALAR_T* out, const SCALAR_T* query,
                                                               CACHE_T** key_caches, CACHE_T** value_caches,
                                                               const int* cache_offsets, const int* context_lens,
                                                               int max_context_len, int num_seqs, cudaStream_t stream,
                                                               void* workspace, size_t work_size,
                                                               const float* alibi_slopes) {
  params_.out_ = out;
  params_.query_ = query;
  params_.key_caches_ = key_caches;
  params_.value_caches_ = value_caches;
  params_.cache_offsets_ = cache_offsets;
  params_.context_lens_ = context_lens;
  params_.max_context_len_ = max_context_len;
  params_.num_seqs_ = num_seqs;

  params_.stream_ = stream;
  params_.alibi_slopes_ = alibi_slopes;

  params_.max_num_partitions_ = params_.GetMaxNumPartitions();
  params_.use_v1_ = params_.IsUseV1();
  params_.SetWorkSpace(workspace, work_size);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void PagedAttentionCuda<SCALAR_T, CACHE_T, KV_DTYPE>::Forward() {
  paged_attention_impl(params_);
}

}  // namespace nvidia
}  // namespace llm_kernels
