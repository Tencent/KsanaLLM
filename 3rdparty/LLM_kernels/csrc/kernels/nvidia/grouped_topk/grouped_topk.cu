/* Copyright 2025 Tencent Inc.  All rights reserved.
   Copyright 2025 Sglang Team
 * Adapted from: https://gist.github.com/whitelok/441146ab3d022bcfa74d07a3a58f2012

==============================================================================*/

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#  include <cuda_bf16.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cub/cub.cuh>
#include <torch/torch.h>

#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"

namespace llm_kernels {
namespace nvidia {

static constexpr int32_t kMaxCandidateExpertNum = 256;
// Adapted from
// [Sglang Project]
// https://github.com/sgl-project/sglang/blob/9858113c336f4565a0a35f9a990cdada0de1988f/sgl-kernel/csrc/moe/moe_fused_gate.cu#L40
static constexpr int WARP_SIZE = 32;
static constexpr int WARPS_PER_CTA = 6;
static constexpr int MAX_VPT = 32;
template <typename T, int N>
using AlignedArray = cutlass::AlignedArray<T, N>;

// Create an alias for Array using AlignedArray
template <typename T, int N>
using Array = AlignedArray<T, N>;
// QQ: NOTE expression must have a constant value, this has to be > params.VPT
template <typename T>
using AccessType = AlignedArray<T, MAX_VPT>;

template <int VPT_, int NUM_EXPERTS_, int THREADS_PER_ROW_, int ROWS_PER_WARP_, int ROWS_PER_CTA_, int WARPS_PER_CTA_>
struct KernelParams {
  static constexpr int VPT = VPT_;
  static constexpr int NUM_EXPERTS = NUM_EXPERTS_;
  static constexpr int THREADS_PER_ROW = THREADS_PER_ROW_;
  static constexpr int ROWS_PER_WARP = ROWS_PER_WARP_;
  static constexpr int ROWS_PER_CTA = ROWS_PER_CTA_;
  static constexpr int WARPS_PER_CTA = WARPS_PER_CTA_;
};

template <int MAX_K>
__device__ void GetTopKIndices(const float* data, int* indices, int n, int k, int index_offset) {
  // Local arrays used as heap storage
  float heap_values[MAX_K];
  int heap_indices[MAX_K];

  // Initialize the heap
  for (int i = 0; i < k; i++) {
    heap_values[i] = data[i];
    heap_indices[i] = i + index_offset;
  }

  // Build min heap
  for (int i = k / 2 - 1; i >= 0; i--) {
    int parent = i;
    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      // Swap elements
      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  // Process remaining elements
  for (int i = k; i < n; i++) {
    if (data[i] > heap_values[0]) {
      heap_values[0] = data[i];
      heap_indices[0] = i + index_offset;

      // Sift down
      int parent = 0;
      while (true) {
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;
        int smallest = parent;

        if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
          smallest = leftChild;
        }
        if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
          smallest = rightChild;
        }

        if (smallest == parent) break;

        float temp_val = heap_values[parent];
        heap_values[parent] = heap_values[smallest];
        heap_values[smallest] = temp_val;

        int temp_idx = heap_indices[parent];
        heap_indices[parent] = heap_indices[smallest];
        heap_indices[smallest] = temp_idx;

        parent = smallest;
      }
    }
  }

  // Extract results
  for (int i = k - 1; i > 0; i--) {
    indices[k - 1 - i] = heap_indices[0];

    heap_values[0] = heap_values[i];
    heap_indices[0] = heap_indices[i];

    int parent = 0;
    int heapSize = i;

    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < heapSize && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < heapSize && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  indices[k - 1] = heap_indices[0];
}

// Select the top k from data of size n specified by data_indices.
template <int MAX_K>
__device__ void GetTopKIndices(const float* data, int* indices, int n, int k, int32_t* data_indices) {
  // Local arrays used as heap storage
  float heap_values[MAX_K];
  int heap_indices[MAX_K];

  // Initialize the heap
  for (int i = 0; i < k; i++) {
    heap_values[i] = data[data_indices[i]];
    heap_indices[i] = data_indices[i];
  }

  // Build min heap
  for (int i = k / 2 - 1; i >= 0; i--) {
    int parent = i;
    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      // Swap elements
      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  // Process remaining elements
  for (int i = k; i < n; i++) {
    if (data[data_indices[i]] > heap_values[0]) {
      heap_values[0] = data[data_indices[i]];
      heap_indices[0] = data_indices[i];

      // Sift down
      int parent = 0;
      while (true) {
        int leftChild = 2 * parent + 1;
        int rightChild = 2 * parent + 2;
        int smallest = parent;

        if (leftChild < k && heap_values[leftChild] < heap_values[smallest]) {
          smallest = leftChild;
        }
        if (rightChild < k && heap_values[rightChild] < heap_values[smallest]) {
          smallest = rightChild;
        }

        if (smallest == parent) break;

        float temp_val = heap_values[parent];
        heap_values[parent] = heap_values[smallest];
        heap_values[smallest] = temp_val;

        int temp_idx = heap_indices[parent];
        heap_indices[parent] = heap_indices[smallest];
        heap_indices[smallest] = temp_idx;

        parent = smallest;
      }
    }
  }

  // Extract results
  for (int i = k - 1; i > 0; i--) {
    indices[k - 1 - i] = heap_indices[0];

    heap_values[0] = heap_values[i];
    heap_indices[0] = heap_indices[i];

    int parent = 0;
    int heapSize = i;

    while (true) {
      int leftChild = 2 * parent + 1;
      int rightChild = 2 * parent + 2;
      int smallest = parent;

      if (leftChild < heapSize && heap_values[leftChild] < heap_values[smallest]) {
        smallest = leftChild;
      }
      if (rightChild < heapSize && heap_values[rightChild] < heap_values[smallest]) {
        smallest = rightChild;
      }

      if (smallest == parent) break;

      float temp_val = heap_values[parent];
      heap_values[parent] = heap_values[smallest];
      heap_values[smallest] = temp_val;

      int temp_idx = heap_indices[parent];
      heap_indices[parent] = heap_indices[smallest];
      heap_indices[smallest] = temp_idx;

      parent = smallest;
    }
  }

  indices[k - 1] = heap_indices[0];
}

// Adapted from
// [Sglang Project]
// https://github.com/sgl-project/sglang/blob/9858113c336f4565a0a35f9a990cdada0de1988f/sgl-kernel/csrc/moe/moe_fused_gate.cu#L51
// typename T: the type of the compute type in kernel
// typename Input_T: the type of the input data
// typename BT: the type of the bias data
template <typename T, typename Input_T, typename BT, typename Params>
__device__ void moe_fused_gate_impl(void* input, void* bias, float* output_ptr, int32_t* indices_ptr, int64_t num_rows,
                                    int64_t topk_group, int64_t topk, int64_t n_share_experts_fusion,
                                    float routed_scaling_factor, Params params) {
  int tidx = threadIdx.x;
  int64_t thread_row =
      blockIdx.x * params.ROWS_PER_CTA + threadIdx.y * params.ROWS_PER_WARP + tidx / params.THREADS_PER_ROW;
  if (thread_row >= num_rows) {
    return;
  }

  // Calculate topk_excluding_share_expert_fusion from topk
  int64_t topk_excluding_share_expert_fusion = topk - (n_share_experts_fusion > 0 ? 1 : 0);

  // Cast pointers to type T:
  Input_T* input_ptr = reinterpret_cast<Input_T*>(input);
  BT* bias_ptr = reinterpret_cast<BT*>(bias);
  Input_T* thread_row_ptr = input_ptr + thread_row * params.NUM_EXPERTS;

  int thread_group_idx = tidx % params.THREADS_PER_ROW;
  int first_elt_read_by_thread = thread_group_idx * params.VPT;

  // Create local arrays for the row chunk and bias chunk and then reinterpret the address of row_chunk as a pointer to
  // AccessType.
  Input_T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> row_chunk;
  AccessType<Input_T> const* vec_thread_read_ptr = reinterpret_cast<AccessType<Input_T> const*>(thread_read_ptr);

  BT* bias_thread_read_ptr = bias_ptr + first_elt_read_by_thread;
  Array<T, MAX_VPT> bias_chunk;
  AccessType<BT> const* vec_bias_thread_read_ptr = reinterpret_cast<AccessType<BT> const*>(bias_thread_read_ptr);

// QQ NOTE: doing the follow will be slower than loop assign and more importantly
// have misaligned address issue when params.VPT < 8 and mismatch with MAX_VPT
// AccessType<T>* row_chunk_vec_ptr = reinterpret_cast<AccessType<T>*>(&row_chunk);
// row_chunk_vec_ptr[0] = vec_thread_read_ptr[0];
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(vec_thread_read_ptr[0][ii]);
    bias_chunk[ii] = static_cast<T>(vec_bias_thread_read_ptr[0][ii]);
  }

  __syncthreads();

////////////////////// Sigmoid //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    row_chunk[ii] = static_cast<T>(1.0f / (1.0f + expf(-float(row_chunk[ii]))));
  }
  __syncthreads();

////////////////////// Add Bias //////////////////////
#pragma unroll
  for (int ii = 0; ii < params.VPT; ++ii) {
    bias_chunk[ii] = row_chunk[ii] + bias_chunk[ii];
  }

////////////////////// Exclude Groups //////////////////////
#pragma unroll
  for (int k_idx = 0; k_idx < params.THREADS_PER_ROW - topk_group;
       ++k_idx) {  // QQ NOTE Here params.THREADS_PER_ROW = num_expert_group
    int expert = first_elt_read_by_thread;
    // local argmax
    T max_val = static_cast<T>(-FLT_MAX);
    T max_val_second = static_cast<T>(-FLT_MAX);
#pragma unroll
    for (int ii = 0; ii < params.VPT; ++ii) {
      T val = bias_chunk[ii];

      if (val > max_val) {
        max_val_second = max_val;
        max_val = val;
      } else if (val > max_val_second) {
        max_val_second = val;
      }
    }

    // QQ NOTE: currently fixed to pick top2 sigmoid weight value in each expert group and sum them as the group weight
    // to select expert groups
    T max_sum = max_val + max_val_second;

// argmin reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max_sum =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_sum), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // higher indices win
      if (max_sum > other_max_sum || (other_max_sum == max_sum && other_expert > expert)) {
        max_sum = other_max_sum;
        expert = other_expert;
      }
    }

    // clear the max value in the thread
    if (k_idx < params.THREADS_PER_ROW - topk_group) {
      int const thread_to_clear_in_group = expert / params.VPT;

      if (thread_group_idx == thread_to_clear_in_group) {
#pragma unroll
        for (int ii = 0; ii < params.VPT; ++ii) {
          bias_chunk[ii] = static_cast<T>(FLT_MAX);
        }
      }
    }
  }

  __syncthreads();

  ////////////////////// Topk //////////////////////
  float output_sum = 0.0f;
  for (int k_idx = 0; k_idx < topk_excluding_share_expert_fusion; ++k_idx) {
    // local argmax
    T max_val = bias_chunk[0];
    int expert = first_elt_read_by_thread;

    if (!(max_val == static_cast<T>(FLT_MAX))) {
#pragma unroll
      for (int ii = 1; ii < params.VPT; ++ii) {
        T val = bias_chunk[ii];
        if (val > max_val) {
          max_val = val;
          expert = first_elt_read_by_thread + ii;
        }
      }
    } else {
      max_val = static_cast<T>(-FLT_MAX);
    }

    // argmax reduce
#pragma unroll
    for (int mask = params.THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      T other_max =
          static_cast<T>(__shfl_xor_sync(0xFFFFFFFF, static_cast<float>(max_val), mask, params.THREADS_PER_ROW));
      int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, params.THREADS_PER_ROW);

      // lower indices to win
      if (other_max > max_val || (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    int thread_to_clear_in_group = expert / params.VPT;
    int64_t idx = topk * thread_row + k_idx;

    if (thread_group_idx == thread_to_clear_in_group) {
      int expert_to_clear_in_thread = expert % params.VPT;

      // clear the max value in the thread
      bias_chunk[expert_to_clear_in_thread] = static_cast<T>(-FLT_MAX);

      // store output
      output_ptr[idx] = static_cast<float>(row_chunk[expert_to_clear_in_thread]);
      indices_ptr[idx] = static_cast<int32_t>(expert);
    }

    // accumulate sum for all elements
    if (thread_group_idx == 0) {
      output_sum += output_ptr[idx];
    }

    __syncthreads();
  }

  ////////////////////// take care shared experts //////////////////////
  // Note(rockcao): the share_experts_fusion is not supported in this version,
  // the following code is reserved for future use.
  // if (thread_group_idx == 0 && n_share_experts_fusion > 0) {
  //   int64_t last_idx = topk * thread_row + topk_excluding_share_expert_fusion;

  //   // Use round-robin to select expert
  //   int64_t expert_offset = thread_row % n_share_experts_fusion;
  //   indices_ptr[last_idx] = static_cast<int32_t>(params.NUM_EXPERTS + expert_offset);

  //   // Set the weight to the sum of all weights divided by routed_scaling_factor
  //   output_ptr[last_idx] = output_sum / routed_scaling_factor;
  // }
  // __syncthreads();

  ////////////////////// Rescale Output //////////////////////
  if (thread_group_idx == 0) {
#pragma unroll
    for (int ii = 0; ii < topk; ++ii) {
      int64_t const idx = topk * thread_row + ii;
      output_ptr[idx] = output_ptr[idx] / output_sum * routed_scaling_factor;
    }
  }
}

// Macro to compute compile-time constants and launch the kernel.
#define LAUNCH_DSV3_FUSED_GROUPED_TOPK(T, EXPERTS, EXPERT_GROUP)                                              \
  do {                                                                                                        \
    constexpr int VPT = (EXPERTS) / (EXPERT_GROUP);                                                           \
    /* If EXPERT_GROUP > WARP_SIZE, fall back to 1 row per warp */                                            \
    constexpr int ROWS_PER_WARP = ((EXPERT_GROUP) <= WARP_SIZE) ? (WARP_SIZE / (EXPERT_GROUP)) : 1;           \
    constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;                                               \
    moe_fused_gate_kernel<T, VPT, (EXPERTS), (EXPERT_GROUP), ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA>      \
        <<<num_blocks, block_dim, 0, stream>>>(gating_output, e_bias, static_cast<float*>(topk_weights),      \
                                               static_cast<int32_t*>(topk_ids), tokens_num, topk_group, topk, \
                                               n_share_experts_fusion, routed_scaling_factor);                \
  } while (0)

template <typename T, int VPT, int NUM_EXPERTS, int THREADS_PER_ROW, int ROWS_PER_WARP, int ROWS_PER_CTA,
          int WARPS_PER_CTA>
__global__ void moe_fused_gate_kernel(void* input, void* bias, float* output_ptr, int32_t* indices_ptr,
                                      int64_t num_rows, int64_t topk_group, int64_t topk,
                                      int64_t n_share_experts_fusion, float routed_scaling_factor) {
  KernelParams<VPT, NUM_EXPERTS, THREADS_PER_ROW, ROWS_PER_WARP, ROWS_PER_CTA, WARPS_PER_CTA> params;
  moe_fused_gate_impl<float, T, float>(input, bias, output_ptr, indices_ptr, num_rows, topk_group, topk,
                                       n_share_experts_fusion, routed_scaling_factor, params);
}

template <typename T>
__global__ void FusedDSV3GroupedTopkKernel(const T* gating_output, const float* e_bias, float routed_scaling_factor,
                                           float* topk_weights, int32_t* topk_ids, int num_expert_group,
                                           int experts_per_group, int topk_group, int topk) {
  // NOTE(karlluo): refer
  // https://github.com/Chen-XiaoBing/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py#L923 memory refer
  // range:
  //     num_experts * sizeof(float) for original_scores
  //     num_experts * sizeof(float) for scores
  //     num_expert_group * sizeof(float) for group_scores
  //     topk_group * sizeof(int32_t) for group_idx
  //     topk_group * topk * sizeof(int32_t) for topk_id_in_group
  extern __shared__ float buf[];

  uint32_t num_experts = blockDim.x;
  uint32_t group_idx = threadIdx.x / experts_per_group;

  // for sigmoid
  float val = static_cast<float>(gating_output[blockIdx.x * num_experts + threadIdx.x]);
  // for add bias
  buf[threadIdx.x] = 1.0f / (1.0f + expf(-val));  // for original_scores
  __syncthreads();

  buf[threadIdx.x + num_experts] = buf[threadIdx.x] + e_bias[threadIdx.x];  // for scores
  // load from global memory need sync
  __syncthreads();

  // NOTE(karlluo): for original py code: group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
  // NOTE(rockcao): But here refer to
  // https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/modeling_deepseek.py#L441, we using scores.view(num_token,
  // num_expert_group, -1).topk(2, -1).sum(-1) [num_token, num_expert_group] get each expert group's max val
  // TODO(karlluo): optimized with shuffle/ballot instruction
  if (threadIdx.x % experts_per_group == 0) {
    float group_max_val_1 = -INFINITY;
    float group_max_val_2 = -INFINITY;
    for (uint32_t expert_idx_in_group = 0; expert_idx_in_group < experts_per_group; ++expert_idx_in_group) {
      if (buf[threadIdx.x + expert_idx_in_group + num_experts] > group_max_val_1) {
        group_max_val_2 = group_max_val_1;
        group_max_val_1 = buf[threadIdx.x + expert_idx_in_group + num_experts];
      } else if (buf[threadIdx.x + expert_idx_in_group + num_experts] > group_max_val_2) {
        group_max_val_2 = buf[threadIdx.x + expert_idx_in_group + num_experts];
      }
    }
    buf[2 * num_experts + group_idx] = group_max_val_1 + group_max_val_2;
  }
  __syncthreads();

  // NOTE(karlluo): for original py code: group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]  #
  // [num_token, top_k_group]

  int32_t* group_indices = reinterpret_cast<int32_t*>(buf + 2 * num_experts + num_expert_group);
  if (threadIdx.x == 0) {
    GetTopKIndices<kMaxCandidateExpertNum>(static_cast<float*>(buf + 2 * num_experts), group_indices, num_expert_group,
                                           topk_group, 0);
  }
  __syncthreads();

  // NOTE(rockcao): the expert groups which are not selected exit directly
  bool is_selected_group = false;
  for (int topk_g_idx = 0; topk_g_idx < topk_group; ++topk_g_idx) {
    if (group_idx == group_indices[topk_g_idx]) {
      is_selected_group = true;
      break;
    }
  }
  if (!is_selected_group) {
    return;
  }
  __syncthreads();

  // NOTE(rockcao): select topk expert from its group (i.e. local topk in expert_group)
  int32_t* group_topk_index = reinterpret_cast<int32_t*>(buf + 2 * num_experts + num_expert_group + topk_group);
  if (threadIdx.x == group_idx * experts_per_group) {
    for (int topk_g_idx = 0; topk_g_idx < topk_group; ++topk_g_idx) {
      if (group_idx == group_indices[topk_g_idx]) {
        GetTopKIndices<kMaxCandidateExpertNum>(static_cast<float*>(buf + num_experts + group_idx * experts_per_group),
                                               group_topk_index + topk_g_idx * topk, experts_per_group, topk,
                                               threadIdx.x);
      }
    }
  }
  __syncthreads();

  // NOTE(rockcao): select topk expert from local topk experts
  bool is_working_thread = (threadIdx.x == group_indices[0] * experts_per_group);
  if (is_working_thread) {
    GetTopKIndices<kMaxCandidateExpertNum>(static_cast<float*>(buf + num_experts), topk_ids + blockIdx.x * topk,
                                           topk_group * topk, topk, group_topk_index);
  }
  __syncthreads();

  for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
    if (threadIdx.x == topk_ids[blockIdx.x * topk + topk_idx]) {
      topk_weights[blockIdx.x * topk + topk_idx] = buf[threadIdx.x];
    }
  }
  __syncthreads();

  if (is_working_thread) {
    float expert_sum_val = 0.0f;
    for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
      expert_sum_val += topk_weights[blockIdx.x * topk + topk_idx];
    }
    for (int topk_idx = 0; topk_idx < topk; ++topk_idx) {
      topk_weights[blockIdx.x * topk + topk_idx] =
          topk_weights[blockIdx.x * topk + topk_idx] / expert_sum_val * routed_scaling_factor;
    }
  }
}

// Main function: Implementation of grouped_topk
template <typename T>
void DeepSeekV3GroupedTopkCudaKernel(const T* gating_output, const float* e_bias, float routed_scaling_factor,
                                     float* topk_weights, int32_t* topk_ids, int tokens_num, int num_experts, int topk,
                                     int num_expert_group, int topk_group, cudaStream_t stream) {
  // NOTE(karlluo): Calculate the number of experts per group
  int experts_per_group = (num_experts + num_expert_group - 1) / num_expert_group;

  dim3 grid_size(tokens_num);
  dim3 block_size(num_experts);

  // NOTE(karlluo): shared memory struct
  //     num_experts * sizeof(float) for original_scores
  //     num_experts * sizeof(float) for scores
  //     num_expert_group * sizeof(float) for group_scores
  //     topk_group * sizeof(int32_t) for group_idx
  //     topk_group * topk * sizeof(int32_t) for topk_id_in_group

  size_t each_token_shm_buf_size = (2 * num_experts + num_expert_group) * sizeof(float) + topk_group * sizeof(int32_t) +
                                   topk_group * topk * sizeof(int32_t);

  FusedDSV3GroupedTopkKernel<T><<<grid_size, block_size, each_token_shm_buf_size, stream>>>(
      gating_output, e_bias, routed_scaling_factor, topk_weights, topk_ids, num_expert_group, experts_per_group,
      topk_group, topk);
}

// Wrapper function for external calls
template <typename T>
void InvokeDeepSeekV3GroupedTopk(void* gating_output, void* e_bias, float routed_scaling_factor, void* topk_weights,
                                 void* topk_ids, int tokens_num, int num_experts, int topk, int num_expert_group,
                                 int topk_group, cudaStream_t stream) {
  // NOTE(rockcao): for DeepSeekV3 with ep disabled, num_experts = 256, num_expert_group = 8
  // TODO(rockcao): support more config
  if (num_experts == 256 && num_expert_group == 8) {
    // NOTE(rockcao): experts_fusion is not supported in this version
    const int64_t n_share_experts_fusion = 0;
    int num_rows = tokens_num;
    int64_t rows_per_warp = std::max<int64_t>(1, WARP_SIZE / num_expert_group);
    int64_t num_warps = (num_rows + rows_per_warp - 1) / rows_per_warp;
    int64_t num_blocks = (num_warps + WARPS_PER_CTA - 1) / WARPS_PER_CTA;
    dim3 block_dim(WARP_SIZE, WARPS_PER_CTA);
    LAUNCH_DSV3_FUSED_GROUPED_TOPK(T, 256, 8);
  } else {
    DeepSeekV3GroupedTopkCudaKernel<T>(static_cast<const T*>(gating_output), static_cast<const float*>(e_bias),
                                       routed_scaling_factor, static_cast<float*>(topk_weights),
                                       static_cast<int32_t*>(topk_ids), tokens_num, num_experts, topk, num_expert_group,
                                       topk_group, stream);
  }
}

#define INVOKE_DEEPSEEK_V3_GROUPED_TOPK(T)                                                                        \
  template void InvokeDeepSeekV3GroupedTopk<T>(                                                                   \
      void* gating_output, void* e_bias, float routed_scaling_factor, void* topk_weights_ptr, void* topk_ids_ptr, \
      int tokens_num, int num_experts, int topk, int num_expert_group, int topk_group, cudaStream_t stream)

INVOKE_DEEPSEEK_V3_GROUPED_TOPK(float);
INVOKE_DEEPSEEK_V3_GROUPED_TOPK(half);
INVOKE_DEEPSEEK_V3_GROUPED_TOPK(__nv_bfloat16);
#undef INVOKE_DEEPSEEK_V3_GROUPED_TOPK

// Basic softmax + topk kernel for simple cases without grouping
template <typename T>
__global__ void BasicSoftmaxTopkKernel(T* gating_output, float* topk_weights, int32_t* topk_ids, int num_rows,
                                       int num_experts, int topk, float routed_scaling_factor) {
  int row_idx = blockIdx.x;
  if (row_idx >= num_rows) return;

  extern __shared__ float shared_mem[];
  float* scores = shared_mem;  // 每个block处理一行，shared memory存储该行的softmax结果

  T* output_row = gating_output + row_idx * num_experts;

  // Step 1: 在专家维度上计算softmax
  // 每个线程负责一个专家，并行加载数据到shared memory
  if (threadIdx.x < num_experts) {
    scores[threadIdx.x] = static_cast<float>(output_row[threadIdx.x]);
  }
  __syncthreads();

  // 找到最大值以保证数值稳定性 - 让thread 0来做串行计算确保正确性
  if (threadIdx.x == 0) {
    float max_val = scores[0];
    for (int i = 1; i < num_experts; i++) {
      max_val = fmaxf(max_val, scores[i]);
    }

    // 计算 exp(x - max) 并求和
    float sum = 0.0f;
    for (int i = 0; i < num_experts; i++) {
      scores[i] = expf(scores[i] - max_val);
      sum += scores[i];
    }

    // 归一化得到softmax概率
    for (int i = 0; i < num_experts; i++) {
      scores[i] = scores[i] / sum;
    }
  }
  __syncthreads();

  // 将softmax结果写回到gating_output，保留注释方便debug
  // if (threadIdx.x < num_experts) {
  //   output_row[threadIdx.x] = static_cast<T>(scores[threadIdx.x]);
  // }

  // Step 2: 使用现有的GetTopKIndices函数进行top-k选择
  if (threadIdx.x == 0) {
    int* temp_indices = reinterpret_cast<int*>(scores + num_experts);  // 复用shared memory
    GetTopKIndices<32>(scores, temp_indices, num_experts, topk, 0);

    // 将结果写入输出数组，并应用scaling factor
    for (int i = 0; i < topk; i++) {
      topk_ids[row_idx * topk + i] = temp_indices[i];
      topk_weights[row_idx * topk + i] = scores[temp_indices[i]] * routed_scaling_factor;
    }
  }
}

// Wrapper function for basic softmax + topk
template <typename T>
void InvokeBasicSoftmaxTopk(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr, int num_rows,
                            int num_experts, int topk, float routed_scaling_factor, cudaStream_t stream) {
  // 验证参数
  if (num_experts >= 1024 || topk > num_experts || topk > 32) {
    // 参数不合法，直接返回
    return;
  }

  dim3 grid_size(num_rows);
  // 由于 num_experts < 1024，可以直接使用 num_experts 作为 block_size
  // 但为了更好的性能，使用 warp 对齐的大小
  int block_size = ((num_experts + 31) / 32) * 32;  // 向上对齐到 32 的倍数（warp size）

  // Shared memory: num_experts * sizeof(float) for scores + 32 * sizeof(int) for temp indices
  size_t shared_mem_size = num_experts * sizeof(float) + 32 * sizeof(int);

  BasicSoftmaxTopkKernel<T><<<grid_size, block_size, shared_mem_size, stream>>>(
      static_cast<T*>(gating_output), static_cast<float*>(topk_weights_ptr), static_cast<int32_t*>(topk_ids_ptr),
      num_rows, num_experts, topk, routed_scaling_factor);
}

#define INVOKE_BASIC_SOFTMAX_TOPK(T)                                                                            \
  template void InvokeBasicSoftmaxTopk<T>(void* gating_output, void* topk_weights_ptr, void* topk_ids_ptr,      \
                                          int num_rows, int num_experts, int topk, float routed_scaling_factor, \
                                          cudaStream_t stream)

INVOKE_BASIC_SOFTMAX_TOPK(float);
INVOKE_BASIC_SOFTMAX_TOPK(half);
INVOKE_BASIC_SOFTMAX_TOPK(__nv_bfloat16);
#undef INVOKE_BASIC_SOFTMAX_TOPK

// fill GPU memory with random integers using PyTorch
void FillRandomInts(int* data_ptr, int size, int start_int, int end_int, int rank, cudaStream_t stream) {
  // Set manual seed for reproducibility
  // NOTE(rockcao): Using torch to facilitate the same random number generation method across different frameworks
  torch::manual_seed(42);

  auto options = torch::TensorOptions()
                     .dtype(torch::kInt32)
                     .device(torch::kCUDA, rank)
                     .requires_grad(false);

  auto tensor = torch::from_blob(data_ptr, {size}, options);
  tensor.random_(start_int, end_int);
}

}  // namespace nvidia
}  // namespace llm_kernels