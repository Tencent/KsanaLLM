/*
 * Copyright 2025 vLLM Team
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
 * [vLLM Project] https://github.com/vllm-project/vllm/blob/v0.8.4/csrc/moe/moe_wna16.cu
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "moe_wna16_utils.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "csrc/utils/nvidia/string_utils.h"

namespace llm_kernels {
namespace nvidia {
namespace moe_wna16 {

#define DIVIDE(x, size) (((x) + (size)-1) / (size))

template <typename scalar_t, int bit, int GROUPS>
__global__ void moe_wna16_gemm_kernel(
    const scalar_t* __restrict__ input, scalar_t* __restrict__ output,

    const uint32_t* __restrict__ qweight, const scalar_t* __restrict__ scales, const uint32_t* __restrict__ qzeros,

    const float* __restrict__ topk_weights, const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids, const int32_t* __restrict__ num_tokens_post_pad,

    uint16_t num_experts, uint16_t group_size, uint16_t top_k, uint32_t size_m, uint32_t size_n, uint32_t size_k,
    uint16_t BLOCK_SIZE_M, uint16_t BLOCK_SIZE_N, uint16_t BLOCK_SIZE_K, bool has_zp, bool mul_topk_weight) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
  if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    return;
  } else {
#endif

    using Dtype = ScalarType<scalar_t>;
    using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;

    if (blockIdx.x * BLOCK_SIZE_M >= num_tokens_post_pad[0]) return;

    const int32_t offset_n = blockIdx.y * BLOCK_SIZE_N + threadIdx.x;
    const int32_t offset_k = blockIdx.z * BLOCK_SIZE_K;

    const int32_t expert_id = expert_ids[blockIdx.x];

    int32_t num_valid_tokens = 0;
    extern __shared__ uint16_t block_input_tmp[];
    scalar_t* block_input = reinterpret_cast<scalar_t*>(block_input_tmp);
    scalar_t2* block_input_half2 = reinterpret_cast<scalar_t2*>(block_input);

    // load BLOCK_SIZE_M * BLOCK_SIZE_K into shared memory
    for (int m = 0; m < BLOCK_SIZE_M; m++) {
      const int32_t offset_m = blockIdx.x * BLOCK_SIZE_M + m;
      const int32_t token_index = sorted_token_ids[offset_m];
      if (token_index / top_k >= size_m) break;

      num_valid_tokens = m + 1;
      if (blockIdx.z == 0 && offset_n < size_n) output[token_index * size_n + offset_n] = Dtype::int2num(0);

      if (expert_id != -1) {
        int k_per_thread = DIVIDE(BLOCK_SIZE_K, BLOCK_SIZE_N);
        for (int i = 0; i < k_per_thread; i++) {
          int k = BLOCK_SIZE_N * i + threadIdx.x;
          if (k >= BLOCK_SIZE_K) break;
          if (offset_k + k >= size_k) break;

          // load input to shared memory
          // use a special layout to fit the layout of dequanted-weight
          int origin_k;
          if constexpr (bit == 4) {
            // [0, 4, 1, 5, 2, 6, 3, 7]
            int8_t order = (threadIdx.x % 2) * 4 + ((threadIdx.x % 8) / 2);
            origin_k = BLOCK_SIZE_N * i + threadIdx.x / 8 * 8 + order;
          } else {
            // [0, 2, 1, 3]
            int8_t order = (threadIdx.x % 2) * 2 + ((threadIdx.x % 4) / 2);
            origin_k = BLOCK_SIZE_N * i + threadIdx.x / 4 * 4 + order;
          }

          origin_k += token_index / top_k * size_k + blockIdx.z * BLOCK_SIZE_K;
          block_input[m * BLOCK_SIZE_K + k] = input[origin_k];
        }
      }
    }

    if (expert_id == -1) return;
    __syncthreads();
    if (threadIdx.x >= BLOCK_SIZE_N || offset_n >= size_n) return;

    float res[64];  // assume BLOCK_SIZE_M <= 64
    scalar_t2 res2;
    scalar_t2 scale_f2;
    scalar_t2 qzero_f2;

    // note that (size_n * size_k * expert_id) may greater than 2 ** 31
    constexpr int8_t pack_factor = 32 / bit;
    const uint64_t expert_offset = ((uint64_t)size_n) * size_k * expert_id;
    const uint32_t* expert_qweight = qweight + expert_offset / pack_factor;
    const scalar_t* expert_scales = scales + expert_offset / group_size;
    const uint32_t* expert_qzeros = qzeros + expert_offset / group_size / pack_factor;

    // load 4*int32 one time: 4 int32 = 128 bit = 1 float4
    // weight would be loaded in loop
    uint32_t expert_qweight_tmp[4];
    float4* expert_qweight_tmp_float4 = reinterpret_cast<float4*>(expert_qweight_tmp);

    // load all required scales one time
    scalar_t expert_scales_groups[GROUPS];
    int scales_offset_tmp = (offset_n * size_k + offset_k) / group_size / GROUPS;
    if constexpr (GROUPS == 1) {
      *expert_scales_groups = expert_scales[scales_offset_tmp];
    } else if constexpr (GROUPS == 2) {
      float* expert_scales_groups_tmp = reinterpret_cast<float*>(expert_scales_groups);
      *expert_scales_groups_tmp = reinterpret_cast<const float*>(expert_scales)[scales_offset_tmp];
    } else if constexpr (GROUPS == 4) {
      float2* expert_scales_groups_tmp = reinterpret_cast<float2*>(expert_scales_groups);
      *expert_scales_groups_tmp = reinterpret_cast<const float2*>(expert_scales)[scales_offset_tmp];
    } else if constexpr (GROUPS == 8) {
      float4* expert_scales_groups_tmp = reinterpret_cast<float4*>(expert_scales_groups);
      *expert_scales_groups_tmp = reinterpret_cast<const float4*>(expert_scales)[scales_offset_tmp];
    }

    // load all required qzeros one time
    uint8_t expert_qzeros_groups[GROUPS];
    if (!has_zp) {
      if constexpr (bit == 4) {
        qzero_f2 = Dtype::num2num2(Dtype::int2num(8));
      } else {
        qzero_f2 = Dtype::num2num2(Dtype::int2num(128));
      }
    } else {
      int qzeros_offset_tmp = (offset_n / (8 / bit)) * (size_k / group_size / GROUPS) + offset_k / group_size / GROUPS;
      if constexpr (GROUPS == 1) {
        uint8_t* expert_qzeros_groups_tmp = reinterpret_cast<uint8_t*>(expert_qzeros_groups);
        *expert_qzeros_groups_tmp = reinterpret_cast<const uint8_t*>(expert_qzeros)[qzeros_offset_tmp];
      } else if constexpr (GROUPS == 2) {
        uint16_t* expert_qzeros_groups_tmp = reinterpret_cast<uint16_t*>(expert_qzeros_groups);
        *expert_qzeros_groups_tmp = reinterpret_cast<const uint16_t*>(expert_qzeros)[qzeros_offset_tmp];
      } else if constexpr (GROUPS == 4) {
        uint32_t* expert_qzeros_groups_tmp = reinterpret_cast<uint32_t*>(expert_qzeros_groups);
        *expert_qzeros_groups_tmp = reinterpret_cast<const uint32_t*>(expert_qzeros)[qzeros_offset_tmp];
      } else if constexpr (GROUPS == 8) {
        uint64_t* expert_qzeros_groups_tmp = reinterpret_cast<uint64_t*>(expert_qzeros_groups);
        *expert_qzeros_groups_tmp = reinterpret_cast<const uint64_t*>(expert_qzeros)[qzeros_offset_tmp];
      }
    }

    for (int tmp_k = 0; tmp_k < BLOCK_SIZE_K / pack_factor; tmp_k++) {
      int k = offset_k + tmp_k * pack_factor;
      if (k >= size_k) break;
      const int32_t weight_offset = offset_n * size_k + k;

      if (tmp_k % 4 == 0) {
        *expert_qweight_tmp_float4 = reinterpret_cast<const float4*>(expert_qweight)[weight_offset / pack_factor / 4];
      }

      if (tmp_k % (group_size / pack_factor) == 0) {
        scalar_t scale_f = expert_scales_groups[tmp_k / (group_size / pack_factor)];
        scale_f2 = Dtype::num2num2(scale_f);

        if (has_zp) {
          uint8_t qzero = expert_qzeros_groups[tmp_k / (group_size / pack_factor)];
          if constexpr (bit == 4) {
            qzero = (qzero >> ((threadIdx.x % 2) * 4)) & 0xF;
          }
          qzero_f2 = Dtype::num2num2(Dtype::int2num(qzero));
        }
      }

      scalar_t2 weight_half2[16 / bit];
      dequant<scalar_t2, bit>(expert_qweight_tmp[tmp_k % 4], weight_half2);

      for (int m = 0; m < num_valid_tokens; m++) {
        res2 = {};

#pragma unroll
        for (int i = 0; i < 16 / bit; i++) {
          int32_t offset_input = m * BLOCK_SIZE_K / 2 + tmp_k * (16 / bit) + i;
          res2 = __hfma2(__hmul2(__hsub2(weight_half2[i], qzero_f2), scale_f2), block_input_half2[offset_input], res2);
        }

        if (tmp_k == 0) {
          res[m] = Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
        } else {
          res[m] += Dtype::num2float(res2.x) + Dtype::num2float(res2.y);
        }
      }
    }

    for (int m = 0; m < num_valid_tokens; ++m) {
      const int32_t token_index = sorted_token_ids[blockIdx.x * BLOCK_SIZE_M + m];
      if (mul_topk_weight) {
        res[m] *= topk_weights[token_index];
      }
      atomicAdd(&output[token_index * size_n + offset_n], Dtype::float2num(res[m]));
    }

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
  }
#endif
}

template <typename scalar_t>
void run_moe_wna16_gemm(cudaStream_t stream, const scalar_t* input, scalar_t* output, const uint32_t* b_qweight,
                        const scalar_t* b_scales, const uint32_t* b_qzeros, const float* topk_weights,
                        const int32_t* sorted_token_ids, const int32_t* expert_ids, const int32_t* num_tokens_post_pad,
                        int num_experts, int group_size, int num_token_blocks, int top_k, int size_m, int size_n,
                        int size_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, bool has_zp,
                        bool mul_topk_weight) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_SIZE_N;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = num_token_blocks;
  gridDim.y = DIVIDE(size_n, BLOCK_SIZE_N);
  gridDim.z = DIVIDE(size_k, BLOCK_SIZE_K);

  auto kernel = moe_wna16_gemm_kernel<scalar_t, 4, 1>;
  if (bit == 4) {
    if (BLOCK_SIZE_K / group_size == 2) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 4, 2>;
    } else if (BLOCK_SIZE_K / group_size == 4) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 4, 4>;
    } else if (BLOCK_SIZE_K / group_size == 8) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 4, 8>;
    }
  } else {
    if (BLOCK_SIZE_K / group_size == 1) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 8, 1>;
    } else if (BLOCK_SIZE_K / group_size == 2) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 8, 2>;
    } else if (BLOCK_SIZE_K / group_size == 4) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 8, 4>;
    } else if (BLOCK_SIZE_K / group_size == 8) {
      kernel = moe_wna16_gemm_kernel<scalar_t, 8, 8>;
    }
  }

  const int shared_mem_size = BLOCK_SIZE_M * BLOCK_SIZE_K * 2;
  kernel<<<gridDim, blockDim, shared_mem_size, stream>>>(input, output, b_qweight, b_scales, b_qzeros, topk_weights,
                                                         sorted_token_ids, expert_ids, num_tokens_post_pad, num_experts,
                                                         group_size, top_k, size_m, size_n, size_k, BLOCK_SIZE_M,
                                                         BLOCK_SIZE_N, BLOCK_SIZE_K, has_zp, mul_topk_weight);
}

template <typename T>
void moe_wna16_gemm(cudaStream_t stream, void* output, const void* input, const void* b_qweight, const void* b_scales,
                    const void* b_qzeros, const void* topk_weights, const void* sorted_token_ids,
                    const void* expert_ids, const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M,
                    int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts, int size_m, int size_n, int size_k,
                    int group_size, int num_token_blocks) {
  if constexpr (std::is_same_v<T, float>) {
    KLLM_KERNEL_THROW("moe_wna16_gemm not support float type.");
  } else {
    int groups_per_block_row = BLOCK_SIZE_K / group_size;
    KLLM_KERNEL_CHECK_WITH_INFO(bit == 4 || bit == 8, "bit must be 4 or 8");
    KLLM_KERNEL_CHECK_WITH_INFO(size_k % BLOCK_SIZE_K == 0, "size_k must divisible by BLOCK_SIZE_K");
    KLLM_KERNEL_CHECK_WITH_INFO(BLOCK_SIZE_K % group_size == 0, "BLOCK_SIZE_K must divisible by group_size");
    KLLM_KERNEL_CHECK_WITH_INFO(BLOCK_SIZE_M <= 64, "BLOCK_SIZE_M must less or equal to 64");
    KLLM_KERNEL_CHECK_WITH_INFO(groups_per_block_row == 1 || groups_per_block_row == 2 || groups_per_block_row == 4 ||
                                    groups_per_block_row == 8,
                                "BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]");

    run_moe_wna16_gemm<T>(
        stream, reinterpret_cast<const T*>(input), reinterpret_cast<T*>(output),
        reinterpret_cast<const uint32_t*>(b_qweight), reinterpret_cast<const T*>(b_scales),
        reinterpret_cast<const uint32_t*>(b_qzeros), reinterpret_cast<const float*>(topk_weights),
        reinterpret_cast<const int32_t*>(sorted_token_ids), reinterpret_cast<const int32_t*>(expert_ids),
        reinterpret_cast<const int32_t*>(num_tokens_post_pad), num_experts, group_size, num_token_blocks, top_k, size_m,
        size_n, size_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, bit, b_qzeros != nullptr, topk_weights != nullptr);
  }
}

template void moe_wna16_gemm<float>(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                                    const void* b_scales, const void* b_qzeros, const void* topk_weights,
                                    const void* sorted_token_ids, const void* expert_ids,
                                    const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
                                    int BLOCK_SIZE_K, int bit, int num_experts, int size_m, int size_n, int size_k,
                                    int group_size, int num_token_blocks);
template void moe_wna16_gemm<half>(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                                   const void* b_scales, const void* b_qzeros, const void* topk_weights,
                                   const void* sorted_token_ids, const void* expert_ids,
                                   const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N,
                                   int BLOCK_SIZE_K, int bit, int num_experts, int size_m, int size_n, int size_k,
                                   int group_size, int num_token_blocks);
template void moe_wna16_gemm<nv_bfloat16>(cudaStream_t stream, void* output, const void* input, const void* b_qweight,
                                          const void* b_scales, const void* b_qzeros, const void* topk_weights,
                                          const void* sorted_token_ids, const void* expert_ids,
                                          const void* num_tokens_post_pad, int top_k, int BLOCK_SIZE_M,
                                          int BLOCK_SIZE_N, int BLOCK_SIZE_K, int bit, int num_experts, int size_m,
                                          int size_n, int size_k, int group_size, int num_token_blocks);

}  // namespace moe_wna16
}  // namespace nvidia
}  // namespace llm_kernels