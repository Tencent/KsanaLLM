/*
 * Adapted from
 * https://github.com/microsoft/tokenweave/blob/main/csrc/tokenweave_fused_kernels.cu
 *
 * MIT License
 *
 * Copyright (c) 2025 Microsoft
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "csrc/kernels/nvidia/others/vllm/main/tokenweave/tokenweave_fused_kernels.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>

#include "csrc/kernels/nvidia/others/vllm/main/tokenweave/tokenweave_multimem_utils.cuh"
#include "csrc/kernels/nvidia/others/vllm/main/tokenweave/type_convert.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace vllm {
/*
 * ********************************************************* *
 * FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
 * Function specialization in the case of BF16/FP16 tensors. *
 * ********************************************************* *
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists> fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ mcptr,         // [token_num, hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,      // [token_num, hidden_size]
    const scalar_t *__restrict__ weight,  // [hidden_size]
    uint32_t **signal_pads, size_t rank, size_t world_size, const float epsilon, const int num_tokens,
    const int hidden_size) {
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  auto *__restrict__ weight_v = reinterpret_cast<const vec_t *>(weight);
  const int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

#pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++) {
    const int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens) {
      continue;
    }
    float variance[1] = {0.0f};
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    __shared__ float s_variance;
    const int offset = token_id * vec_hidden_size;
    const int offset_scalar = token_id * hidden_size;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx) {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      temp += residual_o[idx];
      variance[0] += temp.sum_squares();  // FP32 accumulation
      residual_o[idx] = temp;
    }

    blockReduceSum<float, 1>(variance);
    if (threadIdx.x == 0) {
      s_variance = rsqrtf(variance[0] / hidden_size + epsilon);
    }
    __syncthreads();

    // Second pass: normalize and apply weight
    for (int idx = tid; idx < vec_hidden_size; idx += bdimx) {
      vec_t shared_weight = weight_v[idx];
      vec_t temp = residual_o[idx];
      temp *= s_variance;
      temp *= shared_weight;
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/*
 * ********************************************************* *
 * FUSED RS + RESIDUAL ADD + RMS NORM + AG CTA-BASED KERNEL  *
 * GENERIC NOT SUPPORTED                                     *
 * ********************************************************* *
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists> fused_rs_ln_ag_cta_kernel(
    scalar_t *__restrict__ mcptr,         // [token_num, hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,      // [token_num, hidden_size]
    const scalar_t *__restrict__ weight,  // [hidden_size]
    uint32_t **signal_pads, size_t rank, size_t world_size, const float epsilon, const int num_tokens,
    const int hidden_size) {
  /* Not supported */
  KLLM_KERNEL_THROW("TokenWeave currently only supports bf16/fp16 with width 8.");
}

/*
 * ********************************************************* *
 * FUSED RS + RESIDUAL ADD + AG CTA-BASED KERNEL             *
 * Function specialization in the case of BF16/FP16 tensors. *
 * ********************************************************* *
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists> fused_rs_ag_cta_kernel(
    scalar_t *__restrict__ mcptr,     // [token_num, hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,  // [token_num, hidden_size]
    uint32_t **signal_pads, size_t rank, size_t world_size, const int num_tokens, const int hidden_size) {
  // Check vectorization assumptions
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  using vec_t = _f16Vec<scalar_t, width>;

  // Type-punned vector pointers
  auto *__restrict__ residual_v = reinterpret_cast<vec_t *>(residual);
  const int tokens_per_iter = (num_tokens + gridDim.x - 1) / gridDim.x;

  sync_remote_blocks<MemOpSem::Relaxed>(signal_pads, rank, world_size);
  __syncthreads();

#pragma unroll
  for (int iter = 0; iter < tokens_per_iter; iter++) {
    const int token_id = blockIdx.x + iter * gridDim.x;
    if (token_id >= num_tokens) {
      continue;
    }
    const int tid = threadIdx.x;
    const int bdimx = blockDim.x;

    const int offset = token_id * vec_hidden_size;
    const int offset_scalar = token_id * hidden_size;
    auto residual_o = residual_v + offset;

    for (int idx = tid; idx < vec_hidden_size; idx += bdimx) {
      auto mtemp = multimem_ld_reduce_add<16>(mcptr + offset_scalar + idx * width);
      vec_t temp = *(reinterpret_cast<vec_t *>(&mtemp));
      temp += residual_o[idx];
      multimem_st<16>(mcptr + offset_scalar + idx * width, *(reinterpret_cast<Vec<16> *>(&temp)));
    }
  }
  __syncthreads();
  sync_remote_blocks<MemOpSem::AcqRel>(signal_pads, rank, world_size);
}

/*
 * ********************************************************* *
 * FUSED RS + RESIDUAL ADD + AG CTA-BASED KERNEL             *
 * GENERIC NOT SUPPORTED                                     *
 * ********************************************************* *
 */
template <typename scalar_t, int width>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists> fused_rs_ag_cta_kernel(
    scalar_t *__restrict__ mcptr,     // [token_num, hidden_size] multimem_ptr
    scalar_t *__restrict__ residual,  // [token_num, hidden_size]
    uint32_t **signal_pads, size_t rank, size_t world_size, const int num_tokens, const int hidden_size) {
  /* Not supported */
  KLLM_KERNEL_THROW("TokenWeave currently only supports bf16/fp16 with width 8.");
}

}  // namespace vllm

namespace llm_kernels {
namespace nvidia {

// Multimem based AllReduce implementations require very few SMs
constexpr int kMaxCtas = 16;
/*
 * ******************************************************************* *
 * Fused ReduceScatter plus Fused(Residual, RMSNorm) plus AllGather    *
 * ******************************************************************* *
 */
template <typename T>
void FusedRsLmAgCta(int64_t mcptr,       // [token_num, hidden_size] multimem_ptr
                    void *residual,      // [token_num, hidden_size]
                    const void *weight,  // [hidden_size]
                    void *signal_pads,   // [token_num, hidden_size]
                    int rank, int world_size, float epsilon, int num_tokens, int hidden_size, cudaStream_t stream) {
  dim3 grid(std::min(kMaxCtas, num_tokens));  // full coverage
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops.
   */
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32));  // match kernel assumptions
  /* If the tensor types are FP16/BF16, try to use the optimized kernel
     with packed + vectorized ops.
     Max optimization is achieved with a width-8 vector of FP16/BF16s
     since we can load at most 128 bits at once in a global memory op.
     However, this requires each tensor's data to be aligned to 16
     bytes.
   */
  KLLM_KERNEL_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(residual) % 16 == 0 &&
                                  reinterpret_cast<uintptr_t>(weight) % 16 == 0 && hidden_size % 8 == 0,
                              "Residual, and weight tensors must be 16-byte aligned and hidden_size must be "
                              "divisible by 8 for optimized kernel.");
  constexpr size_t kWidth = 8;
  using C10Type = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, c10::BFloat16, c10::Half>;
  vllm::fused_rs_ln_ag_cta_kernel<C10Type, kWidth><<<grid, block, 0, stream>>>(
      reinterpret_cast<C10Type *>(mcptr), reinterpret_cast<C10Type *>(residual),
      reinterpret_cast<const C10Type *>(weight), reinterpret_cast<uint32_t **>(signal_pads), rank, world_size, epsilon,
      num_tokens, hidden_size);
}

/*
 * ******************************************************************* *
 * Fused ReduceScatter plus Fused(Residual) plus AllGather             *
 * ******************************************************************* *
 */
template <typename T>
void FusedRsAgCta(int64_t mcptr,      // [token_num, hidden_size] multimem_ptr
                  void *residual,     // [token_num, hidden_size]
                  void *signal_pads,  // [token_num, hidden_size]
                  int rank, int world_size, int num_tokens, int hidden_size, cudaStream_t stream) {
  dim3 grid(std::min(kMaxCtas, num_tokens));  // full coverage
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops.
   */
  dim3 block(std::min(1024, (hidden_size / 8 + 31) / 32 * 32));  // match kernel assumptions
  /* If the tensor types are FP16/BF16, try to use the optimized kernel
     with packed + vectorized ops.
     Max optimization is achieved with a width-8 vector of FP16/BF16s
     since we can load at most 128 bits at once in a global memory op.
     However, this requires each tensor's data to be aligned to 16
     bytes.
   */
  KLLM_KERNEL_CHECK_WITH_INFO(reinterpret_cast<uintptr_t>(residual) % 16 == 0 && hidden_size % 8 == 0,
                              "Residual tensors must be 16-byte aligned and hidden_size must be "
                              "divisible by 8 for optimized kernel.");
  constexpr size_t kWidth = 8;
  using C10Type = std::conditional_t<std::is_same_v<T, __nv_bfloat16>, c10::BFloat16, c10::Half>;
  vllm::fused_rs_ag_cta_kernel<C10Type, kWidth><<<grid, block, 0, stream>>>(
      reinterpret_cast<C10Type *>(mcptr), reinterpret_cast<C10Type *>(residual),
      reinterpret_cast<uint32_t **>(signal_pads), rank, world_size, num_tokens, hidden_size);
}

#define INSTANTIATE_FUSED_CTA(T)                                                                                   \
  template void FusedRsLmAgCta<T>(int64_t, void *, const void *, void *, int, int, float, int, int, cudaStream_t); \
  template void FusedRsAgCta<T>(int64_t, void *, void *, int, int, int, int, cudaStream_t);
INSTANTIATE_FUSED_CTA(float);
INSTANTIATE_FUSED_CTA(half);
INSTANTIATE_FUSED_CTA(__nv_bfloat16);
#undef INSTANTIATE_FUSED_CTA

}  // namespace nvidia
}  // namespace llm_kernels
