/*
 * Adapted from
 * https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/pybind.cpp
 *
 * Copyright (c) 2025 DeepSeek
 *
 * MIT License
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

#include "csrc/kernels/nvidia/flash_mla/flash_sparse_mla.h"

#include "csrc/utils/nvidia/cuda_utils.h"

#ifdef ENABLE_FLASH_MLA
#  include <cutlass/fast_math.h>
#  include <vector>

#  include "params.h"
#  include "sm90/decode/dense/splitkv_mla.h"
#  include "sm90/decode/sparse_fp8/splitkv_mla.h"
#  include "sm90/prefill/sparse/fwd.h"
#  include "smxx/get_mla_metadata.h"
#  include "smxx/mla_combine.h"

namespace llm_kernels {
namespace nvidia {

using namespace llm_kernels::utils;

inline std::vector<int64_t> GetStride(const std::vector<int>& shape) {
  std::vector<int64_t> stride(shape.size(), 1);
  if (shape.size() > 1) {
    for (int i = shape.size() - 2; i >= 0; i--) {
      stride[i] = stride[i + 1] * shape[i + 1];
    }
  }
  return stride;
}

DecodingAttnImplMeta GetAttnImplMeta(int num_q_tokens_per_head_k, int num_heads_k, int num_heads_q, bool is_fp8_kvcache,
                                     bool is_sparse_attn) {
  const static int sm_count = GetSMCount();
  if (is_sparse_attn) {
    if (is_fp8_kvcache) {
      KLLM_KERNEL_CHECK(num_heads_q % num_heads_k == 0);
      const int seqlen_q = num_q_tokens_per_head_k * num_heads_k / num_heads_q;
      // FP8 + Sparse MLA
      return {
          std::max((sm_count / 2) / num_heads_k / (cutlass::ceil_div(num_heads_q / num_heads_k, 2 * 64) * seqlen_q), 1),
          5, 64};
    } else {
      // Sparse BF16 MLA
      KLLM_KERNEL_THROW("Sparse BF16 MLA is not supported on SM90");
    }
  } else {
    if (is_fp8_kvcache) {
      // Dense FP8 MLA
      KLLM_KERNEL_THROW("Dense FP8 MLA is not supported on SM90");
    } else {
      // Dense BF16 MLA
      return {std::max(sm_count / num_heads_k / cutlass::ceil_div(num_q_tokens_per_head_k, 64), 1), 5, 64};
    }
  }
}

void InvokeGetSparseMlaMetadata(int* seqlens_k_ptr, int batch_size, int num_q_tokens_per_head_k, int num_heads_k,
                                int num_heads_q, bool is_fp8_kvcache, int topk, cudaStream_t stream,
                                int* tile_scheduler_metadata_ptr, int* num_splits_ptr) {
  bool is_sparse_attn = (topk != -1);
  DecodingAttnImplMeta attn_impl_meta =
      GetAttnImplMeta(num_q_tokens_per_head_k, num_heads_k, num_heads_q, is_fp8_kvcache, is_sparse_attn);

  GetDecodingMetadataParams params = {};
  params.seqlens_k_ptr = seqlens_k_ptr;
  params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
  params.num_splits_ptr = num_splits_ptr;
  params.batch_size = batch_size;
  params.block_size_n = attn_impl_meta.k_block_size;
  params.fixed_overhead_num_blocks = attn_impl_meta.fixed_overhead_num_blocks;
  params.num_sm_parts = attn_impl_meta.num_sm_parts;
  params.topk = topk;
  run_get_mla_metadata_kernel(params, stream);
}

// `cudaMalloc()` is guaranteed to be aligned to at least 256 bytes
constexpr size_t kAlignment = 256 / sizeof(float);

// Helper function to align size
inline size_t AlignSize(const size_t size) {
  static_assert((kAlignment & (kAlignment - 1)) == 0, "Expect kAlignment to be a power of 2");
  return (size + kAlignment - 1) & ~(kAlignment - 1);
};

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(SCALAR_T* q_ptr, CACHE_T* kcache_ptr, int batch_size, int seqlen_q_ori,
                                     int num_heads_q, int num_heads_k, int head_size_k, int head_size_v,
                                     int k_batch_stride, int max_num_blocks_per_seq, int num_blocks, int block_size,
                                     int num_sm_parts, int* seqlens_k_ptr, int* block_table_ptr, float softmax_scale,
                                     bool is_causal, int* tile_scheduler_metadata_ptr, int* num_splits_ptr, bool is_fp8,
                                     int* indices_ptr, int topk, cudaStream_t stream, float* workspace_ptr,
                                     SCALAR_T* out_ptr) {
  bool is_sparse_attn = indices_ptr != nullptr;

  if (!is_fp8) {
    KLLM_KERNEL_CHECK_WITH_INFO((std::is_same_v<SCALAR_T, CACHE_T>), "query and key must have the same dtype");
  } else {
    KLLM_KERNEL_CHECK_WITH_INFO((std::is_same_v<CACHE_T, uint8_t>), "key must have dtype int8");
  }

  KLLM_KERNEL_CHECK_WITH_INFO(head_size_k == 576, "Only head_size_k == 576 is supported");
  KLLM_KERNEL_CHECK_WITH_INFO(head_size_v == 512, "Only head_size_v == 576 is supported");
  KLLM_KERNEL_CHECK_WITH_INFO(block_size == 64, "Currently page_block_size must be 64");
  KLLM_KERNEL_CHECK_WITH_INFO(batch_size > 0, "batch size must be positive");
  KLLM_KERNEL_CHECK_WITH_INFO(num_heads_q % num_heads_k == 0,
                              "Number of heads in key/value must divide number of heads in query");

  if (seqlen_q_ori == 1) {
    is_causal = false;
  }

  const int num_q_heads_per_hk = num_heads_q / num_heads_k;
  const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
  const int num_heads = num_heads_k;
  const int total_num_splits = batch_size + num_sm_parts;

  const std::vector<int> q_shape{batch_size, q_seq_per_hk, num_heads, head_size_k};
  const std::vector<int> block_table_shape{batch_size, max_num_blocks_per_seq};
  const auto kcache_shape = (is_fp8 ? std::vector<int>{num_blocks, block_size, num_heads_k, 656}
                                    : std::vector<int>{num_blocks, block_size, num_heads_k, head_size_k});
  const std::vector<int> out_shape{batch_size, q_seq_per_hk, num_heads, head_size_v};
  const std::vector<int> softmax_lse_shape{batch_size, num_heads, q_seq_per_hk};
  const std::vector<int> indices_shape{batch_size, seqlen_q_ori, topk};
  const std::vector<int> softmax_lse_accum_shape{total_num_splits, num_heads, q_seq_per_hk};
  const std::vector<int> out_accum_shape{total_num_splits, num_heads, q_seq_per_hk, head_size_v};

  const auto q_stride = GetStride(q_shape);
  const auto block_table_stride = GetStride(block_table_shape);
  const auto kcache_stride = GetStride(kcache_shape);
  const auto out_stride = GetStride(out_shape);
  const auto softmax_lse_stride = GetStride(softmax_lse_shape);
  const auto indices_stride = GetStride(indices_shape);
  const auto softmax_lse_accum_stride = GetStride(softmax_lse_accum_shape);

  float* softmax_lse_ptr = workspace_ptr;
  workspace_ptr += AlignSize(softmax_lse_shape[0] * softmax_lse_stride[0]);
  float* softmax_lse_accum_ptr = workspace_ptr;
  workspace_ptr += AlignSize(softmax_lse_accum_shape[0] * softmax_lse_accum_stride[0]);
  float* out_accum_ptr = workspace_ptr;

  DecodingParams params = {};
  // Set the sizes.
  params.b = batch_size;
  params.s_q = seqlen_q_ori;
  params.q_seq_per_hk = q_seq_per_hk;
  params.seqlens_k_ptr = seqlens_k_ptr;
  params.h_q = num_heads_q;
  params.h_k = num_heads_k;
  params.num_blocks = num_blocks;
  params.q_head_per_hk = num_q_heads_per_hk;
  params.is_causal = is_causal;
  params.d = head_size_k;
  params.d_v = head_size_v;
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = float(softmax_scale * M_LOG2E);
  params.topk = topk;
  // Set the pointers and strides.
  params.q_ptr = q_ptr;
  params.k_ptr = kcache_ptr;
  params.o_ptr = out_ptr;
  params.indices_ptr = indices_ptr;
  params.softmax_lse_ptr = softmax_lse_ptr;
  // All stride are in elements, not bytes.
  params.q_batch_stride = q_stride[0];
  params.k_batch_stride = k_batch_stride;  // k_batch_stride can be larger than kcache_stride[0]
  params.o_batch_stride = out_stride[0];
  params.q_row_stride = q_stride[1];
  params.k_row_stride = kcache_stride[1];
  params.o_row_stride = out_stride[1];
  params.q_head_stride = q_stride[2];
  params.k_head_stride = kcache_stride[2];
  params.o_head_stride = out_stride[2];
  params.indices_batch_stride = is_sparse_attn ? indices_stride[0] : 0;
  params.indices_row_stride = is_sparse_attn ? indices_stride[1] : 0;

  params.block_table = block_table_ptr;
  params.block_table_batch_stride = block_table_stride[0];
  params.page_block_size = block_size;

  params.tile_scheduler_metadata_ptr = tile_scheduler_metadata_ptr;
  params.num_sm_parts = num_sm_parts;
  params.num_splits_ptr = num_splits_ptr;

  params.total_num_splits = total_num_splits;
  params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
  params.oaccum_ptr = out_accum_ptr;

  if constexpr (std::is_same_v<SCALAR_T, half>) {
#  ifdef FLASH_MLA_DISABLE_FP16
    KLLM_KERNEL_THROW(
        "FlashMLA is compiled with -DFLASH_MLA_DISABLE_FP16. Please remove this flag from your environment and "
        "re-compile FlashMLA.");
#  endif
  }

  if (is_sparse_attn) {
    if (is_fp8) {
      KLLM_KERNEL_CHECK_WITH_INFO((std::is_same_v<SCALAR_T, __nv_bfloat16>),
                                  "Sparse FP8 MLA only supports BFloat16 on SM90");
      sm90::run_flash_splitkv_mla_fp8_sparse_kernel(params, stream);
    } else {
      KLLM_KERNEL_THROW("Only FP8 kvcahe is supported for sparse MLA on SM90");
    }
  } else {
    if (is_fp8) {
      KLLM_KERNEL_THROW("Dense FP8 MLA is not supported on SM90");
    } else {
      if (std::is_same_v<SCALAR_T, __nv_bfloat16>) {
        sm90::run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
      } else if (std::is_same_v<SCALAR_T, half>) {
#  ifndef FLASH_MLA_DISABLE_FP16
        sm90::run_flash_splitkv_mla_kernel<cutlass::half_t>(params, stream);
#  endif
      } else {
        KLLM_KERNEL_THROW("Unsupported dtype for dense MLA on SM90");
      }
    }
  }
  if constexpr (std::is_same_v<SCALAR_T, __nv_bfloat16>) {
    run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
  } else if constexpr (std::is_same_v<SCALAR_T, half>) {
#  ifndef FLASH_MLA_DISABLE_FP16
    run_flash_mla_combine_kernel<cutlass::half_t>(params, stream);
#  endif
  }
}
#  define INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(SCALAR_T, CACHE_T, KV_DTYPE)                                         \
    template void InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(                                    \
        SCALAR_T*, CACHE_T*, int, int, int, int, int, int, int, int, int, int, int, int*, int*, float, bool, int*, \
        int*, bool, int*, int, cudaStream_t, float*, SCALAR_T*)
INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
#  undef INVOKE_SPARSE_FLASH_MLA_WITH_KVCATH

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaPrefill(SCALAR_T* q_ptr, CACHE_T* kcache_ptr, int* indices_ptr, int seqlen_q, int seqlen_k,
                                 int num_heads_q, int num_heads_k, int head_size_k, int head_size_v, int topk,
                                 float sm_scale, cudaStream_t stream, float* workspace_ptr, SCALAR_T* out_ptr) {
  if constexpr (!std::is_same_v<SCALAR_T, __nv_bfloat16>) {
    KLLM_KERNEL_THROW("Sparse Attention Forward Kernel (sparse_prefill_fwd) only supports bfloat16");
  }

  float* max_logits_ptr = workspace_ptr;
  workspace_ptr += AlignSize(seqlen_q * num_heads_q);
  float* lse_ptr = workspace_ptr;

  SparsePrefillParams params = {seqlen_q,
                                seqlen_k,
                                num_heads_q,
                                num_heads_k,
                                head_size_k,
                                head_size_v,
                                topk,
                                sm_scale,
                                sm_scale * 1.44269504f,
                                (cutlass::bfloat16_t*)q_ptr,
                                (cutlass::bfloat16_t*)kcache_ptr,
                                indices_ptr,
                                num_heads_q * head_size_k,  // q.stride(0)
                                head_size_k,                // q.stride(1)
                                num_heads_k * head_size_k,  // kcache.stride(0)
                                head_size_k,                // kcache.stride(1)
                                num_heads_k * topk,         // indices.stride(0)
                                topk,                       // indices.stride(1)
                                (cutlass::bfloat16_t*)out_ptr,
                                max_logits_ptr,
                                lse_ptr,
                                stream};

  sm90::run_fwd_kernel(params);
}
#  define INVOKE_FLASH_SPARSE_MLA_PREFILL(SCALAR_T, CACHE_T, KV_DTYPE)      \
    template void InvokeFlashSparseMlaPrefill<SCALAR_T, CACHE_T, KV_DTYPE>( \
        SCALAR_T*, CACHE_T*, int*, int, int, int, int, int, int, int, float, cudaStream_t, float*, SCALAR_T*)
INVOKE_FLASH_SPARSE_MLA_PREFILL(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#  undef INVOKE_SPARSE_FLASH_MLA_PREFILL

}  // namespace nvidia
}  // namespace llm_kernels
#else
namespace llm_kernels {
namespace nvidia {

DecodingAttnImplMeta GetAttnImplMeta(int num_q_tokens_per_head_k, int num_heads_k, int num_heads_q, bool is_fp8_kvcache,
                                     bool is_sparse_attn) {
  return DecodingAttnImplMeta{};
}

void InvokeGetSparseMlaMetadata(int* seqlens_k_ptr, int batch_size, int num_q_tokens_per_head_k, int num_heads_k,
                                int num_heads_q, bool is_fp8_kvcache, int topk, cudaStream_t stream,
                                int* tile_scheduler_metadata_ptr, int* num_splits_ptr) {}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(SCALAR_T* q_ptr, CACHE_T* kcache_ptr, int batch_size, int seqlen_q_ori,
                                     int num_heads_q, int num_heads_k, int head_size_k, int head_size_v,
                                     int k_batch_stride, int max_num_blocks_per_seq, int num_blocks, int block_size,
                                     int num_sm_parts, int* seqlens_k_ptr, int* block_table_ptr, float softmax_scale,
                                     bool is_causal, int* tile_scheduler_metadata_ptr, int* num_splits_ptr, bool is_fp8,
                                     int* indices_ptr, int topk, cudaStream_t stream, float* workspace_ptr,
                                     SCALAR_T* out_ptr) {}
#  define INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(SCALAR_T, CACHE_T, KV_DTYPE)                                         \
    template void InvokeFlashSparseMlaWithKVCache<SCALAR_T, CACHE_T, KV_DTYPE>(                                    \
        SCALAR_T*, CACHE_T*, int, int, int, int, int, int, int, int, int, int, int, int*, int*, float, bool, int*, \
        int*, bool, int*, int, cudaStream_t, float*, SCALAR_T*)
INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
INVOKE_FLASH_SPARSE_MLA_WITH_KVCATH(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
#  undef INVOKE_SPARSE_FLASH_MLA_WITH_KVCATH

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaPrefill(SCALAR_T* q_ptr, CACHE_T* kcache_ptr, int* indices_ptr, int seqlen_q, int seqlen_k,
                                 int num_heads_q, int num_heads_k, int head_size_k, int head_size_v, int topk,
                                 float sm_scale, cudaStream_t stream, float* workspace_ptr, SCALAR_T* out_ptr) {}
#  define INVOKE_FLASH_SPARSE_MLA_PREFILL(SCALAR_T, CACHE_T, KV_DTYPE)      \
    template void InvokeFlashSparseMlaPrefill<SCALAR_T, CACHE_T, KV_DTYPE>( \
        SCALAR_T*, CACHE_T*, int*, int, int, int, int, int, int, int, float, cudaStream_t, float*, SCALAR_T*)
INVOKE_FLASH_SPARSE_MLA_PREFILL(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#  undef INVOKE_SPARSE_FLASH_MLA_PREFILL

}  // namespace nvidia
}  // namespace llm_kernels
#endif
