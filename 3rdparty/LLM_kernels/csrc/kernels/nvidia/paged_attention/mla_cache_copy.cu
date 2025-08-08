/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "mla_cache_copy.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "quant_utils.cuh"

namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID_Y 65535

// Copy MLA-kv compressed kv cache to kv blocks.
// This kernel will skip shared prefix blocks, and copy only the kv cache of tokesn that need to be calculated.
//
// Args:
//   k_src:
//     The k need to be copied, not contain prefix part.
//   v_src:
//     The v need to be copyied, not contain prefix part.
//   k_list:
//     A pointer array that contain k block's addr, include the prefix blocks.
//   v_list:
//     A pointer array that contain v block's addr, include the prefix blocks.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [bs + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   without_prefix_offsets:
//     The offset of non-prefix token num.
//     It's shape is [bs + 1], in format [0, seq1_unique_len, seq1_unique_len + seq2_unique_len, ...]
//   block_offsets:
//     The offset of block num of every seq, include prefix part.
//     It's shape is [bs + 1], in format [0, seq1_block_num, seq1_block_num + seq2_block_num]
//   block_size:
//     The token number of every cache block.
//   bs:
//     The batch size.
//   total_len:
//     The toeken number of all tokens, not include prefix part.
//   k_stride_size:
//     The stride of the k cache of one token.
//   v_stride_size:
//     The stride of the v cache of one token.
//   k_scale:
//     The quantization scale of k value.
//   v_scale:
//     The quantization scale of v value.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void KVCacheCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                  size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                  int block_size, int bs, int total_len, int k_stride_size, int v_stride_size,
                                  float k_scale, float v_scale) {
  int token_idx = blockIdx.y + blockIdx.z * gridDim.y;
  if (token_idx >= total_len) {
    return;
  }

  int batch_idx = 0;
  for (batch_idx = 0; batch_idx < bs; batch_idx++) {
    if (token_idx < without_prefix_offsets[batch_idx + 1]) {
      break;
    }
  }

  // The token index in current batch, with prefix part.
  int token_idx_in_batch_with_prefix =
      (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (token_idx - without_prefix_offsets[batch_idx]);
  // The block offset in current batch.
  int block_offset_in_batch = token_idx_in_batch_with_prefix / block_size;
  // The token offset in current block.
  int token_offset_in_block = token_idx_in_batch_with_prefix % block_size;

  CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[batch_idx] + block_offset_in_batch]);
  CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + block_offset_in_batch]);
  SCALAR_T* k_src_ptr = k_src + token_idx * k_stride_size;
  SCALAR_T* v_src_ptr = v_src + token_idx * v_stride_size;

  // For every token, the content is :
  // +------------------+---------+
  // |v1 v2 v3 v4 v5 v6 | k1 k2 k3|
  // +------------------+---------+
  // |    v_stride      | k_stride|
  // +------------------+---------+

  // Process k
  for (int head_size_i = threadIdx.x; head_size_i < k_stride_size; head_size_i += blockDim.x) {
    int k_dst_index = token_offset_in_block * (k_stride_size + v_stride_size) + (v_stride_size + head_size_i);
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      k_dst_base[k_dst_index] = k_src_ptr[head_size_i];
    } else {
      k_dst_base[k_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[head_size_i], k_scale);
    }
  }

  // Process v
  for (int head_size_i = threadIdx.x; head_size_i < v_stride_size; head_size_i += blockDim.x) {
    int v_dst_index = token_offset_in_block * (k_stride_size + v_stride_size) + head_size_i;
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      v_dst_base[v_dst_index] = v_src_ptr[head_size_i];
    } else {
      v_dst_base[v_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(v_src_ptr[head_size_i], v_scale);
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaKVCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* prefix_offsets,
                    size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len,
                    int k_stride_size, int v_stride_size, float k_scale, float v_scale, cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(1, grid_y, grid_z);

  dim3 block_shape(std::min(std::max(k_stride_size, v_stride_size), MAX_THREADS_PER_BLOCK));
  KVCacheCopyKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, prefix_offsets, without_prefix_offsets, block_offsets, block_size, bs, total_len,
      k_stride_size, v_stride_size, k_scale, v_scale);
}

template <typename CACHE_T, typename VEC_T>
__global__ void MlaGetFromCompressedCacheKernel(void* const k_rope_out, void* const latent_out,
                                                const void* const* const block_list, const size_t* const seq_len_offset,
                                                const int* const block_offsets, const int block_size,
                                                const int k_rope_size, const int latent_size) {
  const size_t token_idx = blockIdx.x;
  const size_t copy_offset_bytes = threadIdx.x * sizeof(VEC_T);

  size_t batch_idx = 0;
  while (token_idx >= seq_len_offset[batch_idx + 1]) {
    ++batch_idx;
  }

  const size_t token_offset_in_req = token_idx - seq_len_offset[batch_idx];
  const size_t block_offset_in_req = token_offset_in_req / block_size;
  const size_t token_offset_in_block = token_offset_in_req % block_size;

  const void* const block_base = block_list[block_offsets[batch_idx] + block_offset_in_req];

  // data in cache is: [latent, k_rope, latent, k_rope...]
  const size_t src_offset_bytes =
      token_offset_in_block * (k_rope_size + latent_size) * sizeof(CACHE_T) + copy_offset_bytes;

  const bool is_latent = copy_offset_bytes < latent_size * sizeof(CACHE_T);
  const size_t current_item_size = is_latent ? latent_size : k_rope_size;
  const size_t dst_item_offset = copy_offset_bytes - (is_latent ? 0 : latent_size * sizeof(CACHE_T));
  const size_t dst_offset_bytes = token_idx * current_item_size * sizeof(CACHE_T) + dst_item_offset;
  void* const dst = is_latent ? latent_out : k_rope_out;

  *reinterpret_cast<VEC_T*>(dst + dst_offset_bytes) = *reinterpret_cast<const VEC_T*>(block_base + src_offset_bytes);
}

// copy k_rope and latent from kv cache block to continuous buffer k_rope_out and latent_out
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaGetFromCompressedCache(void* const k_rope_out, void* const latent_out, const void* const* const block_list,
                               const int total_len, const size_t* const seq_len_offset, const int* const block_offsets,
                               const int block_size, const int k_rope_size, const int latent_size,
                               cudaStream_t stream) {
  const size_t item_size = k_rope_size + latent_size;
  const dim3 grid(total_len);
  const dim3 block(item_size * sizeof(CACHE_T) / sizeof(float4));
  assert(block.x <= MAX_THREADS_PER_BLOCK && item_size * sizeof(CACHE_T) % sizeof(float4) == 0);

  MlaGetFromCompressedCacheKernel<CACHE_T, float4><<<grid, block, 0, stream>>>(
      k_rope_out, latent_out, block_list, seq_len_offset, block_offsets, block_size, k_rope_size, latent_size);
}

#define MLA_CACHE_COPY_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                               \
  template void MlaKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(                                                           \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* prefix_offsets,                        \
      size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int k_stride_size,    \
      int v_stride_size, float k_scale, float v_scale, cudaStream_t stream);                                           \
  template void MlaGetFromCompressedCache<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      void* const k_rope_out, void* const latent_out, const void* const* const block_list, const int total_len,        \
      const size_t* const seq_len_offset, const int* const block_offsets, const int block_size, const int k_rope_size, \
      const int latent_size, cudaStream_t stream);

MLA_CACHE_COPY_FUNCTION_DECLARATION(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#undef MLA_CACHE_COPY_FUNCTION_DECLARATION

}  // namespace nvidia
}  // namespace llm_kernels
