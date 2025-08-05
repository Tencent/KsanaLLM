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

__device__ int k_chunk_size = 16;

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

// Extend memory buffer, used to reset K & V size for flash attn in prefill stage.
// This kernel will extand
//     [num_tokens_without_prefix, num_heads, head_size]
// to
//     [num_tokens_with_prefix, num_heads, head_size]
//
// Args:
//   k_src:
//   v_src: The k & v source buffer.
//   k_out:
//   v_out: The k & v destinct buffer.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   without_prefix_offsets:
//     The offset of non-prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_unique_len, seq1_unique_len + seq2_unique_len, ...]
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void ExtendKVPrefixWithEmptyKernel(SCALAR_T* k_src, SCALAR_T* v_src, SCALAR_T* k_out, SCALAR_T* v_out,
                                              size_t* prefix_offsets, size_t* without_prefix_offsets) {
  const size_t without_prefix_token_idx = blockIdx.x;
  const size_t head_idx = blockIdx.y;
  const size_t num_head = gridDim.y;
  const size_t head_size = blockDim.x;

  size_t batch_idx = 0;
  while (without_prefix_token_idx >= without_prefix_offsets[batch_idx + 1]) {
    ++batch_idx;
  }

  const size_t total_token_idx = without_prefix_token_idx + prefix_offsets[batch_idx + 1];

  const size_t src_num_offset = (without_prefix_token_idx * num_head + head_idx) * head_size + threadIdx.x;
  const size_t dst_num_offset = (total_token_idx * num_head + head_idx) * head_size + threadIdx.x;
  k_out[dst_num_offset] = k_src[src_num_offset];
  v_out[dst_num_offset] = v_src[src_num_offset];
}

// Copy prefix k_pe to memory buffer, used to prepare k-input for flash attn.
// This kernel will replicate k_pe to all dst head_nums, the result buffer shape is:
//     [num_tokens_with_prefix, num_heads, qk_nope_head_dim+qk_rope_head_dim]
//
// Args:
//   k_out: The k output buffer.
//   k_list: The k cache block pointer list.
//   src_stride_size: The stride size of k block.
//   src_copy_offset: The offset of data to copy from.
//   dst_head_size: The head size of destinct k.
//   dst_copy_offset: The offset of data to copy to.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   without_prefix_offsets:
//     The offset of non-prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_unique_len, seq1_unique_len + seq2_unique_len, ...]
//   block_offsets:
//     The offset of block num of every seq, include prefix part.
//     It's shape is [bs + 1], in format [0, seq1_block_num, seq1_block_num + seq2_block_num]
//   block_size:
//     The token number of every cache block.
// Vectorized kernel templates for different vector sizes
template <typename SCALAR_T, typename CACHE_T, typename VEC_T>
__global__ void CopyKeyBlockWithReplicationKernel(SCALAR_T* k_out, void** k_list, int src_stride_size,
                                                  int src_copy_offset, int dst_head_size, int dst_copy_offset,
                                                  size_t* prefix_offsets, size_t* without_prefix_offsets,
                                                  int* block_offsets, int block_size, int num_head) {
  const size_t prefix_token_idx = blockIdx.x;
  const size_t head_block_idx = blockIdx.y;
  const size_t head_idx = head_block_idx * blockDim.y + threadIdx.y;

  const size_t copy_idx = threadIdx.x * sizeof(VEC_T);
  size_t batch_idx = 0;
  while (prefix_token_idx >= prefix_offsets[batch_idx + 1]) {
    ++batch_idx;  // Find which batch this token belongs to
  }
  // Position of this token in the request
  const size_t token_offset_in_req = prefix_token_idx - prefix_offsets[batch_idx];
  // Which block in the request this token belongs to
  const size_t block_offset_in_req = token_offset_in_req / block_size;
  // Specific position within the block
  const size_t token_offset_in_block = token_offset_in_req % block_size;
  // Position after flattening all batches
  const size_t total_token_idx = prefix_token_idx + without_prefix_offsets[batch_idx];

  // Get the base address of the source block
  void* const k_block_base = k_list[block_offsets[batch_idx] + block_offset_in_req];

  // Calculate byte offsets
  const size_t src_offset_bytes =
      (token_offset_in_block * src_stride_size + src_copy_offset) * sizeof(CACHE_T) + copy_idx;
  const size_t dst_offset_bytes =
      (total_token_idx * num_head * dst_head_size + head_idx * dst_head_size + dst_copy_offset) * sizeof(SCALAR_T) +
      copy_idx;

  // Get pointers to the source and destination
  void* src_ptr = k_block_base + src_offset_bytes;
  void* dst_ptr = reinterpret_cast<void*>(k_out) + dst_offset_bytes;

  // Copy the data using vector type for efficient memory transfer
  *reinterpret_cast<VEC_T*>(dst_ptr) = *reinterpret_cast<VEC_T*>(src_ptr);
}

// Copy prefix v block to memory buffer, used to prepare v-input for flash attn.
// This kernel will copy compressed_kv to a memory buffer of shape:
//     [num_prefix_tokens, kv_lora_rank]
//
// Args:
//   v_out: The v output buffer.
//   v_list: The cache block pointer list.
//   src_stride_size: The stride size of k block.
//   src_copy_offset: The offset of data to copy from.
//   dst_stride_size: The stride size of destinct memory buffer.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   block_offsets:
//     The offset of block num of every seq, include prefix part.
//     It's shape is [bs + 1], in format [0, seq1_block_num, seq1_block_num + seq2_block_num]
//   block_size:
//     The token number of every cache block.
template <typename SCALAR_T, typename CACHE_T, typename VEC_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void CopyValueBlockToBufferKernel(SCALAR_T* v_out, void** v_list, int src_stride_size, int src_copy_offset,
                                             int dst_stride_size, size_t* prefix_offsets, int* block_offsets,
                                             int block_size) {
  const size_t prefix_token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const size_t copy_idx = threadIdx.x * sizeof(VEC_T);  // Vectorized byte offset.
  size_t batch_idx = 0;
  while (prefix_token_idx >= prefix_offsets[batch_idx + 1]) {
    ++batch_idx;
  }
  const size_t token_offset_in_req = prefix_token_idx - prefix_offsets[batch_idx];
  const size_t block_offset_in_req = token_offset_in_req / block_size;
  const size_t token_offset_in_block = token_offset_in_req % block_size;
  CACHE_T* v_block_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + block_offset_in_req]);
  // Base address of the value block.
  const size_t src_offset_bytes =
      (token_offset_in_block * src_stride_size + src_copy_offset) * sizeof(CACHE_T) + copy_idx;
  const size_t dst_offset_bytes = prefix_token_idx * dst_stride_size * sizeof(SCALAR_T) + copy_idx;
  // Vectorized read and write operations.
  void* src_ptr = reinterpret_cast<void*>(v_block_base) + src_offset_bytes;
  void* dst_ptr = reinterpret_cast<void*>(v_out) + dst_offset_bytes;
  *reinterpret_cast<VEC_T*>(dst_ptr) = *reinterpret_cast<VEC_T*>(src_ptr);
}

// Fill prefix kv to memory buffer, used to prepare kv-input for flash attn.
// This kernel will fill
//     [num_prefix_tokens, num_heads, qk_nope_head_dim]
// to
//     [num_total_tokens, num_heads, qk_nope_head_dim+qk_rope_head_dim]
//
// Args:
//   k_out:
//   v_out: The kv destinct buffer.
//   k_src:
//   v_src: The kv source buffer.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   without_prefix_offsets:
//     The offset of non-prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_unique_len, seq1_unique_len + seq2_unique_len, ...]
//   src_head_size:    The dim of src.
//   dst_head_size:    The dim of dst.
template <typename SCALAR_T>
__global__ void FillKVPrefixKernel(SCALAR_T* k_out, SCALAR_T* v_out, const SCALAR_T* k_src, const SCALAR_T* v_src,
                                   size_t* prefix_offsets, size_t* without_prefix_offsets, int src_head_size,
                                   int dst_head_size) {
  const size_t prefix_token_idx = blockIdx.x;
  const size_t head_idx = blockIdx.y;
  const size_t num_head = gridDim.y;

  size_t batch_idx = 0;
  while (prefix_token_idx >= prefix_offsets[batch_idx + 1]) {
    ++batch_idx;
  }

  const size_t total_token_idx = prefix_token_idx + without_prefix_offsets[batch_idx];

  const size_t src_num_offset = (prefix_token_idx * num_head + head_idx) * src_head_size + threadIdx.x;
  const size_t dst_num_offset = (total_token_idx * num_head + head_idx) * dst_head_size + threadIdx.x;
  k_out[dst_num_offset] = k_src[src_num_offset];
  v_out[dst_num_offset] = v_src[src_num_offset];
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

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaExtendKVPrefixWithEmpty(SCALAR_T* k_src, SCALAR_T* v_src, SCALAR_T* k_out, SCALAR_T* v_out,
                                size_t* prefix_offsets, size_t* without_prefix_offsets, int num_heads, int head_size,
                                int total_len_without_prefix, cudaStream_t stream) {
  const dim3 grid(total_len_without_prefix, num_heads);
  const dim3 block(head_size);
  ExtendKVPrefixWithEmptyKernel<SCALAR_T, CACHE_T, KV_DTYPE>
      <<<grid, block, 0, stream>>>(k_src, v_src, k_out, v_out, prefix_offsets, without_prefix_offsets);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaCopyValueBlockToBuffer(SCALAR_T* v_out, void** v_list, int src_stride_size, int src_copy_offset,
                               int src_copy_len, int dst_stride_size, size_t* prefix_offsets, int* block_offsets,
                               int block_size, int total_prefix_len, cudaStream_t stream) {
  constexpr size_t ScalarSize = sizeof(SCALAR_T);
  constexpr size_t vectorized16Bytes = sizeof(float4);
  constexpr size_t vectorized8Bytes = sizeof(float2);
  int vectorized_size = ScalarSize;
  int copy_bytes = src_copy_len * ScalarSize;
  if (copy_bytes % vectorized16Bytes == 0) {
    vectorized_size = vectorized16Bytes;
  } else if (copy_bytes % vectorized8Bytes == 0) {
    vectorized_size = vectorized8Bytes;
  }
  int num_vectorized_copies = copy_bytes / vectorized_size;
  int block_num_prefix = 1;
  int max_prefix_lens_per_block = MAX_THREADS_PER_BLOCK / num_vectorized_copies;
  for (int i = max_prefix_lens_per_block; i >= 2; i--) {
    if (total_prefix_len % i == 0) {
      block_num_prefix = i;
      break;
    }
  }
  if (block_num_prefix * num_vectorized_copies < 64) {
    num_vectorized_copies = src_copy_len;
    vectorized_size = ScalarSize;
    block_num_prefix = 1;
  }
  int grid_num_prefix = total_prefix_len / block_num_prefix;
  const dim3 grid(grid_num_prefix);
  const dim3 block(num_vectorized_copies, block_num_prefix);
  if (vectorized_size == vectorized16Bytes) {
    CopyValueBlockToBufferKernel<SCALAR_T, CACHE_T, float4, KV_DTYPE><<<grid, block, 0, stream>>>(
        v_out, v_list, src_stride_size, src_copy_offset, dst_stride_size, prefix_offsets, block_offsets, block_size);
  } else if (vectorized_size == vectorized8Bytes) {
    CopyValueBlockToBufferKernel<SCALAR_T, CACHE_T, float2, KV_DTYPE><<<grid, block, 0, stream>>>(
        v_out, v_list, src_stride_size, src_copy_offset, dst_stride_size, prefix_offsets, block_offsets, block_size);
  } else {
    using ScalarVec = SCALAR_T;
    CopyValueBlockToBufferKernel<SCALAR_T, CACHE_T, SCALAR_T, KV_DTYPE><<<grid, block, 0, stream>>>(
        v_out, v_list, src_stride_size, src_copy_offset, dst_stride_size, prefix_offsets, block_offsets, block_size);
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaCopyKeyBlockWithReplication(SCALAR_T* k_out, void** k_list, int src_stride_size, int src_copy_offset,
                                    int src_copy_len, int dst_num_heads, int dst_head_size, int dst_copy_offset,
                                    size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                    int block_size, int total_prefix_len, cudaStream_t stream) {
  // Check memory alignment for vectorization
  constexpr size_t ScalarSize = sizeof(SCALAR_T);
  constexpr size_t vectorized16Bytes = sizeof(float4);
  constexpr size_t vectorized8Bytes = sizeof(float2);
  int vectorized_size = ScalarSize;
  int copy_bytes = src_copy_len * ScalarSize;
  if (copy_bytes % vectorized16Bytes == 0) {
    vectorized_size = vectorized16Bytes;
  } else if (copy_bytes % vectorized8Bytes == 0) {
    vectorized_size = vectorized8Bytes;
  }

  int num_vectorized_copies = copy_bytes / vectorized_size;
  int max_heads_per_block = MAX_THREADS_PER_BLOCK / num_vectorized_copies;
  int heads_per_block = (max_heads_per_block > 0) ? std::min(dst_num_heads, max_heads_per_block) : 1;
  int grid_y = dst_num_heads / heads_per_block;
  if (dst_num_heads % heads_per_block != 0) {
    grid_y = dst_num_heads;
    heads_per_block = 1;
    num_vectorized_copies = src_copy_len;
    vectorized_size = ScalarSize;
  }
  const dim3 grid(total_prefix_len, grid_y);
  const dim3 block(num_vectorized_copies, heads_per_block);

  if (vectorized_size == vectorized16Bytes) {
    // Use float4 vectorization (16 bytes)
    CopyKeyBlockWithReplicationKernel<SCALAR_T, CACHE_T, float4>
        <<<grid, block, 0, stream>>>(k_out, k_list, src_stride_size, src_copy_offset, dst_head_size, dst_copy_offset,
                                     prefix_offsets, without_prefix_offsets, block_offsets, block_size, dst_num_heads);
  } else if (vectorized_size == vectorized8Bytes) {
    // Use float2 vectorization (8 bytes)
    CopyKeyBlockWithReplicationKernel<SCALAR_T, CACHE_T, float2>
        <<<grid, block, 0, stream>>>(k_out, k_list, src_stride_size, src_copy_offset, dst_head_size, dst_copy_offset,
                                     prefix_offsets, without_prefix_offsets, block_offsets, block_size, dst_num_heads);
  } else {
    // Fallback to scalar operations
    using ScalarVec = SCALAR_T;
    CopyKeyBlockWithReplicationKernel<SCALAR_T, CACHE_T, ScalarVec>
        <<<grid, block, 0, stream>>>(k_out, k_list, src_stride_size, src_copy_offset, dst_head_size, dst_copy_offset,
                                     prefix_offsets, without_prefix_offsets, block_offsets, block_size, dst_num_heads);
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFillKVPrefix(SCALAR_T* k_out, SCALAR_T* v_out, SCALAR_T* k_src, SCALAR_T* v_src, size_t* prefix_offsets,
                     size_t* without_prefix_offsets, int num_heads, int src_head_size, int dst_head_size,
                     int total_prefix_len, cudaStream_t stream) {
  const dim3 grid(total_prefix_len, num_heads);
  const dim3 block(src_head_size);
  FillKVPrefixKernel<SCALAR_T><<<grid, block, 0, stream>>>(k_out, v_out, k_src, v_src, prefix_offsets,
                                                           without_prefix_offsets, src_head_size, dst_head_size);
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
  template void MlaExtendKVPrefixWithEmpty<SCALAR_T, CACHE_T, KV_DTYPE>(                                               \
      SCALAR_T * k_src, SCALAR_T * v_src, SCALAR_T * k_out, SCALAR_T * v_out, size_t* prefix_offsets,                  \
      size_t* without_prefix_offsets, int num_heads, int head_size, int total_len_without_prefix,                      \
      cudaStream_t stream);                                                                                            \
  template void MlaCopyKeyBlockWithReplication<SCALAR_T, CACHE_T, KV_DTYPE>(                                           \
      SCALAR_T * k_out, void** k_list, int src_stride_size, int src_copy_offset, int src_copy_len, int dst_num_heads,  \
      int dst_head_size, int dst_copy_offset, size_t* prefix_offsets, size_t* without_prefix_offsets,                  \
      int* block_offsets, int block_size, int total_prefix_len, cudaStream_t stream);                                  \
  template void MlaCopyValueBlockToBuffer<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      SCALAR_T * v_out, void** v_list, int src_stride_size, int src_copy_offset, int src_copy_len,                     \
      int dst_stride_size, size_t* prefix_offsets, int* block_offsets, int block_size, int total_prefix_len,           \
      cudaStream_t stream);                                                                                            \
  template void MlaFillKVPrefix<SCALAR_T, CACHE_T, KV_DTYPE>(                                                          \
      SCALAR_T * k_out, SCALAR_T * v_out, SCALAR_T * k_src, SCALAR_T * v_src, size_t* prefix_offsets,                  \
      size_t* without_prefix_offsets, int num_heads, int src_head_size, int dst_head_size, int total_prefix_len,       \
      cudaStream_t stream);                                                                                            \
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
