/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "mla_cache_copy.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cassert>
#include <cfloat>

#include "csrc/kernels/nvidia/common/vec_dtypes.cuh"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "quant_utils.cuh"

namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID_Y 65535

template <int kTopk>
__global__ void FlashSparseMlaConvertBlockTableKernel(int* indices, int* block_table, size_t* without_prefix_offsets,
                                                      int batch_size, int max_num_blocks_per_seq, int block_size) {
  const int token_idx = blockIdx.x >> 1;
  int batch_idx = 0;
  // Batch size of prefill is usually small, just use a simple loop here
  for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (token_idx < without_prefix_offsets[batch_idx + 1]) {
      break;
    }
  }

  int* const current_indices = indices + token_idx * kTopk + threadIdx.x + (blockIdx.x & 1 ? MAX_THREADS_PER_BLOCK : 0);
  // current_topk_token_idx must have the same batch_idx as token_idx
  const int current_topk_token_idx = *current_indices;
  const int token_offset = current_topk_token_idx;

  if (current_topk_token_idx >= 0) {
    *current_indices = block_table[batch_idx * max_num_blocks_per_seq + token_offset / block_size] * block_size +
                       token_offset % block_size;
  }
}

void FlashSparseMlaConvertBlockTable(int* indices, int* block_table, size_t* without_prefix_offsets, int token_num,
                                     int topk, int batch_size, int max_num_blocks_per_seq, int block_size,
                                     cudaStream_t stream) {
  constexpr int kTopk = 2048;
  assert(topk == kTopk);
  assert(kTopk == 2 * MAX_THREADS_PER_BLOCK);
  // Every two threads handle the topk convertion for one token
  dim3 grid_shape(token_num << 1);
  dim3 block_shape(MAX_THREADS_PER_BLOCK);
  FlashSparseMlaConvertBlockTableKernel<kTopk><<<grid_shape, block_shape, 0, stream>>>(
      indices, block_table, without_prefix_offsets, batch_size, max_num_blocks_per_seq, block_size);
}

template <int kTopk>
__global__ void PagedSparseMlaConvertBlockTableKernel(int* indices, int* block_table, int q_seq_len,
                                                      int max_num_blocks_per_seq, int block_size) {
  const int batch_idx = blockIdx.y;
  // Decode tokens have fixed length, compute the index directly
  const int token_idx = batch_idx * q_seq_len + (blockIdx.x >> 1);

  int* const current_indices = indices + token_idx * kTopk + threadIdx.x + (blockIdx.x & 1 ? MAX_THREADS_PER_BLOCK : 0);
  const int current_topk_token_idx = *current_indices;
  const int token_offset = current_topk_token_idx;

  if (current_topk_token_idx >= 0) {
    *current_indices = block_table[batch_idx * max_num_blocks_per_seq + token_offset / block_size] * block_size +
                       token_offset % block_size;
  }
}

void PagedSparseMlaConvertBlockTable(int* indices, int* block_table, int q_seq_len, int topk, int batch_size,
                                     int max_num_blocks_per_seq, int block_size, cudaStream_t stream) {
  constexpr int kTopk = 2048;
  assert(topk == kTopk);
  assert(kTopk == 2 * MAX_THREADS_PER_BLOCK);
  // Every two threads handle the topk convertion for one token
  dim3 grid_shape(q_seq_len << 1, batch_size);
  dim3 block_shape(MAX_THREADS_PER_BLOCK);
  PagedSparseMlaConvertBlockTableKernel<kTopk>
      <<<grid_shape, block_shape, 0, stream>>>(indices, block_table, q_seq_len, max_num_blocks_per_seq, block_size);
}

#ifdef ENABLE_FP8
__device__ __forceinline__ void IndexerCommonKCacheCopyKernel(const __nv_fp8_e4m3* __restrict__ k_src,
                                                              void** __restrict__ k_list,
                                                              const int* __restrict__ block_offsets, int block_size,
                                                              int k_stride_size, int token_idx, int batch_idx,
                                                              int input_offset) {
  __nv_fp8_e4m3* const k_dst_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(k_list[block_offsets[batch_idx] + input_offset / block_size]) +
      input_offset % block_size * k_stride_size;
  const __nv_fp8_e4m3* k_src_ptr = k_src + token_idx * k_stride_size;
  constexpr unsigned int kVecSize = sizeof(float) / sizeof(__nv_fp8_e4m3);
  vec_t<__nv_fp8_e4m3, kVecSize> k_src_vec;
  k_src_vec.load(k_src_ptr + threadIdx.x * kVecSize);
  k_src_vec.store(k_dst_ptr + threadIdx.x * kVecSize);
}

__device__ __forceinline__ void IndexerCommonVCacheCopyKernel(const float* __restrict__ v_src,
                                                              void** __restrict__ v_list,
                                                              const int* __restrict__ block_offsets, int block_size,
                                                              int v_stride_size, int token_idx, int batch_idx,
                                                              int input_offset, int k_thread_num) {
  float* const v_dst_ptr = reinterpret_cast<float*>(v_list[block_offsets[batch_idx] + input_offset / block_size]) +
                           input_offset % block_size * v_stride_size;
  const float* v_src_ptr = v_src + token_idx * v_stride_size;
  v_dst_ptr[threadIdx.x - k_thread_num] = v_src_ptr[threadIdx.x - k_thread_num];
}

__global__ void MlaIndexerFlashKVCacheCopyKernel(const __nv_fp8_e4m3* __restrict__ k_src,
                                                 const float* __restrict__ v_src, void** __restrict__ k_list,
                                                 void** __restrict__ v_list, const size_t* __restrict__ prefix_offsets,
                                                 const size_t* __restrict__ without_prefix_offsets,
                                                 const int* __restrict__ block_offsets, int block_size, int batch_size,
                                                 int k_stride_size, int v_stride_size, int k_thread_num) {
  const int token_idx = blockIdx.x;
  int batch_idx = 0;
  for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (token_idx < without_prefix_offsets[batch_idx + 1]) {
      break;
    }
  }

  // Token index in the whole input of current request
  const int input_offset =
      (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (token_idx - without_prefix_offsets[batch_idx]);

  threadIdx.x < k_thread_num ? IndexerCommonKCacheCopyKernel(k_src, k_list, block_offsets, block_size, k_stride_size,
                                                             token_idx, batch_idx, input_offset)
                             : IndexerCommonVCacheCopyKernel(v_src, v_list, block_offsets, block_size, v_stride_size,
                                                             token_idx, batch_idx, input_offset, k_thread_num);
}

void MlaIndexerFlashKVCacheCopy(__nv_fp8_e4m3* k_src, float* v_src, void** k_list, void** v_list,
                                size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                int block_size, int batch_size, int total_len, int k_stride_size, int v_stride_size,
                                cudaStream_t stream) {
  // num_heads == 1
  dim3 grid_shape(total_len);
  // For DeepSeek-V32: k_thread_num = 128 / 4 = 32 (1 warp), v_thread_num = 1
  // Each thread reads and writes one float
  const int k_thread_num = k_stride_size * sizeof(__nv_fp8_e4m3) / sizeof(float);
  const int v_thread_num = v_stride_size;
  assert(k_thread_num + v_thread_num <= MAX_THREADS_PER_BLOCK);
  const dim3 block_shape(k_thread_num + v_thread_num);
  MlaIndexerFlashKVCacheCopyKernel<<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, prefix_offsets, without_prefix_offsets, block_offsets, block_size, batch_size,
      k_stride_size, v_stride_size, k_thread_num);
}

__global__ void MlaIndexerPagedKVCacheCopyKernel(const __nv_fp8_e4m3* __restrict__ k_src,
                                                 const float* __restrict__ v_src, void** __restrict__ k_list,
                                                 void** __restrict__ v_list, const int* __restrict__ input_lengths,
                                                 const int* __restrict__ block_offsets, const int block_size,
                                                 const int k_stride_size, const int v_stride_size,
                                                 const int k_thread_num) {
  const int batch_idx = blockIdx.y;
  const int token_idx = batch_idx * gridDim.x + blockIdx.x;

  // Token index in the whole input of current request
  const int input_offset = input_lengths[batch_idx] - gridDim.x + blockIdx.x;

  threadIdx.x < k_thread_num ? IndexerCommonKCacheCopyKernel(k_src, k_list, block_offsets, block_size, k_stride_size,
                                                             token_idx, batch_idx, input_offset)
                             : IndexerCommonVCacheCopyKernel(v_src, v_list, block_offsets, block_size, v_stride_size,
                                                             token_idx, batch_idx, input_offset, k_thread_num);
}

void MlaIndexerPagedKVCacheCopy(__nv_fp8_e4m3* k_src, float* v_src, void** k_list, void** v_list, int* input_lengths,
                                int* block_offsets, int block_size, int batch_size, int req_q_len, int k_stride_size,
                                int v_stride_size, cudaStream_t stream) {
  // num_heads == 1
  const dim3 grid_shape(req_q_len, batch_size);
  assert(k_stride_size % (sizeof(float) / sizeof(__nv_fp8_e4m3)) == 0);
  // For DeepSeek-V32: k_thread_num = 128 / 4 = 32 (1 warp), v_thread_num = 1
  // Each thread reads and writes one float
  const int k_thread_num = k_stride_size * sizeof(__nv_fp8_e4m3) / sizeof(float);
  const int v_thread_num = v_stride_size;
  assert(k_thread_num + v_thread_num <= MAX_THREADS_PER_BLOCK);
  const dim3 block_shape(k_thread_num + v_thread_num);
  MlaIndexerPagedKVCacheCopyKernel<<<grid_shape, block_shape, 0, stream>>>(k_src, v_src, k_list, v_list, input_lengths,
                                                                           block_offsets, block_size, k_stride_size,
                                                                           v_stride_size, k_thread_num);
}
#endif

// Adapted from
// https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu#L405
// For the NoPE part, each tile of 128 elements is handled by half of one warp
// (16 threads). There are 4 total tiles, so 2 warps (64 threads).
// Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
// The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
// So in total, we use 3 warps (96 threads) per block.
template <typename SCALAR_T, int kVecSize, int kBlockStrideSize, int kEntryStrideSize, int kKStrideSize,
          int kVStrideSize>
__device__ void MlaCommonFp8DsMlaKVCacheCopyKernel(const SCALAR_T* k_src, const SCALAR_T* v_src,
                                                   uint8_t* const kv_list_base_ptr, const int token_idx) {
  // The last warp handles the RoPE part
  if (constexpr unsigned int kKThreadNum = kKStrideSize / kVecSize; threadIdx.x >= kKThreadNum) {
    // Cast kv_cache to 16_bit for RoPE values
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    SCALAR_T* const v_dst_ptr = reinterpret_cast<SCALAR_T*>(kv_list_base_ptr + kEntryStrideSize);
    const SCALAR_T* v_src_ptr = v_src + token_idx * kVStrideSize;
    // Each thread handles eight elements of RoPE
    vec_t<SCALAR_T, kVecSize> v_src_vec;
    const unsigned int head_offset = (threadIdx.x - kKThreadNum) * kVecSize;
    // Vectorized load of eight 16-bit values, performed as one 128-bit load
    v_src_vec.load(v_src_ptr + head_offset);
    // Vectorized store of eight 16-bit values, performed as one 128-bit store
    v_src_vec.store(v_dst_ptr + head_offset);
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = (warp_idx << 1) | (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const SCALAR_T* k_src_ptr = k_src + token_idx * kKStrideSize;
  // Vectorized load of eight 16-bit values, performed as an int4 load
  vec_t<SCALAR_T, kVecSize> k_src_vec;
  k_src_vec.load(k_src_ptr + threadIdx.x * kVecSize);

  // Max absolute value of this thread's elements
  float max_abs =
      fmaxf(fmaxf(fmaxf(fabsf(k_src_vec[0]), fabsf(k_src_vec[1])), fmaxf(fabsf(k_src_vec[2]), fabsf(k_src_vec[3]))),
            fmaxf(fmaxf(fabsf(k_src_vec[4]), fabsf(k_src_vec[5])), fmaxf(fabsf(k_src_vec[6]), fabsf(k_src_vec[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor_sync(uint32_t(-1), max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = max_abs / 448.f;
  tile_scale = fmaxf(tile_scale, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    reinterpret_cast<float*>(kv_list_base_ptr + kKStrideSize)[tile_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  uint8_t result[kVecSize];
#pragma unroll
  for (int i = 0; i < kVecSize; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, SCALAR_T, llm_kernels::utils::KVCacheType::kFp8E4M3>(k_src_vec[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(kv_list_base_ptr + threadIdx.x * kVecSize) = *reinterpret_cast<const uint64_t*>(result);
}

template <typename SCALAR_T, int kVecSize, int kBlockStrideSize, int kEntryStrideSize, int kKStrideSize,
          int kVStrideSize>
__global__ void MlaFlashFp8DsMlaKVCacheCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** kv_list,
                                                  size_t* prefix_offsets, size_t* without_prefix_offsets,
                                                  int* block_offsets, int block_size, int batch_size) {
  const int token_idx = blockIdx.x;
  int batch_idx = 0;
  for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (token_idx < without_prefix_offsets[batch_idx + 1]) {
      break;
    }
  }

  // Token index in the whole input of current request
  const int input_offset =
      (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (token_idx - without_prefix_offsets[batch_idx]);
  uint8_t* const kv_list_base_ptr =
      reinterpret_cast<uint8_t*>(kv_list[block_offsets[batch_idx] + input_offset / block_size]) +
      input_offset % block_size * kBlockStrideSize;

  MlaCommonFp8DsMlaKVCacheCopyKernel<SCALAR_T, kVecSize, kBlockStrideSize, kEntryStrideSize, kKStrideSize,
                                     kVStrideSize>(k_src, v_src, kv_list_base_ptr, token_idx);
}

// Copy MLA-kv compressed kv cache to kv blocks.
// This kernel will skip shared prefix blocks, and copy only the kv cache of tokens that need to be calculated.
//
// Args:
//   k_src:
//     The k need to be copied, not contain prefix part.
//   v_src:
//     The v need to be copied, not contain prefix part.
//   k_list:
//     A pointer array that contain k block's addr, include the prefix blocks.
//   v_list:
//     A pointer array that contain v block's addr, include the prefix blocks.
//   prefix_offsets:
//     The offset of prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_prefix_len, seq1_prefix_len + seq2_prefix_len, ...]
//   without_prefix_offsets:
//     The offset of non-prefix token num.
//     It's shape is [batch_size + 1], in format [0, seq1_unique_len, seq1_unique_len + seq2_unique_len, ...]
//   block_offsets:
//     The offset of block num of every seq, include prefix part.
//     It's shape is [batch_size + 1], in format [0, seq1_block_num, seq1_block_num + seq2_block_num]
//   block_size:
//     The token number of every cache block.
//   batch_size:
//     The batch size.
//   k_stride_size:
//     The stride of the k cache of one token.
//   v_stride_size:
//     The stride of the v cache of one token.
//   k_scale:
//     The quantization scale of k value.
//   v_scale:
//     The quantization scale of v value.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void MlaFlashKVCacheCopyKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                          size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                          int block_size, int batch_size, int k_stride_size, int v_stride_size,
                                          float k_scale, float v_scale) {
  const int token_idx = blockIdx.x;
  int batch_idx = 0;
  for (batch_idx = 0; batch_idx < batch_size; batch_idx++) {
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
void MlaFlashKVCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* prefix_offsets,
                         size_t* without_prefix_offsets, int* block_offsets, int block_size, int batch_size,
                         int total_len, int k_stride_size, int v_stride_size, float k_scale, float v_scale,
                         cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  dim3 grid_shape(total_len);
  assert(k_list == v_list);
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla) {
    assert(k_stride_size == 512 && v_stride_size == 64);
    constexpr int kVecSize = sizeof(float4) / sizeof(SCALAR_T);
    constexpr int kKStrideSize = 512;
    constexpr int kVStrideSize = 64;
    static_assert(kKStrideSize % kVecSize == 0 && kVStrideSize % kVecSize == 0);
    constexpr int kQuantBlockSize = 128;
    constexpr int kEntryStrideSize = kKStrideSize + kKStrideSize / kQuantBlockSize * sizeof(float);
    static_assert(kEntryStrideSize == 528);
    constexpr int kBlockStrideSize = kEntryStrideSize + kVStrideSize * sizeof(SCALAR_T);
    assert(kBlockStrideSize == 656);
    dim3 block_shape((kKStrideSize + kVStrideSize) / kVecSize);
    MlaFlashFp8DsMlaKVCacheCopyKernel<SCALAR_T, kVecSize, kBlockStrideSize, kEntryStrideSize, kKStrideSize,
                                      kVStrideSize><<<grid_shape, block_shape, 0, stream>>>(
        k_src, v_src, k_list, prefix_offsets, without_prefix_offsets, block_offsets, block_size, batch_size);
  } else {
    // In this case, v denotes the compress_kv (larger part), and k denotes the k_pe (smaller part),
    // which is inversed
    assert(v_stride_size >= k_stride_size && v_stride_size <= MAX_THREADS_PER_BLOCK);
    dim3 block_shape(v_stride_size);
    MlaFlashKVCacheCopyKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
        k_src, v_src, k_list, v_list, prefix_offsets, without_prefix_offsets, block_offsets, block_size, batch_size,
        k_stride_size, v_stride_size, k_scale, v_scale);
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE, int kVecBytes>
__global__ void MlaFlashPrefixKVReverseCacheCopyKernel(
    SCALAR_T* __restrict__ k_dst, SCALAR_T* __restrict__ v_dst, void** __restrict__ kv_list,
    const size_t* __restrict__ prefix_offsets, const size_t* __restrict__ seq_len_offset,
    const int* __restrict__ block_offsets, const int block_size, const int total_len, const int k_stride_size,
    const int v_stride_size, const float k_scale, const float v_scale, const int v_thread_num) {
  constexpr unsigned int kVecSize = kVecBytes / sizeof(SCALAR_T);

  const size_t token_idx = blockIdx.z * gridDim.y + blockIdx.y;
  if (token_idx >= total_len) {
    return;
  }

  size_t batch_idx = 0;
  while (token_idx >= seq_len_offset[batch_idx + 1]) {
    ++batch_idx;
  }

  // Calculate the index of the first forwarding new token (without prefix part) of the current batch.
  size_t prefix_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + seq_len_offset[batch_idx];
  if (token_idx >= prefix_limit) {
    return;
  }

  const size_t token_offset_in_req = token_idx - seq_len_offset[batch_idx];
  const size_t block_offset_in_req = token_offset_in_req / block_size;
  const size_t token_offset_in_block = token_offset_in_req % block_size;

  const size_t cache_offset = block_offsets[batch_idx] + block_offset_in_req;
  const size_t cache_head_offset = token_offset_in_block * (k_stride_size + v_stride_size) + threadIdx.x * kVecSize;
  CACHE_T* cache_ptr = reinterpret_cast<CACHE_T*>(kv_list[cache_offset]) + cache_head_offset;

  const bool is_latent = threadIdx.x < v_thread_num;
  const size_t dst_stride_size = is_latent ? v_stride_size : k_stride_size;
  const size_t dst_head_offset = is_latent ? threadIdx.x : threadIdx.x - v_thread_num;
  const float scale = is_latent ? v_scale : k_scale;
  SCALAR_T* dst_base = is_latent ? v_dst : k_dst;
  SCALAR_T* dst_ptr = dst_base + token_idx * dst_stride_size;

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
    vec_t<SCALAR_T, kVecSize> cache_vec;
    cache_vec.load(cache_ptr);
    cache_vec.store(dst_ptr + dst_head_offset * kVecSize);
  } else {
#pragma unroll
    for (unsigned int i = 0; i < kVecSize; ++i) {
      dst_ptr[dst_head_offset * kVecSize + i] = fp8::scaled_convert<SCALAR_T, CACHE_T, KV_DTYPE>(cache_ptr[i], scale);
    }
  }
}

// Prefix KV CacheCopy, supports copy-only and copy with quantization, CACHE_T to SCALAR_T.
// Only fp8 kv cache uses this now because KVCacheType::kAuto copies all cache, not only prefix part.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashPrefixKVReverseCacheCopy(SCALAR_T* k_dst, SCALAR_T* v_dst, void** kv_list, size_t* prefix_offsets,
                                      size_t* seq_len_offset, int* block_offsets, int block_size, int total_len,
                                      int k_stride_size, int v_stride_size, float k_scale, float v_scale,
                                      cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  const int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  const int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  const dim3 grid(1, grid_y, grid_z);

  constexpr int kVecBytes = 16;
  const int v_thread_num = v_stride_size * sizeof(SCALAR_T) / kVecBytes;
  const int stride_size = k_stride_size + v_stride_size;
  dim3 block(stride_size * sizeof(SCALAR_T) / kVecBytes);
  assert(block.x <= MAX_THREADS_PER_BLOCK && stride_size * sizeof(SCALAR_T) % kVecBytes == 0);
  MlaFlashPrefixKVReverseCacheCopyKernel<SCALAR_T, CACHE_T, KV_DTYPE, kVecBytes>
      <<<grid, block, 0, stream>>>(k_dst, v_dst, kv_list, prefix_offsets, seq_len_offset, block_offsets, block_size,
                                   total_len, k_stride_size, v_stride_size, k_scale, v_scale, v_thread_num);
}

template <typename SCALAR_T, int kVecBytes>
__global__ void MlaFlashWithoutPrefixKVCopyKernel(SCALAR_T* __restrict__ k_dst, SCALAR_T* __restrict__ v_dst,
                                                  const SCALAR_T* __restrict__ k_new,
                                                  const SCALAR_T* __restrict__ v_new,
                                                  const size_t* __restrict__ prefix_offsets,
                                                  const size_t* __restrict__ without_prefix_offsets,
                                                  const int total_len, const int k_stride_size, const int v_stride_size,
                                                  const int v_thread_num) {
  constexpr unsigned int kVecSize = kVecBytes / sizeof(SCALAR_T);

  int token_idx = blockIdx.z * gridDim.y + blockIdx.y;
  if (token_idx >= total_len) {
    return;
  }

  size_t batch_idx = 0;
  while (token_idx >= without_prefix_offsets[batch_idx + 1]) {
    ++batch_idx;
  }

  const size_t src_token_idx = prefix_offsets[batch_idx + 1] + token_idx;

  const bool is_latent = threadIdx.x < v_thread_num;
  const size_t stride_size = is_latent ? v_stride_size : k_stride_size;
  const size_t head_offset = is_latent ? threadIdx.x : threadIdx.x - v_thread_num;
  const SCALAR_T* new_base = is_latent ? v_new : k_new;
  const SCALAR_T* new_ptr = new_base + token_idx * stride_size;
  SCALAR_T* dst_base = is_latent ? v_dst : k_dst;
  SCALAR_T* dst_ptr = dst_base + /* skip prefix */ src_token_idx * stride_size;

  vec_t<SCALAR_T, kVecSize> new_vec;
  new_vec.load(new_ptr + head_offset * kVecSize);
  new_vec.store(dst_ptr + head_offset * kVecSize);
}

// Copy KV that did not hit prefix cache, SCALAR_T to SCALAR_T.
// Only fp8 kv cache uses this now because KVCacheType::kAuto copies all cache, including prefix part.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashWithoutPrefixKVCopy(SCALAR_T* k_dst, SCALAR_T* v_dst, SCALAR_T* k_new, SCALAR_T* v_new,
                                 size_t* prefix_offsets, size_t* without_prefix_offsets, int total_q_len,
                                 int k_stride_size, int v_stride_size, cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  const int grid_y = std::min(total_q_len, MAX_BLOCKS_PER_GRID_Y);
  const int grid_z = (total_q_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  const dim3 grid(1, grid_y, grid_z);

  constexpr int kVecBytes = 16;
  const int v_thread_num = v_stride_size * sizeof(SCALAR_T) / kVecBytes;
  const int stride_size = k_stride_size + v_stride_size;
  dim3 block(stride_size * sizeof(SCALAR_T) / kVecBytes);
  assert(block.x <= MAX_THREADS_PER_BLOCK && stride_size * sizeof(SCALAR_T) % kVecBytes == 0);
  MlaFlashWithoutPrefixKVCopyKernel<SCALAR_T, kVecBytes>
      <<<grid, block, 0, stream>>>(k_dst, v_dst, k_new, v_new, prefix_offsets, without_prefix_offsets, total_q_len,
                                   k_stride_size, v_stride_size, v_thread_num);
}

template <typename SCALAR_T, int kVecSize, int kBlockStrideSize, int kEntryStrideSize, int kKStrideSize,
          int kVStrideSize>
__global__ void MlaPagedFp8DsMlaKVCacheCopyKernel(const SCALAR_T* __restrict__ kv_c_src,
                                                  const SCALAR_T* __restrict__ k_pe_src, void** __restrict__ kv_list,
                                                  const int* __restrict__ input_lengths,
                                                  const int* __restrict__ block_offsets, const int block_size) {
  const int batch_idx = blockIdx.y;
  const int token_idx = batch_idx * gridDim.x + blockIdx.x;

  // Token index in the whole input of current request
  const int input_offset = input_lengths[batch_idx] - gridDim.x + blockIdx.x;
  uint8_t* const kv_list_base_ptr =
      reinterpret_cast<uint8_t*>(kv_list[block_offsets[batch_idx] + input_offset / block_size]) +
      input_offset % block_size * kBlockStrideSize;

  MlaCommonFp8DsMlaKVCacheCopyKernel<SCALAR_T, kVecSize, kBlockStrideSize, kEntryStrideSize, kKStrideSize,
                                     kVStrideSize>(kv_c_src, k_pe_src, kv_list_base_ptr, token_idx);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void MlaPagedKVCacheCopyKernel(const SCALAR_T* __restrict__ kv_c_src, const SCALAR_T* __restrict__ k_pe_src,
                                          void** __restrict__ kv_list, const int* __restrict__ input_lengths,
                                          const int* __restrict__ block_offsets, const int block_size,
                                          const int kv_c_stride_size, const int k_pe_stride_size, const float kv_scale,
                                          const int kv_c_thread_num) {
  constexpr unsigned int VEC_SIZE = sizeof(float4) / sizeof(SCALAR_T);

  const unsigned int batch_idx = blockIdx.y;
  const unsigned int token_idx = batch_idx * gridDim.x + blockIdx.x;

  const unsigned int input_offset = input_lengths[batch_idx] - gridDim.x + blockIdx.x;
  const unsigned int block_idx = input_offset / block_size;
  const unsigned int block_offset = input_offset % block_size;
  const unsigned int kv_list_offset = block_offsets[batch_idx] + block_idx;
  CACHE_T* const dst =
      reinterpret_cast<CACHE_T*>(kv_list[kv_list_offset]) + block_offset * (kv_c_stride_size + k_pe_stride_size);

  const bool is_latent = threadIdx.x < kv_c_thread_num;
  const int offset = is_latent ? 0 : kv_c_stride_size;
  const SCALAR_T* src = is_latent ? kv_c_src : k_pe_src;
  const int stride_size = is_latent ? kv_c_stride_size : k_pe_stride_size;
  const unsigned int head_offset = is_latent ? threadIdx.x : threadIdx.x - kv_c_thread_num;

  CACHE_T* const kv_dst_ptr = dst + offset;
  const SCALAR_T* k_src_ptr = src + token_idx * stride_size;

  vec_t<SCALAR_T, VEC_SIZE> k_src_vec;
  k_src_vec.load(k_src_ptr + head_offset * VEC_SIZE);

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
    k_src_vec.store(kv_dst_ptr + head_offset * VEC_SIZE);
  } else {
#pragma unroll
    for (unsigned int i = 0; i < VEC_SIZE; i++) {
      kv_dst_ptr[head_offset * VEC_SIZE + i] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_vec[i], kv_scale);
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaPagedKVCacheCopy(SCALAR_T* kv_c_src, SCALAR_T* k_pe_src, void** kv_list, int* input_lengths, int* block_offsets,
                         int block_size, int batch_size, int req_q_len, int kv_c_stride_size, int k_pe_stride_size,
                         float kv_scale, cudaStream_t stream) {
  // num_heads == 1
  const dim3 grid(req_q_len, batch_size);
  constexpr unsigned int kVecSize = sizeof(float4) / sizeof(SCALAR_T);
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8DsMla) {
    assert(kv_c_stride_size == 512 && k_pe_stride_size == 64);
    constexpr int kKvCStrideSize = 512;
    constexpr int kKPeStrideSize = 64;
    static_assert(kKvCStrideSize % kVecSize == 0 && kKPeStrideSize % kVecSize == 0);
    constexpr int kQuantBlockSize = 128;
    constexpr int kEntryStrideSize = kKvCStrideSize + kKvCStrideSize / kQuantBlockSize * sizeof(float);
    static_assert(kEntryStrideSize == 528);
    constexpr int kBlockStrideSize = kEntryStrideSize + kKPeStrideSize * sizeof(SCALAR_T);
    assert(kBlockStrideSize == 656);
    dim3 block((kKvCStrideSize + kKPeStrideSize) / kVecSize);
    MlaPagedFp8DsMlaKVCacheCopyKernel<SCALAR_T, kVecSize, kBlockStrideSize, kEntryStrideSize, kKvCStrideSize,
                                      kKPeStrideSize>
        <<<grid, block, 0, stream>>>(kv_c_src, k_pe_src, kv_list, input_lengths, block_offsets, block_size);
  } else {
    assert(kv_c_stride_size % kVecSize == 0);
    // For DeepSeek-V3: kv_c_thread_num = 512 / 8 = 64
    const int kv_c_thread_num = kv_c_stride_size / kVecSize;
    assert(k_pe_stride_size % kVecSize == 0);
    // For DeepSeek-V3: k_pe_thread_num = 64 / 8 = 8
    const int k_pe_thread_num = k_pe_stride_size / kVecSize;
    assert(kv_c_thread_num + k_pe_thread_num <= MAX_THREADS_PER_BLOCK);
    const dim3 block(kv_c_thread_num + k_pe_thread_num);
    MlaPagedKVCacheCopyKernel<SCALAR_T, CACHE_T, KV_DTYPE>
        <<<grid, block, 0, stream>>>(kv_c_src, k_pe_src, kv_list, input_lengths, block_offsets, block_size,
                                     kv_c_stride_size, k_pe_stride_size, kv_scale, kv_c_thread_num);
  }
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

  const char* const block_base = static_cast<const char*>(block_list[block_offsets[batch_idx] + block_offset_in_req]);

  // data in cache is: [latent, k_rope, latent, k_rope...]
  const size_t src_offset_bytes =
      token_offset_in_block * (k_rope_size + latent_size) * sizeof(CACHE_T) + copy_offset_bytes;

  const bool is_latent = copy_offset_bytes < latent_size * sizeof(CACHE_T);
  const size_t current_item_size = is_latent ? latent_size : k_rope_size;
  const size_t dst_item_offset = copy_offset_bytes - (is_latent ? 0 : latent_size * sizeof(CACHE_T));
  const size_t dst_offset_bytes = token_idx * current_item_size * sizeof(CACHE_T) + dst_item_offset;
  char* const dst = static_cast<char*>(is_latent ? latent_out : k_rope_out);

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

// Copy compressed_kv and k_rope from src_flexible_kv_cache_block to dst_flexible_kv_cache_block.
// Handles flexible cached tokens. Note that the copied k_rope has wrong ROPE information.
//
// Args:
//   kv_src:
//     A pointer array of src kv cache block (which contains the src token) ptrs, has no offset in layer_idx dim.
//   kv_dst:
//     A pointer array of dst kv cache block (which contains the dst token) ptrs, has no offset in layer_idx dim.
//   kv_list_src:
//     An int array of flexible cached tokens offset_in_src_block.
//   kv_list_dst:
//     An int array of flexible cached tokens offset_in_dst_block.
//   block_size:
//     The token number of every cache block.
//   layer_idx:
//     The index of the layer, helps finding the start of current layer in kv_src and kv_dst.
//   total_len:
//     The number of flexible cached tokens.
//   stride_size:
//     The size of the k&v cache of one token, equals qk_rope_head_dim + kv_lora_rank.
template <typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE, int kVecBytes>
__global__ void MlaFlexibleTokenCacheCopyKernel(CACHE_T** kv_src, CACHE_T** kv_dst, int* kv_list_src, int* kv_list_dst,
                                                int block_size, int layer_idx, int total_len, int stride_size) {
  constexpr unsigned int kVecSize = kVecBytes / sizeof(CACHE_T);

  int input_idx = blockIdx.z * gridDim.y + blockIdx.y;
  if (input_idx >= total_len) {
    return;
  }

  size_t layer_size = block_size * stride_size;
  CACHE_T* src_base = kv_src[input_idx] + layer_size * layer_idx;
  CACHE_T* dst_base = kv_dst[input_idx] + layer_size * layer_idx;
  int src_idx = kv_list_src[input_idx] % block_size;
  int dst_idx = kv_list_dst[input_idx] % block_size;
  CACHE_T* src_ptr = src_base + src_idx * stride_size;
  CACHE_T* dst_ptr = dst_base + dst_idx * stride_size;
  size_t offset = threadIdx.x * kVecSize;

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
    vec_t<CACHE_T, kVecSize> src_vec;
    src_vec.load(src_ptr + offset);
    src_vec.store(dst_ptr + offset);
  } else {
#pragma unroll
    for (unsigned int i = 0; i < kVecSize; ++i) {
      dst_ptr[offset + i] = src_ptr[offset + i];
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlexibleTokenCacheCopy(CACHE_T** kv_src, CACHE_T** kv_dst, int* kv_list_src, int* kv_list_dst, int block_size,
                               int layer_idx, int total_len, int stride_size, cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  const int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  const int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  const dim3 grid(1, grid_y, grid_z);

  constexpr int kVecBytes = 16;
  dim3 block(stride_size * sizeof(CACHE_T) / kVecBytes);
  assert(block.x <= MAX_THREADS_PER_BLOCK && stride_size * sizeof(CACHE_T) % kVecBytes == 0);

  MlaFlexibleTokenCacheCopyKernel<CACHE_T, KV_DTYPE, kVecBytes><<<grid, block, 0, stream>>>(
      kv_src, kv_dst, kv_list_src, kv_list_dst, block_size, layer_idx, total_len, stride_size);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void MlaFlashFlexibleKCacheCopyKernel(SCALAR_T* k_src, void** k_list, size_t* flexible_offsets,
                                                 size_t* prefix_offsets, size_t* seq_len_offset, int* block_offsets,
                                                 int block_size, int bs, int total_len, int k_stride_size,
                                                 int v_stride_size, float k_scale) {
  // Each batch in the sequence structures as follow:
  // | prefix_cached_tokens | flexible_cached_tokens | forwarding_new_tokens |
  // |       skipped        |        copying         |        skipped        |

  const size_t token_idx = blockIdx.z * gridDim.y + blockIdx.y;
  if (token_idx >= total_len) {
    return;
  }

  size_t batch_idx = 0;
  while (token_idx >= seq_len_offset[batch_idx + 1]) {
    ++batch_idx;
  }

  // Calculate the index of the first forwarding new token (without prefix part) of the current batch.
  size_t flexible_token_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + seq_len_offset[batch_idx];
  if (token_idx >= flexible_token_limit) {
    return;
  }

  // Calculate the index of the first flexible cached token of the current batch.
  size_t prefix_token_limit = flexible_offsets[batch_idx + 1] - flexible_offsets[batch_idx] + seq_len_offset[batch_idx];
  if (token_idx < prefix_token_limit) {
    return;
  }

  const size_t token_offset_in_req = token_idx - seq_len_offset[batch_idx];
  const size_t block_offset_in_req = token_offset_in_req / block_size;
  const size_t token_offset_in_block = token_offset_in_req % block_size;

  const size_t k_cache_offset = block_offsets[batch_idx] + block_offset_in_req;
  const size_t k_cache_head_offset = token_offset_in_block * (k_stride_size + v_stride_size) + v_stride_size;
  CACHE_T* k_cache_ptr = reinterpret_cast<CACHE_T*>(k_list[k_cache_offset]) + k_cache_head_offset;

  SCALAR_T* k_src_ptr = k_src + token_idx * k_stride_size;

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
    k_cache_ptr[threadIdx.x] = k_src_ptr[threadIdx.x];
  } else {
    k_cache_ptr[threadIdx.x] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[threadIdx.x], k_scale);
  }
}

// Copy k_rope with correct ROPE information from contigous buffer to cache block.
// Only handles flexible cached tokens, ignore prefix cached tokens and forwarding new tokens.
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashFlexibleKCacheCopy(SCALAR_T* k_src, void** k_list, size_t* flexible_offsets, size_t* prefix_offsets,
                                size_t* seq_len_offset, int* block_offsets, int block_size, int bs, int total_len,
                                int k_stride_size, int v_stride_size, float k_scale, cudaStream_t stream) {
  // There is only one head for MLA-kvcache.
  const int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  const int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  const dim3 grid(1, grid_y, grid_z);

  assert(k_stride_size <= MAX_THREADS_PER_BLOCK);
  const dim3 block(k_stride_size);
  MlaFlashFlexibleKCacheCopyKernel<SCALAR_T, CACHE_T, KV_DTYPE>
      <<<grid, block, 0, stream>>>(k_src, k_list, flexible_offsets, prefix_offsets, seq_len_offset, block_offsets,
                                   block_size, bs, total_len, k_stride_size, v_stride_size, k_scale);
}

#define MLA_CACHE_COPY_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                               \
  template void MlaFlashKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(SCALAR_T*, SCALAR_T*, void**, void**, size_t*,        \
                                                                 size_t*, int*, int, int, int, int, int, float, float, \
                                                                 cudaStream_t);                                        \
  template void MlaPagedKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(SCALAR_T*, SCALAR_T*, void**, int*, int*, int, int,   \
                                                                 int, int, int, float, cudaStream_t);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_CACHE_COPY_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_CACHE_COPY_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_CACHE_COPY_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8DsMla);
#undef MLA_CACHE_COPY_FUNCTION_DECLARATION

#define MLA_CACHE_COPY_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                               \
  template void MlaFlashPrefixKVReverseCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(                                         \
      SCALAR_T * k_dst, SCALAR_T * v_dst, void** kv_list, size_t* prefix_offsets, size_t* seq_len_offset,              \
      int* block_offsets, int block_size, int total_len, int k_stride_size, int v_stride_size, float k_scale,          \
      float v_scale, cudaStream_t stream);                                                                             \
  template void MlaFlashWithoutPrefixKVCopy<SCALAR_T, CACHE_T, KV_DTYPE>(                                              \
      SCALAR_T * k_dst, SCALAR_T * v_dst, SCALAR_T * k_new, SCALAR_T * v_new, size_t * prefix_offsets,                 \
      size_t * without_prefix_offsets, int total_q_len, int k_stride_size, int v_stride_size, cudaStream_t stream);    \
  template void MlaGetFromCompressedCache<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      void* const k_rope_out, void* const latent_out, const void* const* const block_list, const int total_len,        \
      const size_t* const seq_len_offset, const int* const block_offsets, const int block_size, const int k_rope_size, \
      const int latent_size, cudaStream_t stream);                                                                     \
  template void MlaFlexibleTokenCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      CACHE_T * *kv_src, CACHE_T * *kv_dst, int* kv_list_src, int* kv_list_dst, int block_size, int layer_idx,         \
      int total_len, int stride_size, cudaStream_t stream);                                                            \
  template void MlaFlashFlexibleKCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(                                               \
      SCALAR_T * k_src, void** k_list, size_t* flexible_offsets, size_t* prefix_offsets, size_t* seq_len_offset,       \
      int* block_offsets, int block_size, int bs, int total_len, int k_stride_size, int v_stride_size, float k_scale,  \
      cudaStream_t stream);
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

__global__ void fill_kv_scale_into_buffer_kernel(float* k_scale_buffer, float* v_scale_buffer) {
  k_scale_buffer[threadIdx.x] = k_scale_buffer[0];
  v_scale_buffer[threadIdx.x] = v_scale_buffer[0];
}

void InvokeFillKVScaleIntoBuffer(void* k_scale_ptr, void* v_scale_ptr, float* k_scale_host, float* v_scale_host,
                                 int kv_head_num, const cudaStream_t& stream) {
  cudaMemcpyAsync(k_scale_ptr, reinterpret_cast<void*>(k_scale_host), sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(v_scale_ptr, reinterpret_cast<void*>(v_scale_host), sizeof(float), cudaMemcpyHostToDevice, stream);
  assert(kv_head_num <= MAX_THREADS_PER_BLOCK);
  fill_kv_scale_into_buffer_kernel<<<1, kv_head_num, 0, stream>>>(reinterpret_cast<float*>(k_scale_ptr),
                                                                  reinterpret_cast<float*>(v_scale_ptr));
}

}  // namespace nvidia
}  // namespace llm_kernels
