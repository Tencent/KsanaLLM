/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#ifdef ENABLE_FP8
#  include <cuda_fp8.h>
#endif
#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

#ifdef ENABLE_FP8
void MlaIndexerFlashKVCacheCopy(__nv_fp8_e4m3* k_src, float* v_src, void** k_list, void** v_list,
                                size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                int block_size, int batch_size, int total_len, int k_stride_size, int v_stride_size,
                                cudaStream_t stream);

void MlaIndexerPagedKVCacheCopy(__nv_fp8_e4m3* k_src, float* v_src, void** k_list, void** v_list, int* input_lengths,
                                int* block_offsets, int block_size, int batch_size, int req_q_len, int k_stride_size,
                                int v_stride_size, cudaStream_t stream);
#endif

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashKVCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* prefix_offsets,
                         size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len,
                         int k_stride_size, int v_stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashPrefixKVReverseCacheCopy(SCALAR_T* k_dst, SCALAR_T* v_dst, void** kv_list, size_t* prefix_offsets,
                                      size_t* seq_len_offset, int* block_offsets, int block_size, int total_len,
                                      int k_stride_size, int v_stride_size, float k_scale, float v_scale,
                                      cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashWithoutPrefixKVCopy(SCALAR_T* k_dst, SCALAR_T* v_dst, SCALAR_T* k_new, SCALAR_T* v_new,
                                 size_t* prefix_offsets, size_t* without_prefix_offsets, int total_q_len,
                                 int k_stride_size, int v_stride_size, cudaStream_t stream);

/**
 * @brief Copy `kv_c` (compress-kv) and `k_pe` (key-rope) into `kv_list` (the joint key-value cache list of MLA) for MQA
 * computations
 *
 * @param kv_c_src         Pointer to the compress-kv tensor
 * @param k_pe_src         Pointer to the key-rope tensor
 * @param kv_list          Pointers to the joint key-value cache list
 * @param input_lengths    Array of input sequence lengths for each decode request
 * @param block_offsets    Array of accumulated block offsets for each decode request
 * @param block_size       Number of tokens in each block
 * @param bs               Batch size, i.e., the number of decode requests
 * @param req_q_len        Length of each decode request
 * @param kv_c_dim         Size of last dimension of the compress-kv tensor
 * @param k_pe_dim         Size of last dimension of the key-rope tensor
 * @param kv_c_stride_size Last stride of the compress-kv tensr
 * @param k_pe_stride_size Last stride of the key-rotary tensor
 * @param kv_scale         Scaling factor applied to the copied values
 * @param stream           CUDA stream on which the copy kernel is executed
 */
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaPagedKVCacheCopy(SCALAR_T* kv_c_src, SCALAR_T* k_pe_src, void** kv_list, int* input_lengths, int* block_offsets,
                         int block_size, int bs, int req_q_len, int kv_c_dim, int k_pe_dim, int kv_c_stride_size,
                         int k_pe_stride_size, float kv_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaGetFromCompressedCache(void* const k_rope_out, void* const latent_out, const void* const* const block_list,
                               const int total_len, const size_t* const seq_len_offset, const int* const block_offsets,
                               const int block_size, const int k_rope_size, const int latent_size, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlexibleTokenCacheCopy(CACHE_T** kv_src, CACHE_T** kv_dst, int* kv_list_src, int* kv_list_dst, int block_size,
                               int layer_idx, int total_len, int stride_size, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFlashFlexibleKCacheCopy(SCALAR_T* k_src, void** k_list, size_t* flexible_offsets, size_t* prefix_offsets,
                                size_t* seq_len_offset, int* block_offsets, int block_size, int bs, int total_len,
                                int k_stride_size, int v_stride_size, float k_scale, cudaStream_t stream);

// FA3 requires qkv scale input of [1, head_num] shape
void InvokeFillKVScaleIntoBuffer(void* k_scale_ptr, void* v_scale_ptr, float* k_scale_host, float* v_scale_host,
                                 int kv_head_num, const cudaStream_t& stream);

}  // namespace nvidia
}  // namespace llm_kernels
