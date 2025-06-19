/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cuda_runtime.h>

#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaKVCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* prefix_offsets,
                    size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len,
                    int k_stride_size, int v_stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaExtendKVPrefixWithEmpty(SCALAR_T* k_src, SCALAR_T* v_src, SCALAR_T* k_out, SCALAR_T* v_out,
                                size_t* prefix_offsets, size_t* without_prefix_offsets, int num_heads, int head_size,
                                int total_len_without_prefix, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaCopyKeyBlockWithReplication(SCALAR_T* k_out, void** k_list, int src_stride_size, int src_copy_offset,
                                    int src_copy_len, int dst_num_heads, int dst_head_size, int dst_copy_offset,
                                    size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                                    int block_size, int total_prefix_len, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaCopyValueBlockToBuffer(SCALAR_T* v_out, void** v_list, int src_stride_size, int src_copy_offset,
                               int src_copy_len, int dst_stride_size, size_t* prefix_offsets, int* block_offsets,
                               int block_size, int total_prefix_len, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaFillKVPrefix(SCALAR_T* k_out, SCALAR_T* v_out, SCALAR_T* k_src, SCALAR_T* v_src, size_t* prefix_offsets,
                     size_t* without_prefix_offsets, int num_heads, int src_head_size, int dst_head_size,
                     int total_prefix_len, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaDrainAttnOutPrefix(SCALAR_T* out, SCALAR_T* input, size_t* prefix_offsets, size_t* without_prefix_offsets,
                           int num_heads, int head_size, int batch_size, int total_len_without_prefix,
                           cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
