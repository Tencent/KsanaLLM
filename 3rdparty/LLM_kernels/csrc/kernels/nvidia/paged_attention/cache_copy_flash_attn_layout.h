#pragma once

#include <cuda_runtime.h>
#include "csrc/utils/quant_type.h"
namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                              size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                              int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                              float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CachePosCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, int* input_lengths,
                                 int* block_offsets, int block_size, int bs, int req_q_len, int num_heads,
                                 int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ConvertToScalar(CACHE_T* src, SCALAR_T* dst, int* src_table, int* dst_table, int table_len, int data_num,
                     float k_scale, float v_scale, cudaStream_t stream);
}  // namespace nvidia
}  // namespace llm_kernels
