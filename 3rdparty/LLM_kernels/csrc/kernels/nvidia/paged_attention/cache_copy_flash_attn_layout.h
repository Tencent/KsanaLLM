#pragma once

#include <cuda_runtime.h>
#include "csrc/utils/quant_type.h"
namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                              size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                              int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                              float k_scale, float v_scale, cudaStream_t stream, bool is_kv_with_prefix);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CachePosCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, int* input_lengths,
                                 int* block_offsets, int block_size, int bs, int req_q_len, int num_heads,
                                 int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ReverseCacheCopyFlashAttnLayout(SCALAR_T* k_dst, SCALAR_T* v_dst, void** k_list, void** v_list,
                                     size_t* input_offsets, size_t* prefix_offsets, int* block_offsets, int block_size,
                                     int bs, int total_len, int num_heads, int head_size, int stride_size,
                                     float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ConvertToCacheType(SCALAR_T* qkv_src, CACHE_T* qkv_dst, int total_len, int num_heads, int head_size,
                        int stride_size, float qkv_scale, cudaStream_t stream);

template <typename CACHE_T>
void FP8WithPrefixReverseCacheCopyFlashAttnLayout(CACHE_T* k_src, CACHE_T* v_src, void** k_list, void** v_list,
                                                  size_t* input_offsets, int* block_offsets, int block_size, int bs,
                                                  int total_len, int num_heads, int head_size, int stride_size,
                                                  size_t size_of_scalar_t, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
