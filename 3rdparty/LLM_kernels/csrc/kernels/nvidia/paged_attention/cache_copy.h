#pragma once

#include <cuda_runtime.h>
#include "csrc/utils/quant_type.h"
namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
               size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
               int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CachePosCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, int* input_lengths,
                  int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size,
                  int stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ReverseCacheCopy(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                      size_t* prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,
                      int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void FlexibleReverseCacheCopy(CACHE_T** kv_src, CACHE_T** kv_dst, int* kv_list_src, int* kv_list_dst, int block_size,
                              int layer_idx, int total_len, int num_heads, int head_size, int stride_size,
                              cudaStream_t stream);

/**
 * @brief Converts data to FP8 format and then back to its original format.
 *
 * This function is designed for testing the effects of computations in lower precision by
 * first converting the input data to FP8 format, performing operations, and then converting it back.
 *
 * @param data Pointer to the data array that will be converted.
 * @param dim0 The size of the first dimension of the data array.
 * @param dim1 The size of the second dimension of the data array.
 * @param stride_size The stride size, indicating the separation between consecutive elements.
 * @param k_scale The scaling parameter of K cache FP8 quantization.
 * @param v_scale The scaling parameter of V cache FP8 quantization.
 * @param scale The scaling parameter of K/V cache FP8 quantization. Should be used in ConvertFP8AndBack() only.
 * @param stream The CUDA stream for asynchronous execution.
 * @tparam SCALAR_T The data type of the input data (e.g., __nv_bfloat16, half).
 * @tparam CACHE_T The data type used for intermediate storage during conversion.
 * @tparam KV_DTYPE An enum class parameter that determines the FP8 format to use.
 *                  0 for kAuto, 1 for kFp8E4M3, 2 for kFp8E5M2.
 */
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ConvertFP8AndBack(SCALAR_T* data, size_t dim0, size_t dim1, int stride_size, float scale, cudaStream_t stream);

void ProcessKvList(void** kv_list, size_t layer_num, size_t block_num, size_t block_size, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
