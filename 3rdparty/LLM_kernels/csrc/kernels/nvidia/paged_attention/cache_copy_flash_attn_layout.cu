#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "cache_copy_flash_attn_layout.h"
#include "csrc/utils/nvidia/cuda_utils.h"
#include "quant_utils.cuh"

namespace llm_kernels {
namespace nvidia {

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS_PER_GRID_Y 65535

__device__ int k_chunk_size = 16;

/*
  block_size:     Number of tokens stored in each block.
  block_offsets:  Records the number of blocks for each batch size   [bs + 1,]
  prefix_offsets: Records the prefix length for each batch size (bs)
                  (accumulated from 0 to the current batch).         [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void CacheCopyFlashAttnLayoutKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                               size_t* input_offsets, size_t* prefix_offsets,
                                               size_t* without_prefix_offsets, int* block_offsets, int block_size,
                                               int bs, int total_len, int num_heads, int head_size, int stride_size,
                                               float k_scale, float v_scale, bool is_kv_with_prefix) {
  int idx = blockIdx.y + blockIdx.z * gridDim.y;
  int cur_block_offset = 0;
  int cur_batch_offset = 0;
  if (idx < total_len) {
    int batch_idx = 0;
    if (is_kv_with_prefix) {
      // copy from k,v(with_prefix) to cache list (input_offsets with prefix offsets)
      for (batch_idx = 0; batch_idx < bs; batch_idx++) {
        if (idx < input_offsets[batch_idx + 1]) {
          break;
        }
      }
      size_t prefix_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + input_offsets[batch_idx];
      if (idx < prefix_limit) {
        return;
      }
      cur_block_offset = (idx - input_offsets[batch_idx]) / block_size;
      cur_batch_offset = (idx - input_offsets[batch_idx]) % block_size;
    } else {
      // copy from k,v(without_prefix_offsets) to cache list (input_offsets with prefix offsets)
      for (batch_idx = 0; batch_idx < bs; batch_idx++) {
        if (idx < without_prefix_offsets[batch_idx + 1]) {
          break;
        }
      }
      size_t cur_batch_token_idx_with_prefix =
          (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (idx - without_prefix_offsets[batch_idx]);
      cur_block_offset = cur_batch_token_idx_with_prefix / block_size;
      cur_batch_offset = cur_batch_token_idx_with_prefix % block_size;
    }
    CACHE_T* k_dst_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[batch_idx] + cur_block_offset]);
    CACHE_T* v_dst_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + cur_block_offset]);
    SCALAR_T* k_src_ptr = k_src + idx * stride_size;
    SCALAR_T* v_src_ptr = v_src + idx * stride_size;
    for (int head_size_i = threadIdx.x; head_size_i < head_size; head_size_i += blockDim.x) {
      for (int num_head_i = blockIdx.x; num_head_i < num_heads; num_head_i += gridDim.x) {
        int k_src_index = num_head_i * head_size + head_size_i;
        int k_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int v_src_index = num_head_i * head_size + head_size_i;
        int v_dst_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        // Assignment operation
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          k_dst_base[k_dst_index] = k_src_ptr[k_src_index];
          v_dst_base[v_dst_index] = v_src_ptr[v_src_index];
        } else {
          k_dst_base[k_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[k_src_index], k_scale);
          v_dst_base[v_dst_index] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(v_src_ptr[v_src_index], v_scale);
        }
      }
    }
  }
}

/*
  block_size:    Number of tokens stored in each block.
  block_offsets: Records the number of blocks for each batch size   [bs + 1,]
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void CachePosCopyFlashAttnLayoutKernel(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list,
                                                  int* input_lengths, int* block_offsets, int block_size,
                                                  int stride_size, float k_scale, float v_scale) {
  const unsigned int batch_idx = blockIdx.z;
  const unsigned int token_idx = batch_idx * gridDim.y + blockIdx.y;
  const unsigned int num_heads = gridDim.x;
  const unsigned int head_size = blockDim.x;

  const unsigned int input_offset = input_lengths[batch_idx] - gridDim.y + blockIdx.y;
  const unsigned int kv_list_offset = block_offsets[batch_idx] + input_offset / block_size;
  const unsigned int cur_batch_offset = input_offset % block_size * num_heads * head_size;
  CACHE_T* const k_dst_base = reinterpret_cast<CACHE_T*>(k_list[kv_list_offset]) + cur_batch_offset;
  CACHE_T* const v_dst_base = reinterpret_cast<CACHE_T*>(v_list[kv_list_offset]) + cur_batch_offset;
  SCALAR_T* const k_src_ptr = k_src + token_idx * stride_size;
  SCALAR_T* const v_src_ptr = v_src + token_idx * stride_size;

  const unsigned int head_offset = blockIdx.x * head_size + threadIdx.x;
  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
    k_dst_base[head_offset] = k_src_ptr[head_offset];
    v_dst_base[head_offset] = v_src_ptr[head_offset];
  } else {
    k_dst_base[head_offset] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(k_src_ptr[head_offset], k_scale);
    v_dst_base[head_offset] = fp8::scaled_convert<CACHE_T, SCALAR_T, KV_DTYPE>(v_src_ptr[head_offset], v_scale);
  }
}

/*
  Reverse cache copy for FlashAttn layout: copy from cache back to contiguous tensor with dequantization.

  k_dst/v_dst: Output contiguous tensors [total_len, num_heads, head_size]
  k_list/v_list: Input cache blocks (FlashAttn layout, possibly fp8)
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void ReverseCacheCopyFlashAttnLayoutKernel(SCALAR_T* k_dst, SCALAR_T* v_dst, void** k_list, void** v_list,
                                                      size_t* input_offsets, size_t* prefix_offsets, int* block_offsets,
                                                      int block_size, int bs, int total_len, int num_heads, int head_size,
                                                      int stride_size, float k_scale, float v_scale) {
  // Reverse copy from cache list (only prefix part) to k_dst, v_dst
  int idx = blockIdx.y + blockIdx.z * gridDim.y;
  if (idx < total_len) {
    int batch_idx = 0;
    for (batch_idx = 0; batch_idx < bs; batch_idx++) {
      if (idx < input_offsets[batch_idx + 1]) {
        break;
      }
    }

    size_t prefix_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + input_offsets[batch_idx];
    if (idx >= prefix_limit) {
      return;  // skip non-prefix tokens
    }

    int cur_block_offset = (idx - input_offsets[batch_idx]) / block_size;
    int cur_batch_offset = (idx - input_offsets[batch_idx]) % block_size;

    CACHE_T* k_src_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[batch_idx] + cur_block_offset]);
    CACHE_T* v_src_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + cur_block_offset]);
    SCALAR_T* k_dst_ptr = k_dst + idx * stride_size;
    SCALAR_T* v_dst_ptr = v_dst + idx * stride_size;

    for (int head_size_i = threadIdx.x; head_size_i < head_size; head_size_i += blockDim.x) {
      for (int num_head_i = blockIdx.x; num_head_i < num_heads; num_head_i += gridDim.x) {
        // FlashAttn layout in cache: [block_size, num_heads, head_size]
        int k_src_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int k_dst_index = num_head_i * head_size + head_size_i;
        int v_src_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int v_dst_index = num_head_i * head_size + head_size_i;

        // Reverse assignment with dequantization
        if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
          k_dst_ptr[k_dst_index] = k_src_base[k_src_index];
          v_dst_ptr[v_dst_index] = v_src_base[v_src_index];
        } else {
          k_dst_ptr[k_dst_index] = fp8::scaled_convert<SCALAR_T, CACHE_T, KV_DTYPE>(k_src_base[k_src_index], k_scale);
          v_dst_ptr[v_dst_index] = fp8::scaled_convert<SCALAR_T, CACHE_T, KV_DTYPE>(v_src_base[v_src_index], v_scale);
        }
      }
    }
  }
}

/*
  block_size:     Number of tokens stored in each block.
  input_offsets:  Records the number of tokens for each batch size   [bs + 1,]
  block_offsets:  Records the number of blocks for each batch size   [bs + 1,]
*/
template <typename CACHE_T>
__global__ void FP8WithPrefixReverseCacheCopyFlashAttnLayoutKernel(CACHE_T* k_dst, CACHE_T* v_dst, void** k_list, void** v_list,
                                                                   size_t* input_offsets, int* block_offsets, int block_size, int bs,
                                                                   int total_len, int num_heads, int head_size, int stride_size,
                                                                   size_t size_of_scalar_t) {

  int token_idx = blockIdx.y + blockIdx.z * gridDim.y;
  if (token_idx < total_len) {
    int batch_idx = 0;
    for (batch_idx = 0; batch_idx < bs; batch_idx++) {
      if (token_idx < input_offsets[batch_idx + 1]) {
        break;
      }
    }

    int cur_block_offset = (token_idx - input_offsets[batch_idx]) / block_size;
    int cur_batch_offset = (token_idx - input_offsets[batch_idx]) % block_size;
    CACHE_T* k_src_base = reinterpret_cast<CACHE_T*>(k_list[block_offsets[batch_idx] + cur_block_offset]);
    CACHE_T* v_src_base = reinterpret_cast<CACHE_T*>(v_list[block_offsets[batch_idx] + cur_block_offset]);
    CACHE_T* k_dst_ptr = k_dst + token_idx * stride_size;
    CACHE_T* v_dst_ptr = v_dst + token_idx * stride_size;

    for (int head_size_i = threadIdx.x; head_size_i < head_size; head_size_i += blockDim.x) {
      for (int num_head_i = blockIdx.x; num_head_i < num_heads; num_head_i += gridDim.x) {
        // FlashAttn layout in cache: [token_in_block, num_heads, head_size]
        int k_src_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int k_dst_index = num_head_i * head_size + head_size_i;
        int v_src_index = cur_batch_offset * num_heads * head_size + num_head_i * head_size + head_size_i;
        int v_dst_index = num_head_i * head_size + head_size_i;
        k_dst_ptr[k_dst_index] = k_src_base[k_src_index];
        v_dst_ptr[v_dst_index] = v_src_base[v_src_index];
      }
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE, typename VEC_IN_TYPE,
          typename VEC_OUT_TYPE>
__global__ void ConvertToCacheTypeKernel(const SCALAR_T* __restrict__ src_ptr, CACHE_T* __restrict__ dst_ptr, int total_len,
                                         int elems_num, int stride_size, float scale, int vec_size) {
  int batch_idx = blockIdx.z;
  int token_idx = batch_idx * gridDim.y + blockIdx.y;
  if (token_idx >= total_len) {
    return;
  }

  // 该 token 的起始位置
  const SCALAR_T* __restrict__ src = src_ptr + token_idx * stride_size;
  CACHE_T* __restrict__ dst = dst_ptr + token_idx * stride_size;

  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int elem_idx = vec_idx * vec_size;

  if (elem_idx < elems_num) {
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {  // only copy
      VEC_IN_TYPE vec_src = *reinterpret_cast<const VEC_IN_TYPE*>(&src[elem_idx]);
      *reinterpret_cast<VEC_IN_TYPE*>(&dst[elem_idx]) = vec_src;
    } else {
      VEC_IN_TYPE vec_src = *reinterpret_cast<const VEC_IN_TYPE*>(&src[elem_idx]);
      VEC_OUT_TYPE vec_dst = fp8::scaled_convert<VEC_OUT_TYPE, VEC_IN_TYPE, KV_DTYPE>(vec_src, scale);
      *reinterpret_cast<VEC_OUT_TYPE*>(&dst[elem_idx]) = vec_dst;
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                              size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                              int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                              float k_scale, float v_scale, cudaStream_t stream, bool is_kv_with_prefix) {
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(num_heads, grid_y, grid_z);

  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  CacheCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_offsets, prefix_offsets, without_prefix_offsets, block_offsets, block_size,
      bs, total_len, num_heads, head_size, stride_size, k_scale, v_scale, is_kv_with_prefix);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CachePosCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, int* input_lengths,
                                 int* block_offsets, int block_size, int bs, int req_q_len, int num_heads,
                                 int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream) {
  const dim3 grid_shape(num_heads, req_q_len, bs);
  const dim3 block_shape(head_size);
  CachePosCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_lengths, block_offsets, block_size, stride_size, k_scale, v_scale);
}

// input [total_len, num_heads, head_size]
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ConvertToCacheType(SCALAR_T* src, CACHE_T* dst, int total_len, int num_heads, int head_size,
                         int stride_size, float scale, cudaStream_t stream) {
  constexpr int kBlockThreads = 256;
  constexpr int kVecSize = 8;
  constexpr bool is_bfloat16 = std::is_same_v<SCALAR_T, __nv_bfloat16>;
  
  int elems_num = num_heads * head_size;
  int vec_size = kVecSize;
  
  // Only support bfloat16 vectorization
  // Check alignment: addresses must be aligned for vectorized access
  int src_align_size = kVecSize * sizeof(SCALAR_T);
  int dst_align_size = kVecSize * sizeof(CACHE_T);
  bool is_aligned = (reinterpret_cast<uintptr_t>(src) % src_align_size == 0) &&
                    (reinterpret_cast<uintptr_t>(dst) % dst_align_size == 0);
  if (!is_bfloat16 || elems_num % vec_size != 0 || !is_aligned) {
    vec_size = 1;
  }

  int vecs_num = ceil(static_cast<float>(elems_num) / vec_size);

  int grid_x = ceil(static_cast<float>(vecs_num) / kBlockThreads);
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid(grid_x, grid_y, grid_z);
  dim3 block(kBlockThreads);
  if (vec_size > 1) {
    if constexpr (is_bfloat16) {  // Only supports vectorization on bfloat16
      using VEC_IN_TYPE = typename Vec<__nv_bfloat16, kVecSize>::Type;
      using VEC_OUT_TYPE = uint2;

      ConvertToCacheTypeKernel<SCALAR_T, CACHE_T, KV_DTYPE, VEC_IN_TYPE, VEC_OUT_TYPE>
          <<<grid, block, 0, stream>>>(src, dst, total_len, elems_num, stride_size, scale, vec_size);
      return;
    }
  }
  // For half and float or vec_size==1, use scalar version
  ConvertToCacheTypeKernel<SCALAR_T, CACHE_T, KV_DTYPE, SCALAR_T, CACHE_T>
      <<<grid, block, 0, stream>>>(src, dst, total_len, elems_num, stride_size, scale, vec_size);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ReverseCacheCopyFlashAttnLayout(SCALAR_T* k_dst, SCALAR_T* v_dst, void** k_list, void** v_list,
                                     size_t* input_offsets, size_t* prefix_offsets, int* block_offsets,
                                     int block_size, int bs, int total_len, int num_heads, int head_size,
                                     int stride_size, float k_scale, float v_scale, cudaStream_t stream) {
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(num_heads, grid_y, grid_z);

  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  ReverseCacheCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_dst, v_dst, k_list, v_list, input_offsets, prefix_offsets, block_offsets, block_size, bs, total_len,
      num_heads, head_size, stride_size, k_scale, v_scale);
}

// Copy full KV(with FlashAttn Layout) from src(cache blocks) to dst(continuous space) for input preparation of FA3 FP8 inference.
template <typename CACHE_T>
void FP8WithPrefixReverseCacheCopyFlashAttnLayout(CACHE_T* k_src, CACHE_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                                                  int* block_offsets, int block_size, int bs, int total_len, int num_heads,
                                                  int head_size, int stride_size, size_t size_of_scalar_t, cudaStream_t stream) {
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(num_heads, grid_y, grid_z);

  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  FP8WithPrefixReverseCacheCopyFlashAttnLayoutKernel<CACHE_T>
      <<<grid_shape, block_shape, 0, stream>>>(k_src, v_src, k_list, v_list, input_offsets, block_offsets, block_size,
                                               bs, total_len, num_heads, head_size, stride_size, size_of_scalar_t);
}

#define CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                 \
  template void CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                                 \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,        \
      int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream, bool is_kv_with_prefix);      \
  template void CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                              \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, int* input_lengths, int* block_offsets,        \
      int block_size, int bs, int req_q_len, int num_heads, int head_size, int stride_size, float k_scale,             \
      float v_scale, cudaStream_t stream);                                                                             \
  template void ReverseCacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                          \
      SCALAR_T * k_dst, SCALAR_T * v_dst, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      int* block_offsets, int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,        \
      float k_scale, float v_scale, cudaStream_t stream);                                                              \
  template void ConvertToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(SCALAR_T * src, CACHE_T * dst, int total_len,          \
                                                                int num_heads, int head_size, int stride_size,         \
                                                                float scale, cudaStream_t stream);

CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, float, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, half, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#undef CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION


template void FP8WithPrefixReverseCacheCopyFlashAttnLayout<uint8_t>(uint8_t* k_src, uint8_t* v_src, void** k_list, void** v_list,
                                                                    size_t* input_offsets, int* block_offsets, int block_size, int bs,
                                                                    int total_len, int num_heads, int head_size, int stride_size,
                                                                    size_t size_of_scalar_t, cudaStream_t stream);

}  // namespace nvidia
}  // namespace llm_kernels
