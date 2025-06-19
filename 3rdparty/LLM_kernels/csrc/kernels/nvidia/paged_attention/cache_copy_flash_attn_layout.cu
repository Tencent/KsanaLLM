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
                                               float k_scale, float v_scale) {
  // copy from k,v(without_prefix_offsets) to cache list (input_offsets with prefix offsets)
  int idx = blockIdx.y + blockIdx.z * gridDim.y;
  if (idx < total_len) {
    int batch_idx = 0;
    for (batch_idx = 0; batch_idx < bs; batch_idx++) {
      if (idx < without_prefix_offsets[batch_idx + 1]) {
        break;
      }
    }
    // size_t prefix_limit = prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx] + input_offsets[batch_idx];
    int cur_batch_token_idx_with_prefix =
        (prefix_offsets[batch_idx + 1] - prefix_offsets[batch_idx]) + (idx - without_prefix_offsets[batch_idx]);
    int cur_block_offset = cur_batch_token_idx_with_prefix / block_size;
    int cur_batch_offset = cur_batch_token_idx_with_prefix % block_size;
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

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
__global__ void ConvertToScalarKernel(CACHE_T* src, SCALAR_T* dst, int* src_table, int* dst_table, int table_len,
                                      int data_num, float k_scale, float v_scale) {
  // 计算当前线程处理的块索引，考虑多维网格
  int idx = blockIdx.x + blockIdx.y * gridDim.x;

  // 确保不超出表长度
  if (idx < table_len) {
    // 获取源数据和目标数据的起始位置
    size_t src_offset = size_t(src_table[idx]) * data_num;
    size_t dst_offset = size_t(dst_table[idx]) * data_num;

    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kAuto) {
      // 每个线程处理一个数据元素
      for (int i = threadIdx.x; i < data_num; i += blockDim.x) {
        // 直接赋值
        dst[dst_offset + i] = src[src_offset + i];
      }
    } else if constexpr (std::is_same<SCALAR_T, __nv_bfloat16>::value && std::is_same<CACHE_T, uint8_t>::value) {
      if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
        // 静态确定FP8类型为E4M3
        constexpr __nv_fp8_interpretation_t fp8_type = __NV_E4M3;
        constexpr int vec_size = 8;

        // 计算步长，避免在循环中重复计算
        const int stride = blockDim.x * vec_size;

        // 假设数据是8的倍数，直接使用向量化处理
        for (int i = threadIdx.x * vec_size; i < data_num; i += stride) {
          // 直接将源数据指针强制转换为uint2*类型，避免数据拷贝
          const uint2& src_vec = *reinterpret_cast<const uint2*>(&src[src_offset + i]);

          // 使用scaled_vec_conversion一次性转换8个数据
          *reinterpret_cast<bf16_8_t*>(&dst[dst_offset + i]) =
              fp8::scaled_vec_conversion<bf16_8_t, uint2>(src_vec, v_scale, fp8_type);
        }
      } else if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
        // 静态确定FP8类型为E5M2
        constexpr __nv_fp8_interpretation_t fp8_type = __NV_E5M2;
        constexpr int vec_size = 8;

        // 计算步长，避免在循环中重复计算
        const int stride = blockDim.x * vec_size;

        // 假设数据是8的倍数，直接使用向量化处理
        for (int i = threadIdx.x * vec_size; i < data_num; i += stride) {
          // 直接将源数据指针强制转换为uint2*类型，避免数据拷贝
          const uint2& src_vec = *reinterpret_cast<const uint2*>(&src[src_offset + i]);

          // 使用scaled_vec_conversion一次性转换8个数据
          *reinterpret_cast<bf16_8_t*>(&dst[dst_offset + i]) =
              fp8::scaled_vec_conversion<bf16_8_t, uint2>(src_vec, v_scale, fp8_type);
        }
      } else {
        // 其他KV_DTYPE情况，每个线程处理一个数据元素
        for (int i = threadIdx.x; i < data_num; i += blockDim.x) {
          dst[dst_offset + i] = fp8::scaled_convert<SCALAR_T, CACHE_T, KV_DTYPE>(src[src_offset + i], v_scale);
        }
      }
    } else {
      // 其他情况，每个线程处理一个数据元素
      for (int i = threadIdx.x; i < data_num; i += blockDim.x) {
        // 使用scaled_convert进行转换
        dst[dst_offset + i] = fp8::scaled_convert<SCALAR_T, CACHE_T, KV_DTYPE>(src[src_offset + i], v_scale);
      }
    }
  }
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void CacheCopyFlashAttnLayout(SCALAR_T* k_src, SCALAR_T* v_src, void** k_list, void** v_list, size_t* input_offsets,
                              size_t* prefix_offsets, size_t* without_prefix_offsets, int* block_offsets,
                              int block_size, int bs, int total_len, int num_heads, int head_size, int stride_size,
                              float k_scale, float v_scale, cudaStream_t stream) {
  int grid_y = std::min(total_len, MAX_BLOCKS_PER_GRID_Y);
  int grid_z = (total_len + MAX_BLOCKS_PER_GRID_Y - 1) / MAX_BLOCKS_PER_GRID_Y;
  dim3 grid_shape(num_heads, grid_y, grid_z);

  dim3 block_shape(std::min(head_size, MAX_THREADS_PER_BLOCK));
  CacheCopyFlashAttnLayoutKernel<SCALAR_T, CACHE_T, KV_DTYPE><<<grid_shape, block_shape, 0, stream>>>(
      k_src, v_src, k_list, v_list, input_offsets, prefix_offsets, without_prefix_offsets, block_offsets, block_size,
      bs, total_len, num_heads, head_size, stride_size, k_scale, v_scale);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void ConvertToScalar(CACHE_T* src, SCALAR_T* dst, int* src_table, int* dst_table, int table_len, int data_num,
                     float k_scale, float v_scale, cudaStream_t stream) {
  // 计算网格和块的大小，处理边界问题
  int grid_x = std::min(table_len, llm_kernels::utils::DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM);
  int grid_y = (table_len + llm_kernels::utils::DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM - 1) /
               llm_kernels::utils::DEFAULT_CUDA_GPU_DEVICE_MAX_BLOCKS_NUM;
  dim3 grid_shape(grid_x, grid_y);

  // 计算块大小
  int block_size = std::min(data_num, llm_kernels::utils::DEFAULT_CUDA_BLOCK_HALF_THREADS_NUM);

  // TODO(zakwang): 完全支持向量化转换, 因为block_size包含block_num目前一定为8的倍数
  constexpr int kVecSize = 8;
  if (block_size % kVecSize == 0) {
    // 调用内核函数
    ConvertToScalarKernel<SCALAR_T, CACHE_T, KV_DTYPE>
        <<<grid_shape, block_size, 0, stream>>>(src, dst, src_table, dst_table, table_len, data_num, k_scale, v_scale);
  }
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

#define CACHE_COPY_FLASH_ATTN_LAYOUT_FUNCTION_DECLARATION(SCALAR_T, CACHE_T, KV_DTYPE)                                 \
  template void CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                                 \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, size_t* input_offsets, size_t* prefix_offsets, \
      size_t* without_prefix_offsets, int* block_offsets, int block_size, int bs, int total_len, int num_heads,        \
      int head_size, int stride_size, float k_scale, float v_scale, cudaStream_t stream);                              \
  template void CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(                                              \
      SCALAR_T * k_src, SCALAR_T * v_src, void** k_list, void** v_list, int* input_lengths, int* block_offsets,        \
      int block_size, int bs, int req_q_len, int num_heads, int head_size, int stride_size, float k_scale,             \
      float v_scale, cudaStream_t stream);                                                                             \
  template void ConvertToScalar<SCALAR_T, CACHE_T, KV_DTYPE>(CACHE_T * src, SCALAR_T * dst, int* src_table,            \
                                                             int* dst_table, int table_len, int data_num,              \
                                                             float k_scale, float v_scale, cudaStream_t stream);

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

}  // namespace nvidia
}  // namespace llm_kernels
