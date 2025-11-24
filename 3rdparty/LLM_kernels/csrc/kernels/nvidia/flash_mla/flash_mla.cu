/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 * Copyright (c) 2025 DeepSeek
 *
 * MIT License
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Adapted from
 * [FlashMLA Project] https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/flash_api.cpp
 */

#ifdef ENABLE_FLASH_MLA
#  include <cassert>
#  include <cstdint>
#  include <cstring>
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"

#  include "csrc/kernels/nvidia/flash_mla/kernels/config.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/fp8_flash_fwd_mla.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/get_mla_metadata.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/mla_combine.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/params.h"
#  include "csrc/kernels/nvidia/flash_mla/kernels/splitkv_mla.h"

#  include <cutlass/fast_math.h>

#  include <cuda_fp16.h>
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#  include <device_launch_parameters.h>
namespace llm_kernels {
namespace nvidia {
inline std::vector<size_t> GetStride(std::vector<int> shape) {
  std::vector<size_t> strides(shape.size(), 1);
  for (int i = shape.size() - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

void GetNumSmParts(FlashMlaWorkspaceMap& workspace_param, const int num_heads_per_head_k, const int num_heads_k,
                   int rank) {
  static int sm_count = 0;
  if (sm_count == 0) {
    cudaDeviceProp dprops;
    cudaGetDeviceProperties(&dprops, rank);
    sm_count = dprops.multiProcessorCount;
  }
  workspace_param.num_sm_parts = sm_count / num_heads_k / cutlass::ceil_div(num_heads_per_head_k, Config::BLOCK_SIZE_M);
}

size_t ApplyWorkspaceBuffer(void* workspace_ptr, FlashMlaWorkspaceMap& workspace_param, int batch_size, int num_heads,
                            int q_seq_per_hk, int head_size_v, cudaStream_t stream = nullptr) {
  const int total_num_splits = batch_size + workspace_param.num_sm_parts;

  size_t softmax_lse_size = sizeof(float) * batch_size * num_heads * q_seq_per_hk;
  size_t softmax_lse_accum_size = sizeof(float) * total_num_splits * num_heads * q_seq_per_hk;
  size_t out_accum_size = sizeof(float) * total_num_splits * num_heads * q_seq_per_hk * head_size_v;
  void* workspace_align_ptr = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(workspace_ptr) + 1023) & ~1023);
  softmax_lse_size = (softmax_lse_size + 1023) & ~1023;
  softmax_lse_accum_size = (softmax_lse_accum_size + 1023) & ~1023;
  out_accum_size = (out_accum_size + 1023) & ~1023;
  char* workspace = reinterpret_cast<char*>(workspace_align_ptr);

  workspace_param.softmax_lse_ptr = reinterpret_cast<float*>(workspace);
  workspace += softmax_lse_size;

  workspace_param.softmax_lse_accum_ptr = reinterpret_cast<float*>(workspace);
  workspace += softmax_lse_accum_size;

  workspace_param.out_accum_ptr = reinterpret_cast<float*>(workspace);
  workspace += out_accum_size;

  size_t total_size = workspace - reinterpret_cast<char*>(workspace_align_ptr);
  return total_size;
}

void InvokeGetMlaMetadata(int* b_seqlen, FlashMlaWorkspaceMap& workspace_param, int batch_size, cudaStream_t stream) {
  // tile_scheduler_metadata [num_sm_parts, TileSchedulerMetaDataSize]
  // num_splits [tokens_num +  1]
  Mla_metadata_params params = {};
  params.seqlens_k_ptr = b_seqlen;
  params.tile_scheduler_metadata_ptr = workspace_param.tile_scheduler_metadata_ptr;
  params.num_splits_ptr = workspace_param.num_splits_ptr;
  params.batch_size = batch_size;
  params.block_size_n = Config::PAGE_BLOCK_SIZE;
  params.fixed_overhead_num_blocks = Config::FIXED_OVERHEAD_NUM_BLOCKS;
  params.num_sm_parts = workspace_param.num_sm_parts;
  llm_kernels::nvidia::GetMlaMetadata(params, stream);
}

void SetFlashMlaAttribute(const int max_batch_size, cudaStream_t stream) {
  llm_kernels::nvidia::SetMlaMetadataKernelAttribute(max_batch_size, stream);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashMla(CACHE_T* q, CACHE_T* k_buffer, const int seqlen_q_ori, float sm_scale, void* block_table_ptr,
                    void* b_seqlen, void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace,
                    void* att_out, int batch_size, int num_heads, int kv_lora_rank, int qk_rope_head_dim, int page_size,
                    float k_scale, float v_scale, int max_blocks_per_seq, int rank, size_t block_num,
                    cudaStream_t stream) {
  // q [batch_size, seqlen_q_ori, num_heads_q, head_size_k]
  const int num_heads_q = num_heads;
  const int head_size_k = kv_lora_rank + qk_rope_head_dim;  // must 576
  const bool is_casal = seqlen_q_ori != 1;

  // k [num_blocks, page_block_size, num_heads_k, head_size_k]   = [block_num, page_size, 1, kv_lora_rank +
  // qk_rope_head_dim] v [num_blocks, page_block_size, num_heads_k, head_size_v] = [block_num, page_size, 1,
  // kv_lora_rank]
  constexpr int num_heads_k = 1;
  const int head_size_v = kv_lora_rank;  // 512

  const int num_q_heads_per_hk = num_heads_q / num_heads_k;
  const int q_seq_per_hk = seqlen_q_ori * num_q_heads_per_hk;
  num_heads = num_heads_k;

  // block_table [batch_size, max_blocks_per_seq]
  const std::vector<size_t>& block_table_stride = GetStride({batch_size, max_blocks_per_seq});
  // o [batch_size, q_seq_per_hk, num_heads, head_size_v] T
  // softmax_lse [batch_size, num_heads, q_seq_per_hk] Float
  const std::vector<size_t>& o_stride = GetStride({batch_size, q_seq_per_hk, num_heads, head_size_v});
  const std::vector<size_t>& q_stride = GetStride({batch_size, q_seq_per_hk, num_heads, head_size_k});
  const std::vector<size_t>& kcache_stride =
      GetStride({static_cast<int>(block_num), page_size, num_heads_k, head_size_k});

  // tile_scheduler_metadata [num_sm_parts, TileSchedulerMetaDataSize]
  // num_splits [batch_size + 1]
  // softmax_lse_accum [batch_size, num_sm_parts, num_heads, q_seq_per_hk] Float
  // out_accum [batch_size + num_sm_parts, num_heads, q_seq_per_hk, head_size_v] Float
  FlashMlaWorkspaceMap workspace_param = {};
  GetNumSmParts(workspace_param, q_seq_per_hk, num_heads_k, rank);
  ApplyWorkspaceBuffer(workspace, workspace_param, batch_size, num_heads, q_seq_per_hk, head_size_v);
  workspace_param.tile_scheduler_metadata_ptr = reinterpret_cast<int*>(tile_scheduler_metadata_ptr);
  workspace_param.num_splits_ptr = reinterpret_cast<int*>(num_splits_ptr);

  const int total_num_splits = batch_size + workspace_param.num_sm_parts;

  Flash_fwd_mla_params params = {};
  params.b = batch_size;
  params.s_q = seqlen_q_ori;
  params.q_seq_per_hk = q_seq_per_hk;
  params.seqlens_k_ptr = reinterpret_cast<int*>(b_seqlen);
  params.h_q = num_heads_q;
  params.h_k = num_heads_k;
  params.num_blocks = block_num;
  params.q_head_per_hk = num_q_heads_per_hk;
  params.is_causal = is_casal;
  params.d = head_size_k;
  params.d_v = head_size_v;
  params.scale_softmax = sm_scale;
  params.scale_softmax_log2 = float(sm_scale * M_LOG2E);
  // Set the pointers and strides.
  params.q_ptr = q;
  params.k_ptr = k_buffer;
  params.o_ptr = att_out;
  params.softmax_lse_ptr = workspace_param.softmax_lse_ptr;
  // All stride are in elements, not bytes.
  params.q_batch_stride = q_stride[0];
  params.k_batch_stride = kcache_stride[0];
  params.o_batch_stride = o_stride[0];
  params.q_row_stride = q_stride[q_stride.size() - 3];
  params.k_row_stride = kcache_stride[kcache_stride.size() - 3];
  params.o_row_stride = o_stride[o_stride.size() - 3];
  params.q_head_stride = q_stride[q_stride.size() - 2];
  params.k_head_stride = kcache_stride[kcache_stride.size() - 2];
  params.o_head_stride = o_stride[o_stride.size() - 2];

  params.block_table = reinterpret_cast<int*>(block_table_ptr);
  params.block_table_batch_stride = block_table_stride[0];
  params.page_block_size = page_size;

  params.tile_scheduler_metadata_ptr = workspace_param.tile_scheduler_metadata_ptr;
  params.num_sm_parts = workspace_param.num_sm_parts;
  params.num_splits_ptr = workspace_param.num_splits_ptr;

  params.total_num_splits = total_num_splits;
  params.softmax_lseaccum_ptr = workspace_param.softmax_lse_accum_ptr;
  params.oaccum_ptr = workspace_param.out_accum_ptr;

  if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3 ||
                KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
    params.h = params.h_k;
    params.h_h_k_ratio = params.h / params.h_k;
    params.ngroups = params.q_head_per_hk;
    params.seqlen_q = params.q_seq_per_hk;
    params.cu_seqlens_k = params.seqlens_k_ptr;
    // It is difficult and unnecessary to pass a float type function parameter to a pointer for storage, so we store the
    // descales in float directly instead of pointers.
    params.descale_q = k_scale;
    params.descale_k = k_scale;
  }

  assert(head_size_k == 576);
  if constexpr (std::is_same<SCALAR_T, half>::value) {
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::half_t, 576>(params, stream);
    } else {
      run_flash_splitkv_mla_kernel<cutlass::half_t>(params, stream);
      run_flash_mla_combine_kernel<cutlass::half_t>(params, stream);
    }
  } else if constexpr (std::is_same<SCALAR_T, __nv_bfloat16>::value) {
    if constexpr (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      run_mha_fwd_splitkv_mla<cutlass::float_e4m3_t, cutlass::bfloat16_t, 576>(params, stream);
    } else {
      run_flash_splitkv_mla_kernel<cutlass::bfloat16_t>(params, stream);
      run_flash_mla_combine_kernel<cutlass::bfloat16_t>(params, stream);
    }
  }
}

#  define INVOKE_FLASH_MLA(SCALAR_T, CACHE_T, KV_DTYPE)                                                          \
    template void InvokeFlashMla<SCALAR_T, CACHE_T, KV_DTYPE>(                                                   \
        CACHE_T * q, CACHE_T * k_buffer, const int seqlen_q_ori, float sm_scale, void* block_table_ptr,          \
        void* b_seqlen, void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace, void* att_out, \
        int tokens_num, int num_heads, int kv_lora_rank, int qk_rope_head_dim, int page_size, float k_scale,     \
        float v_scale, int max_blocks_per_seq, int rank, size_t block_num, cudaStream_t stream)
INVOKE_FLASH_MLA(half, half, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
INVOKE_FLASH_MLA(float, float, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
INVOKE_FLASH_MLA(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#  undef INVOKE_FLASH_MLA
}  // namespace nvidia
}  // namespace llm_kernels
#else
#  include <cutlass/fast_math.h>
#  include <cassert>
#  include <cstdint>
#  include <cstring>
#  include "csrc/kernels/nvidia/flash_mla/flash_mla.h"

#  include <cuda_fp16.h>
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#  include <device_launch_parameters.h>
namespace llm_kernels {
namespace nvidia {
void SetFlashMlaAttribute(const int max_batch_size, cudaStream_t stream) {}
void InvokeGetMlaMetadata(int* b_seqlen, FlashMlaWorkspaceMap& workspace_param, int tokens_num, cudaStream_t stream) {}
void GetNumSmParts(FlashMlaWorkspaceMap& workspace_param, const int num_heads_per_head_k, const int num_heads_k,
                   int rank) {}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashMla(CACHE_T* q, CACHE_T* k_buffer, const int seqlen_q_ori, float sm_scale, void* block_table_ptr,
                    void* b_seqlen, void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace,
                    void* att_out, int tokens_num, int num_heads, int kv_lora_rank, int qk_rope_head_dim, int page_size,
                    float k_scale, float v_scale, int max_blocks_per_seq, int rank, size_t block_num,
                    cudaStream_t stream) {}

#  define INVOKE_FLASH_MLA(SCALAR_T, CACHE_T, KV_DTYPE)                                                          \
    template void InvokeFlashMla<SCALAR_T, CACHE_T, KV_DTYPE>(                                                   \
        CACHE_T * q, CACHE_T * k_buffer, const int seqlen_q_ori, float sm_scale, void* block_table_ptr,          \
        void* b_seqlen, void* tile_scheduler_metadata_ptr, void* num_splits_ptr, void* workspace, void* att_out, \
        int tokens_num, int num_heads, int kv_lora_rank, int qk_rope_head_dim, int page_size, float k_scale,     \
        float v_scale, int max_blocks_per_seq, int rank, size_t block_num, cudaStream_t stream)
INVOKE_FLASH_MLA(half, half, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
INVOKE_FLASH_MLA(float, float, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
INVOKE_FLASH_MLA(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
INVOKE_FLASH_MLA(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
INVOKE_FLASH_MLA(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#  undef INVOKE_FLASH_MLA
}  // namespace nvidia
}  // namespace llm_kernels
#endif
