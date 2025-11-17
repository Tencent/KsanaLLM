// Copyright 2024 Tencent Inc.  All rights reserved.

#pragma once

#include <cuda_runtime.h>
#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
struct PagedAttentionParams {
 public:
  SCALAR_T* out_;           // [num_seqs, num_heads, head_size]
  const SCALAR_T* query_;   // [num_seqs, num_heads, head_size]
  CACHE_T** key_caches_;    // num_seqs x [seq_blocks, num_kv_heads, head_size/x, block_size, x]
  CACHE_T** value_caches_;  // num_seqs x [seq_blocks, num_kv_heads, head_size, block_size]
  int num_head_repeats_;    // num_heads / num_kv_heads
  int num_seqs_;
  int num_heads_;
  int head_size_;
  int q_stride_;
  int kv_head_stride_;
  float k_scale_;
  float v_scale_;
  float scale_;
  const int* cache_offsets_;  // [num_seqs]
  const int* context_lens_;   // [num_seqs]
  int max_context_len_;
  int block_size_;
  cudaStream_t stream_;

  const float* alibi_slopes_ = nullptr;

  bool use_v1_;
  int max_num_partitions_;
  float* exp_sums_ = nullptr;    // [num_seqs, num_heads, max_num_partitions]
  float* max_logits_ = nullptr;  // [num_seqs, num_heads, max_num_partitions]
  SCALAR_T* tmp_out_ = nullptr;  // [num_seqs, num_heads, max_num_partitions, head_size]

 public:
  int GetTmpOutNumel() const;
  int GetExpSumsNumel() const;
  int GetMaxLogitsNumel() const;
  int GetMaxNumPartitions() const;
  bool IsUseV1() const;
  size_t GetWorkSize() const;
  void SetWorkSpace(void* workspace, size_t work_size);
};

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
class PagedAttentionCuda {
 public:
  void SetConfig(int num_kv_heads, int num_heads, int head_size, int block_size,
                 int stride_size, float k_scale, float v_scale);
  size_t GetWorkspaceSize(int num_seqs, int max_context_len);
  void SetInput(SCALAR_T* out,             // [num_seqs, num_heads, head_size]
                const SCALAR_T* query,     // [num_seqs, num_heads, head_size]
                CACHE_T** key_caches,      // num_seqs x [seq_blocks, num_heads, head_size/x, block_size, x]
                CACHE_T** value_caches,    // num_seqs x [seq_blocks, num_heads, head_size, block_size]
                const int* cache_offsets,  // [num_heads]
                const int* context_lens,   // [num_seqs]
                int max_context_len, int num_seqs, cudaStream_t stream, void* workspace, size_t work_size,
                const float* alibi_slopes = nullptr);
  void Forward();

 private:
  PagedAttentionParams<SCALAR_T, CACHE_T, KV_DTYPE> params_;
};

}  // namespace nvidia
}  // namespace llm_kernels
