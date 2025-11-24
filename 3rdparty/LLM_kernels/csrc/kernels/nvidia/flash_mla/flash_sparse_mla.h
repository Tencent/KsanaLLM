/*
 * Adapted from
 * https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/pybind.cpp
 *
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
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "csrc/utils/quant_type.h"

namespace llm_kernels {
namespace nvidia {

// DecodingAttnImplMeta - A struct to hold metadata for Decoding Attention Implementation (i.e. SM90 Dense BF16, SM90
// Sparse FP8, etc.)
struct DecodingAttnImplMeta {
  int num_sm_parts = 1;
  int fixed_overhead_num_blocks = 0;
  int k_block_size = 0;
};

DecodingAttnImplMeta GetAttnImplMeta(int num_q_tokens_per_head_k, int num_heads_k, int num_heads_q, bool is_fp8_kvcache,
                                     bool is_sparse_attn);

// Wrapper of get_mla_metadata_kernel
void InvokeGetSparseMlaMetadata(int* seqlens_k_ptr,  // batch_size
                                int batch_size, int num_q_tokens_per_head_k, int num_heads_k, int num_heads_q,
                                bool is_fp8_kvcache, int topk, cudaStream_t stream,
                                int* tile_scheduler_metadata_ptr,  // num_sm_parts x TileSchedulerMetaDataSize
                                int* num_splits_ptr                // batch_size + 1
);

// Wrapper of flash_fwd_splitkv_mla_fp8_sparse_kernel and flash_fwd_mla_combine_kernel
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaWithKVCache(
    SCALAR_T* q_ptr,      // batch_size x seqlen_q_ori x num_heads_q x head_size_k
    CACHE_T* kcache_ptr,  // num_blocks x block_size x num_heads_k x head_size_k (when is_fp8 is False) or
                          // num_blocks x block_size x num_heads_k x 656 (when is_fp8 is True)
    int batch_size, int seqlen_q_ori, int num_heads_q, int num_heads_k, int head_size_k, int head_size_v,
    int k_batch_stride,  // stride of kcache_ptr in dimension 0, can be larger than block_size x num_heads_k x
                         // head_size_k/656
    int max_num_blocks_per_seq, int num_blocks, int block_size, int num_sm_parts,
    int* seqlens_k_ptr,    // batch_size
    int* block_table_ptr,  // batch_size x max_num_blocks_per_seq
    float softmax_scale, bool is_causal,
    int* tile_scheduler_metadata_ptr,  // num_sm_parts x TileSchedulerMetaDataSize
    int* num_splits_ptr,               // batch_size + 1
    bool is_fp8,
    int* indices_ptr,  // None (dense mode), or batch_size x seqlen_q x topk (sparse mode)
    int topk, cudaStream_t stream, float* workspace_ptr,  // Used for softmax_lse, softmax_lse_accum, out_accum
    SCALAR_T* out_ptr                                     // batch_size x seqlen_q_ori x num_heads_q x head_size_v
);

// Wrapper of sparse_attn_fwd_kernel
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeFlashSparseMlaPrefill(SCALAR_T* q_ptr,      // batch_size x seqlen_q x num_heads_q x head_size_k
                                 CACHE_T* kcache_ptr,  // num_blocks x block_size x num_heads_k x head_size_k
                                 int* indices_ptr,     // batch_size x seqlen_q x topk
                                 int seqlen_q, int seqlen_k, int num_heads_q, int num_heads_k, int head_size_k,
                                 int head_size_v, int topk, float sm_scale, cudaStream_t stream,
                                 float* workspace_ptr,  // Used for max_logits, lse
                                 SCALAR_T* out_ptr      // batch_size x seqlen_q x num_heads_q x head_size_v
);

}  // namespace nvidia
}  // namespace llm_kernels
