/* Copyright 2024 Tencent Inc.  All rights reserved.
   modify from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

==============================================================================*/
#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#ifdef ENABLE_VLLM_FLASH_ATTN_2
#  if defined(ENABLE_VLLM_FLASH_ATTN_MINOR_6) || defined(ENABLE_VLLM_FLASH_ATTN_MINOR_7)
// attention for prefill
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,        // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                          // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                          // page_block_size x num_heads_k x head_size if there's a block_table.
    c10::optional<at::Tensor> &out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,   // b+1
    const at::Tensor &cu_seqlens_k,   // b+1
    c10::optional<at::Tensor>
        &seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    c10::optional<at::Tensor> &block_table_,   // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right, const float softcap, const bool return_softmax,
    c10::optional<at::Generator> gen_);
// attention for decode
std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor &q,             // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    const at::Tensor &vcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    c10::optional<const at::Tensor> &k_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &v_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &seqlens_k_,        // batch_size
    c10::optional<const at::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
    c10::optional<at::Tensor> &block_table_,            // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_,           // num_heads or batch_size x num_heads
    c10::optional<at::Tensor> &out_,                    // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, const float softcap,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits);
#  endif
#endif

#ifdef ENABLE_FLASH_ATTN_2

// mha_varlen_fwd api of flash-attn.
#  if defined(ENABLE_FLASH_ATTN_MINOR_5)
// NOTE(karlluo): this function is wrapped in flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,                    // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &v,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor> &out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,   // b+1
    const at::Tensor &cu_seqlens_k,   // b+1
    c10::optional<at::Tensor>
        &seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    c10::optional<at::Tensor> &alibi_slopes_,  // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right, const bool return_softmax,
    c10::optional<at::Generator> gen_);
#  else
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,                    // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor &k,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &v,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
    c10::optional<at::Tensor> &out_,  // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const at::Tensor &cu_seqlens_q,   // b+1
    const at::Tensor &cu_seqlens_k,   // b+1
    c10::optional<at::Tensor>
        &seqused_k,  // b. If given, only this many elements of each batch element's keys are used.
    c10::optional<const at::Tensor> &leftpad_k_,  // indices that the KV cache starts. [batch_size,], nullptr, default 0
    c10::optional<at::Tensor> &block_table_,      //
    c10::optional<at::Tensor> &alibi_slopes_,     // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right, const float softcap,
    /* default 0.0 */ const bool return_softmax, c10::optional<at::Generator> gen_);
#  endif

// mha_fwd_kvcache api of flash-attn.
#  ifdef ENABLE_FLASH_ATTN_MINOR_5
// Added for compiling succeed when enable_blocked_multi_token_forwarding_kv, not used in runtime.  TBD@xingjinglu
// attention for decode
std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor &q,             // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    const at::Tensor &vcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    c10::optional<const at::Tensor> &k_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &v_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &seqlens_k_,        // batch_size
    c10::optional<const at::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
    c10::optional<at::Tensor> &block_table_,            // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_,           // num_heads or batch_size x num_heads
    c10::optional<at::Tensor> &out_,                    // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits);

#  else  // Since 2.7.2 upgrade to this api.
// Added for compiling succeed when enable_blocked_multi_token_forwarding_kv, not used in runtime.  TBD@xingjinglu
// attention for decode
std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor &q,             // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor &kcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    const at::Tensor &vcache,  // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                               // num_heads_k x head_size if there's a block_table.
    c10::optional<const at::Tensor> &k_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &v_,                // batch_size x seqlen_knew x num_heads_k x head_size
    c10::optional<const at::Tensor> &seqlens_k_,        // batch_size
    c10::optional<const at::Tensor> &rotary_cos_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &rotary_sin_,       // seqlen_ro x (rotary_dim / 2)
    c10::optional<const at::Tensor> &cache_batch_idx_,  // indices to index into the KV cache
    c10::optional<const at::Tensor> &leftpad_k_,  // indices that the KV cache starts. [batch_size,], nullptr, default 0
    c10::optional<at::Tensor> &block_table_,      // batch_size x max_num_blocks_per_seq
    c10::optional<at::Tensor> &alibi_slopes_,     // num_heads or batch_size x num_heads
    c10::optional<at::Tensor> &out_,              // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
    const float softcap,         // Since v2.6.0, support this param.
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits);
#  endif

#endif
