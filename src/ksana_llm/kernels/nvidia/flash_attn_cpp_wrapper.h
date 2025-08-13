/* Copyright 2024 Tencent Inc.  All rights reserved.
   modify from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

==============================================================================*/
#pragma once

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

namespace ksana_llm {

// attention for prefill
// FA3 function pointer - returns tuple, will be converted to vector for compatibility
using mha_fwd_fa3_ptr = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> (*)(
    at::Tensor q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d)
                   // if there is page_table.
    at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k,
                   // dv) if there is page_table.
    std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    std::optional<at::Tensor> q_v_,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> out_,    // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> cu_seqlens_q_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
    std::optional<at::Tensor>
        seqused_q_,  // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<at::Tensor>
        seqused_k_,  // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,      // (b_k, max_num_pages_per_seq)
    std::optional<at::Tensor> kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<at::Tensor> leftpad_k_,       // b
    std::optional<at::Tensor> rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> seqlens_rotary_,  // b
    std::optional<at::Tensor> q_descale_,       // (b, h_k), not (b, h)
    std::optional<at::Tensor> k_descale_,       // (b, h_k)
    std::optional<at::Tensor> v_descale_,       // (b, h_k)
    std::optional<double> softmax_scale_, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, double softcap,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
    int64_t num_splits, std::optional<bool> pack_gqa_, int64_t sm_margin);

using mha_varlen_fwd_vllm_flash_attn_v26_ptr = std::vector<at::Tensor> (*)(
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
using mha_fwd_kvcache_vllm_flash_attn_v26_ptr = std::vector<at::Tensor> (*)(
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

// NOTE(karlluo): this function is wrapped in flash_attn_2_cuda.cpython-39-x86_64-linux-gnu.so
using mha_varlen_fwd_flash_attn_v25_ptr = std::vector<at::Tensor> (*)(
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

using mha_varlen_fwd_flash_attn_v26_ptr = std::vector<at::Tensor> (*)(
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

// mha_fwd_kvcache api of flash-attn.
// Added for compiling succeed when enable_blocked_multi_token_forwarding_kv, not used in runtime.  TBD@xingjinglu
// attention for decode
using mha_fwd_kvcache_flash_attn_v25_ptr = std::vector<at::Tensor> (*)(
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

// Since 2.7.2 upgrade to this api.
// Added for compiling succeed when enable_blocked_multi_token_forwarding_kv, not used in runtime.  TBD@xingjinglu
// attention for decode
using mha_fwd_kvcache_flash_attn_v26_ptr = std::vector<at::Tensor> (*)(
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

// Function declarations only - implementations moved to flash_attn_cpp_wrapper.cpp

// FA3 function declaration
std::vector<at::Tensor> mha_fwd(
    at::Tensor q,  // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    at::Tensor k,  // (b_k, s_k, h_k, d) or (total_k, h_k, d) if there is cu_seqlens_k or (num_pages, page_size, h_k, d)
                   // if there is page_table.
    at::Tensor v,  // (b_k, s_k, h_k, dv) or (total_k, h_k, dv) if there is cu_seqlens_k or (num_pages, page_size, h_k,
                   // dv) if there is page_table.
    std::optional<at::Tensor> k_new_,  // (b, s_k_new, h_k, d) or (total_k_new, h_k, d) if there is cu_seqlens_k_new
    std::optional<at::Tensor> v_new_,  // (b, s_k_new, h_k, dv) or (total_k_new, h_k, dv) if there is cu_seqlens_k_new
    std::optional<at::Tensor> q_v_,    // (b, s_q, h, dv) or (total_q_new, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> out_,    // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    std::optional<at::Tensor> cu_seqlens_q_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_,      // b+1
    std::optional<at::Tensor> cu_seqlens_k_new_,  // b+1
    std::optional<at::Tensor>
        seqused_q_,  // b. If given, only this many elements of each batch element's queries and outputs are used.
    std::optional<at::Tensor>
        seqused_k_,  // b. If given, only this many elements of each batch element's keys are used.
    std::optional<int64_t> max_seqlen_q_,
    std::optional<int64_t> max_seqlen_k_,
    std::optional<at::Tensor> page_table_,      // (b_k, max_num_pages_per_seq)
    std::optional<at::Tensor> kv_batch_idx_,    // b. indices to index into the KV cache
    std::optional<at::Tensor> leftpad_k_,       // b
    std::optional<at::Tensor> rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<at::Tensor> seqlens_rotary_,  // b
    std::optional<at::Tensor> q_descale_,       // (b, h_k), not (b, h)
    std::optional<at::Tensor> k_descale_,       // (b, h_k)
    std::optional<at::Tensor> v_descale_,       // (b, h_k)
    std::optional<double> softmax_scale_, bool is_causal, int64_t window_size_left, int64_t window_size_right,
    int64_t attention_chunk, double softcap,
    bool is_rotary_interleaved,  // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    std::optional<at::Tensor> scheduler_metadata_,  // (b + 1)
    int64_t num_splits, std::optional<bool> pack_gqa_, int64_t sm_margin);

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

}  // namespace ksana_llm