/* Copyright 2025 Tencent Inc.  All rights reserved.
   modify from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

==============================================================================*/

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"

#ifdef ENABLE_CUDA
// Implementation of mha_varlen_fwd functions

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
    c10::optional<at::Generator> gen_) {
        return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_(
            q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, block_table_, alibi_slopes_, max_seqlen_q,
            max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right,
            softcap, return_softmax, gen_);
}

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
    int num_splits) {
        return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_vllm_flash_attn_v26_(
            q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, block_table_,
            alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap,
            is_rotary_interleaved, num_splits);
}

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
    c10::optional<at::Generator> gen_) {
    return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v25_(
        q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes_, max_seqlen_q, max_seqlen_k, p_dropout,
        softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, return_softmax, gen_);
}

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
    /* default 0.0 */ const bool return_softmax, c10::optional<at::Generator> gen_) {
    return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v26_(
        q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q,
        max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, softcap,
        return_softmax, gen_);
}

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
    int num_splits) {
    return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v25_(
        q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, block_table_,
        alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, is_rotary_interleaved,
        num_splits);
}

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
    int num_splits) {
    return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v26_(
        q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_,
        alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap,
        is_rotary_interleaved, num_splits);
}
#endif