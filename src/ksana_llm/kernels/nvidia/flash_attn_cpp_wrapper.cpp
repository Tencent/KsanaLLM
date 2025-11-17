/* Copyright 2025 Tencent Inc.  All rights reserved.
   modify from https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/flash_api.cpp

==============================================================================*/
#include <tuple>

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/attention_backend/flash_attention_backend.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"

#ifdef ENABLE_CUDA
namespace ksana_llm {
// Helper to get user preference for FlashAttention implementation.
static AttnBackendConfig::FlashAttnImplChoice GetFlashAttnImplChoice() {
  RuntimeConfig runtime_config;
  auto env = Singleton<Environment>::GetInstance();
  if (env) {
    env->GetRuntimeConfig(runtime_config);
    return runtime_config.attn_backend_config.flash_attn_impl_choice;
  }
  return AttnBackendConfig::FlashAttnImplChoice::AUTO;
}

// Availability helpers
static bool HasFA3() { return ksana_llm::FlashAttentionBackend::mha_fwd_fa3_ != nullptr; }

// Public function to check if FA3 is actually being used
bool IsUsingFA3() {
  // Get the user's flash attention implementation choice
  auto choice = GetFlashAttnImplChoice();

  // Check if user explicitly chose FA3
  if (choice == AttnBackendConfig::FlashAttnImplChoice::FA3) {
    return true;
  }

  // Check if user chose AUTO and FA3 would be selected
  if (choice == AttnBackendConfig::FlashAttnImplChoice::AUTO) {
    return HasFA3();  // AUTO mode selects FA3 if available
  }

  // For other choices (FA2_V25, FA2_V26, VLLM_V26), FA3 is not used
  return false;
}
static bool HasVllmV26() {
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_ != nullptr &&
         ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_vllm_flash_attn_v26_ != nullptr;
}
static bool HasFA2V26() {
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v26_ != nullptr &&
         ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v26_ != nullptr;
}
static bool HasFA2V25() {
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v25_ != nullptr &&
         ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v25_ != nullptr;
}

// Call helpers (wrap type conversions and refs)
static std::vector<at::Tensor> CallFA3Varlen(const MhaVarlenFwdParams &params) {
  if (params.alibi_slopes.has_value()) {
    KLLM_THROW("Flash Attention 3 does not support alibi_slopes.");
  }
  return mha_fwd_fa3(params.q, params.k, params.v, c10::nullopt, c10::nullopt, params.qv, params.out, params.seqlen_q,
                     params.seqlen_k, c10::nullopt, params.seqused_q, params.seqused_k, params.max_seqlen_q,
                     params.max_seqlen_k, c10::nullopt, c10::nullopt, params.leftpad_k, c10::nullopt, c10::nullopt,
                     c10::nullopt, params.q_descale, params.k_descale, params.v_descale,
                     std::optional<double>(params.softmax_scale), params.is_causal, params.window_size_left,
                     params.window_size_right, 0, static_cast<double>(params.softcap), false, c10::nullopt,
                     params.num_splits, params.pack_gqa, params.sm_margin);
}

static std::vector<at::Tensor> CallVllmV26Varlen(const MhaVarlenFwdParams &params) {
  auto max_seqlen_q_val = static_cast<int>(params.max_seqlen_q.value_or(0));
  auto max_seqlen_k_val = static_cast<int>(params.max_seqlen_k.value_or(0));
  return mha_varlen_fwd_vllm_flash_attn_v26(
      const_cast<at::Tensor &>(params.q), params.k, params.v, const_cast<c10::optional<at::Tensor> &>(params.out),
      params.seqlen_q, params.seqlen_k, const_cast<c10::optional<at::Tensor> &>(params.seqused_k),
      const_cast<c10::optional<at::Tensor> &>(params.block_table),
      const_cast<c10::optional<at::Tensor> &>(params.alibi_slopes), max_seqlen_q_val, max_seqlen_k_val,
      params.p_dropout, static_cast<float>(params.softmax_scale), params.zero_tensors, params.is_causal,
      params.window_size_left, params.window_size_right, params.softcap, params.return_softmax,
      const_cast<c10::optional<at::Generator> &>(params.gen));
}

static std::vector<at::Tensor> CallFA2V26Varlen(const MhaVarlenFwdParams &params) {
  auto max_seqlen_q_val = static_cast<int>(params.max_seqlen_q.value_or(0));
  auto max_seqlen_k_val = static_cast<int>(params.max_seqlen_k.value_or(0));
  auto &leftpad_k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.leftpad_k));
  return mha_varlen_fwd_flash_attn_v26(
      const_cast<at::Tensor &>(params.q), params.k, params.v, const_cast<c10::optional<at::Tensor> &>(params.out),
      params.seqlen_q, params.seqlen_k, const_cast<c10::optional<at::Tensor> &>(params.seqused_k), leftpad_k_ref,
      const_cast<c10::optional<at::Tensor> &>(params.block_table),
      const_cast<c10::optional<at::Tensor> &>(params.alibi_slopes), max_seqlen_q_val, max_seqlen_k_val,
      params.p_dropout, static_cast<float>(params.softmax_scale), params.zero_tensors, params.is_causal,
      params.window_size_left, params.window_size_right, params.softcap, params.return_softmax,
      const_cast<c10::optional<at::Generator> &>(params.gen));
}

static std::vector<at::Tensor> CallFA2V25Varlen(const MhaVarlenFwdParams &params) {
  auto max_seqlen_q_val = static_cast<int>(params.max_seqlen_q.value_or(0));
  auto max_seqlen_k_val = static_cast<int>(params.max_seqlen_k.value_or(0));
  return mha_varlen_fwd_flash_attn_v25(
      const_cast<at::Tensor &>(params.q), params.k, params.v, const_cast<c10::optional<at::Tensor> &>(params.out),
      params.seqlen_q, params.seqlen_k, const_cast<c10::optional<at::Tensor> &>(params.seqused_k),
      const_cast<c10::optional<at::Tensor> &>(params.alibi_slopes), max_seqlen_q_val, max_seqlen_k_val,
      params.p_dropout, static_cast<float>(params.softmax_scale), params.zero_tensors, params.is_causal,
      params.window_size_left, params.window_size_right, params.return_softmax,
      const_cast<c10::optional<at::Generator> &>(params.gen));
}

static void CallFA3Kvcache(MhaFwdKVCacheParams &params) {
  if (!ksana_llm::FlashAttentionBackend::mha_fwd_fa3_) {
    KLLM_THROW("FlashAttention 3 is not available but is forced by configuration.");
  }
  if (params.alibi_slopes.has_value()) {
    KLLM_THROW("Flash Attention 3 does not support alibi_slopes.");
  }
  mha_fwd_fa3(params.q, params.k_cache, params.v_cache, params.k, params.v, params.qv, params.out, params.seqlen_q,
              c10::nullopt, params.cu_seqlens_k_new, c10::nullopt, params.seqlens_k, params.max_seqlen_q, c10::nullopt,
              params.block_table, params.cache_batch_idx, params.leftpad_k, params.rotary_cos, params.rotary_sin,
              params.rotary_seqlens, params.q_descale, params.k_descale, params.v_descale, params.softmax_scale,
              params.is_causal, params.window_size_left, params.window_size_right, 0, params.softcap,
              params.rotary_interleaved, params.scheduler_metadata, params.num_splits, params.pack_gqa,
              params.sm_margin);
}

static void CallVllmV26Kvcache(MhaFwdKVCacheParams &params) {
  if (!HasVllmV26()) {
    KLLM_THROW("vLLM FlashAttention 2.6 is not available but is forced by configuration.");
  }
  auto &k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.k));
  auto &v_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.v));
  auto &seqlens_k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.seqlens_k));
  auto &rotary_cos_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_cos));
  auto &rotary_sin_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_sin));
  auto &cache_batch_idx_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.cache_batch_idx));

  mha_fwd_kvcache_vllm_flash_attn_v26(params.q, params.k_cache, params.v_cache, k_ref, v_ref, seqlens_k_ref,
                                      rotary_cos_ref, rotary_sin_ref, cache_batch_idx_ref, params.block_table,
                                      params.alibi_slopes, params.out, static_cast<float>(params.softmax_scale),
                                      params.is_causal, params.window_size_left, params.window_size_right,
                                      params.softcap, params.rotary_interleaved, params.num_splits);
}

static void CallFA2V26Kvcache(MhaFwdKVCacheParams &params) {
  if (!HasFA2V26()) {
    KLLM_THROW("FlashAttention 2.6 is not available but is forced by configuration.");
  }
  auto &k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.k));
  auto &v_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.v));
  auto &seqlens_k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.seqlens_k));
  auto &rotary_cos_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_cos));
  auto &rotary_sin_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_sin));
  auto &cache_batch_idx_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.cache_batch_idx));
  auto &leftpad_k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.leftpad_k));

  mha_fwd_kvcache_flash_attn_v26(params.q, params.k_cache, params.v_cache, k_ref, v_ref, seqlens_k_ref, rotary_cos_ref,
                                 rotary_sin_ref, cache_batch_idx_ref, leftpad_k_ref, params.block_table,
                                 params.alibi_slopes, params.out, static_cast<float>(params.softmax_scale),
                                 params.is_causal, params.window_size_left, params.window_size_right, params.softcap,
                                 params.rotary_interleaved, params.num_splits);
}

static void CallFA2V25Kvcache(MhaFwdKVCacheParams &params) {
  if (!HasFA2V25()) {
    KLLM_THROW("FlashAttention 2.5 is not available but is forced by configuration.");
  }
  auto &k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.k));
  auto &v_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.v));
  auto &seqlens_k_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.seqlens_k));
  auto &rotary_cos_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_cos));
  auto &rotary_sin_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.rotary_sin));
  auto &cache_batch_idx_ref = const_cast<std::optional<const at::Tensor> &>(
      reinterpret_cast<const std::optional<const at::Tensor> &>(params.cache_batch_idx));

  mha_fwd_kvcache_flash_attn_v25(params.q, params.k_cache, params.v_cache, k_ref, v_ref, seqlens_k_ref, rotary_cos_ref,
                                 rotary_sin_ref, cache_batch_idx_ref, params.block_table, params.alibi_slopes,
                                 params.out, static_cast<float>(params.softmax_scale), params.is_causal,
                                 params.window_size_left, params.window_size_right, params.rotary_interleaved,
                                 params.num_splits);
}

// FA3 implementation - compatible with FA2 return type
std::vector<at::Tensor> mha_fwd_fa3(
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
    std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
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
    int64_t num_splits, std::optional<bool> pack_gqa_, int64_t sm_margin) {
  // Call FA3 backend function and convert tuple to vector for compatibility
  auto result_tuple = ksana_llm::FlashAttentionBackend::mha_fwd_fa3_(
      q, k, v, k_new_, v_new_, q_v_, out_, cu_seqlens_q_, cu_seqlens_k_, cu_seqlens_k_new_, seqused_q_, seqused_k_,
      max_seqlen_q_, max_seqlen_k_, page_table_, kv_batch_idx_, leftpad_k_, rotary_cos_, rotary_sin_, seqlens_rotary_,
      q_descale_, k_descale_, v_descale_, softmax_scale_, is_causal, window_size_left, window_size_right,
      attention_chunk, softcap, is_rotary_interleaved, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
  std::vector<at::Tensor> result_vector;
  result_vector.push_back(std::get<0>(result_tuple));
  result_vector.push_back(std::get<1>(result_tuple));
  result_vector.push_back(std::get<2>(result_tuple));
  result_vector.push_back(std::get<3>(result_tuple));
  return result_vector;
}

std::vector<at::Tensor> mha_varlen_fwd_vllm_flash_attn_v26(
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
  KLLM_LOG_DEBUG << "VLLM FlashAttention Varlen V26 is used";
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_vllm_flash_attn_v26_(
      q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, block_table_, alibi_slopes_, max_seqlen_q, max_seqlen_k,
      p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, softcap, return_softmax,
      gen_);
}

std::vector<at::Tensor> mha_fwd_kvcache_vllm_flash_attn_v26(
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
  KLLM_LOG_DEBUG << "VLLM FlashAttention KVcache V26 is used";
  return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_vllm_flash_attn_v26_(
      q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, block_table_, alibi_slopes_,
      out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap, is_rotary_interleaved, num_splits);
}

std::vector<at::Tensor> mha_varlen_fwd_flash_attn_v25(
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
  KLLM_LOG_DEBUG << "FlashAttention Varlen V25 is used";
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v25_(
      q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, alibi_slopes_, max_seqlen_q, max_seqlen_k, p_dropout,
      softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, return_softmax, gen_);
}

std::vector<at::Tensor> mha_varlen_fwd_flash_attn_v26(
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
  KLLM_LOG_DEBUG << "FlashAttention Varlen V26 is used";
  return ksana_llm::FlashAttentionBackend::mha_varlen_fwd_flash_attn_v26_(
      q, k, v, out_, cu_seqlens_q, cu_seqlens_k, seqused_k, leftpad_k_, block_table_, alibi_slopes_, max_seqlen_q,
      max_seqlen_k, p_dropout, softmax_scale, zero_tensors, is_causal, window_size_left, window_size_right, softcap,
      return_softmax, gen_);
}

std::vector<at::Tensor> mha_fwd_kvcache_flash_attn_v25(
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
  KLLM_LOG_DEBUG << "FlashAttention KVcache V25 is used";
  return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v25_(
      q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, block_table_, alibi_slopes_,
      out_, softmax_scale, is_causal, window_size_left, window_size_right, is_rotary_interleaved, num_splits);
}

std::vector<at::Tensor> mha_fwd_kvcache_flash_attn_v26(
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
  KLLM_LOG_DEBUG << "FlashAttention KVcache V26 is used";
  return ksana_llm::FlashAttentionBackend::mha_fwd_kvcache_flash_attn_v26_(
      q, kcache, vcache, k_, v_, seqlens_k_, rotary_cos_, rotary_sin_, cache_batch_idx_, leftpad_k_, block_table_,
      alibi_slopes_, out_, softmax_scale, is_causal, window_size_left, window_size_right, softcap,
      is_rotary_interleaved, num_splits);
}

// Unified interface functions that handle version selection
std::vector<at::Tensor> InvokeMhaVarlenFwd(MhaVarlenFwdParams &params) {
  const bool is_rope = !params.alibi_slopes.has_value();
  const bool need_blocked = params.block_table.has_value();
  auto choice = GetFlashAttnImplChoice();
  // User-forced selection (non-auto)
  if (choice != AttnBackendConfig::FlashAttnImplChoice::AUTO) {
    switch (choice) {
      case AttnBackendConfig::FlashAttnImplChoice::FA3: {
        if (!HasFA3()) {
          KLLM_THROW("FlashAttention 3 is not available but is forced by configuration.");
        }
        return CallFA3Varlen(params);
      }
      case AttnBackendConfig::FlashAttnImplChoice::VLLM_V26: {
        if (!HasVllmV26()) {
          KLLM_THROW("vLLM FlashAttention 2.6 is not available but is forced by configuration.");
        }
        return CallVllmV26Varlen(params);
      }
      case AttnBackendConfig::FlashAttnImplChoice::FA2_V26: {
        if (!HasFA2V26()) {
          KLLM_THROW("FlashAttention 2.6 is not available but is forced by configuration.");
        }
        return CallFA2V26Varlen(params);
      }
      case AttnBackendConfig::FlashAttnImplChoice::FA2_V25: {
        if (!HasFA2V25()) {
          KLLM_THROW("FlashAttention 2.5 is not available but is forced by configuration.");
        }
        return CallFA2V25Varlen(params);
      }
      default:
        break;
    }
  }
  // AUTO
  static int cached_sm = -1;
  if (cached_sm < 0) {
    ksana_llm::FlashAttentionBackend backend;
    cached_sm = backend.GetCudaComputeCapability();
  }
  // 1) SM >= 90 && rope && !need_blocked => prefer FA3
  // TODO(qiannanzhou): fa3是支持block table的方式计算attention的，但是现在没有地方用到，先关闭。
  if (cached_sm >= 90 && is_rope && !need_blocked) {
    if (!HasFA3()) {
      KLLM_THROW("FlashAttention 3 is not available but is forced by configuration.");
    }
    return CallFA3Varlen(params);
  }
  // 2) SM in [80, 90)
  if (cached_sm >= 80) {
    if (HasVllmV26()) return CallVllmV26Varlen(params);
    if (HasFA2V26()) return CallFA2V26Varlen(params);
    if (HasFA2V25() && !need_blocked) return CallFA2V25Varlen(params);
  }
  KLLM_THROW("No suitable mha_varlen_fwd function loaded.");
}

void InvokeMhaFwdKvcCache(MhaFwdKVCacheParams &params) {
  auto choice = GetFlashAttnImplChoice();
  // User-forced selection (non-auto)
  if (choice != AttnBackendConfig::FlashAttnImplChoice::AUTO) {
    switch (choice) {
      case AttnBackendConfig::FlashAttnImplChoice::FA3: {
        if (!HasFA3()) {
          KLLM_THROW("FlashAttention 3 is not available but is forced by configuration.");
        }
        if (params.alibi_slopes.has_value()) {
          KLLM_THROW("Flash Attention 3 does not support alibi_slopes.");
        }
        CallFA3Kvcache(params);
        return;
      }
      case AttnBackendConfig::FlashAttnImplChoice::VLLM_V26: {
        if (!HasVllmV26()) {
          KLLM_THROW("vLLM FlashAttention 2.6 is not available but is forced by configuration.");
        }
        CallVllmV26Kvcache(params);
        return;
      }
      case AttnBackendConfig::FlashAttnImplChoice::FA2_V26: {
        if (!HasFA2V26()) {
          KLLM_THROW("FlashAttention 2.6 is not available but is forced by configuration.");
        }
        CallFA2V26Kvcache(params);
        return;
      }
      case AttnBackendConfig::FlashAttnImplChoice::FA2_V25: {
        if (!HasFA2V25()) {
          KLLM_THROW("FlashAttention 2.5 is not available but is forced by configuration.");
        }
        CallFA2V25Kvcache(params);
        return;
      }
      default:
        break;
    }
  }

  // AUTO
  static int cached_sm = -1;
  if (cached_sm < 0) {
    ksana_llm::FlashAttentionBackend backend;
    cached_sm = backend.GetCudaComputeCapability();
  }
  const bool is_rope = !params.alibi_slopes.has_value();
  const bool need_blocked = params.block_table.has_value();

  // 1) SM >= 90 且使用 RoPE，优先 FA3
  if (cached_sm >= 90 && is_rope && !need_blocked) {
    if (!HasFA3()) {
      KLLM_THROW("FlashAttention 3 is not available.");
    }
    if (params.alibi_slopes.has_value()) {
      KLLM_THROW("Flash Attention 3 does not support alibi_slopes.");
    }
    CallFA3Kvcache(params);
    return;
  }
  // 2) SM in [80, 90)
  if (cached_sm >= 80) {
    if (HasVllmV26()) {
      CallVllmV26Kvcache(params);
      return;
    }
    if (HasFA2V26()) {
      CallFA2V26Kvcache(params);
      return;
    }
    if (HasFA2V25() && !need_blocked) {
      CallFA2V25Kvcache(params);
      return;
    }
  }
  KLLM_THROW("No suitable mha_fwd_kvcache function loaded.");
}

}  // namespace ksana_llm
#endif
