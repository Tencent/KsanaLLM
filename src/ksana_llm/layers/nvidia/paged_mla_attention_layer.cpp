/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/paged_mla_attention_layer.h"

#include <thread>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#include "ksana_llm/utils/singleton.h"

#include "ksana_llm/kernels/nvidia/triton_wrapper.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "csrc/kernels/nvidia/add/add.h"
#include "csrc/kernels/nvidia/all_reduce/custom_all_reduce.h"
#include "csrc/kernels/nvidia/assemble_tokens_hidden/assemble_tokens_hidden.h"
#include "csrc/kernels/nvidia/blockwise_gemm/blockwise_gemm.h"
#include "csrc/kernels/nvidia/cast/cast.h"
#include "csrc/kernels/nvidia/concat/concat.h"
#include "csrc/kernels/nvidia/embedding/embedding.h"
#include "csrc/kernels/nvidia/expand/expand.h"
#include "csrc/kernels/nvidia/flash_mla/flash_mla.h"
#include "csrc/kernels/nvidia/fused_add_norm/fused_add_norm.h"
#include "csrc/kernels/nvidia/fused_moe_gptq_int4_fp8_kernel/dequant.h"
#include "csrc/kernels/nvidia/gemm_wrapper/gemm_wrapper.h"
#include "csrc/kernels/nvidia/grouped_topk/grouped_topk.h"
#include "csrc/kernels/nvidia/layernorm/layernorm.h"
#include "csrc/kernels/nvidia/moe/moe.h"
#include "csrc/kernels/nvidia/moe_wna16/moe_wna16.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/cache_copy_flash_attn_layout.h"
#include "csrc/kernels/nvidia/paged_attention/mla_cache_copy.h"
#include "csrc/kernels/nvidia/paged_attention/paged_attention.h"
#include "csrc/kernels/nvidia/per_token_group_quant/per_token_group_quant_8bit.h"
#include "csrc/kernels/nvidia/permute/permute.h"
#include "csrc/kernels/nvidia/samplers/greedy.h"
#include "csrc/utils/nvidia/cuda_fp8_utils.h"

#include "ksana_llm/kernels/argmax.h"
#include "ksana_llm/kernels/cast.h"

#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/search_status.h"

namespace ksana_llm {

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeMlaPagedAttention(
    void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale,
    size_t o_proj_dim, void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
    void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr, int max_context_len, cudaStream_t stream,
    void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,
    int v_head_dim, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale,
    int batch, void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, void* workspace_ptr,
    float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,
    const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,
    int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, QuantMode mm_quant_mode,
    int q_seq_len) {
  // 修改stride_size 和 head_size
  stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  head_size = qk_nope_head_dim + qk_rope_head_dim;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  // kv_b_proj升维矩阵乘
  // 1. kv_nope_proj
  if (kv_b_nope_weight_scale != nullptr) {
    if (workspace == nullptr) {
      KLLM_THROW("FP8 quantized matmul has not workspace");
    }
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);

      float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, qkv_workspace, total_tokens, kv_lora_rank,
                                num_heads * qk_nope_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                        compressed_kv_ptr, kv_b_nope_proj_weight, qkv_workspace, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
                        std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                            static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    } else {
      KLLM_THROW(fmt::format("MLA Decode not support quant mode: {}", mm_quant_mode));
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(kv_b_nope_proj_weight), qkv_workspace, stream, nullptr,
                           nullptr);
  }
  // 2. v_head_proj
  if (v_head_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      // TODO(winminkong): 两个矩阵乘的输入一致，应该可以只求一次input_scale, 后期验证后改成只求一次
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);
      float* b_scale = static_cast<float*>(v_head_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, hidden_buffer_1, total_tokens, kv_lora_rank,
                                num_heads * v_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, workspace, stream, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                        compressed_kv_ptr, v_head_proj_weight, hidden_buffer_1, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), hidden_buffer_1, stream, nullptr,
                           nullptr);
  }

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask), q_pe_ptr, k_pe_ptr, total_tokens,
                                    stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
  }
  /*
  Input Parameters:
    q_nope : q_nope_ptr       [total_tokens, num_heads, qk_nope_head_dim]
    k_nope : qkv_workspace  [total_tokens, num_heads, qk_nope_head_dim]
    q_pe   : q_pe_ptr         [total_tokens, num_heads, qk_rope_head_dim]
    k_pe   : k_pe_ptr         [total_tokens, 1, qk_rope_head_dim]
    v_pe   : hidden_buffer_1       [total_tokens, num_heads, v_head_dim]

  Intermediate Tensors:
    k_pe_expanded : k_pe_expanded_ptr  [total_tokens, num_heads, qk_rope_head_dim]
    v_pad         : v_pad_ptr          [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim - v_head_dim]
    q_tensor      : q_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    k_tensor      : k_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
    v_tensor      : v_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  Memory Buffer Allocation:
    qkv_workspace   : [k_nope][q_tensor][k_tensor]
    out             : [v_pe][k_pe_expanded][v_pad][v_tensor]
  */
  size_t qk_combined_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(SCALAR_T);
  size_t v_tensor_size = total_tokens * num_heads * v_head_dim * sizeof(SCALAR_T);
  size_t k_pe_expanded_size = total_tokens * num_heads * qk_rope_head_dim * sizeof(SCALAR_T);
  size_t v_pad_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim - v_head_dim) * sizeof(SCALAR_T);
  size_t k_nope_size = total_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);

  size_t q_tensor_offset = (k_nope_size + 1023) & ~(1023);
  size_t k_tensor_offset = (q_tensor_offset + qk_combined_size + 1023) & ~(1023);
  size_t k_pe_expanded_offset = (v_tensor_size + 1023) & ~(1023);
  size_t v_pad_offset = (k_pe_expanded_offset + k_pe_expanded_size + 1023) & ~(1023);
  size_t v_tensor_offset = (v_pad_offset + v_pad_size + 1023) & ~(1023);

  void* q_tensor_ptr = static_cast<char*>(qkv_workspace) + q_tensor_offset;
  void* k_tensor_ptr = static_cast<char*>(qkv_workspace) + k_tensor_offset;
  void* k_pe_expanded_ptr = static_cast<char*>(hidden_buffer_1) + k_pe_expanded_offset;
  void* v_pad_ptr = static_cast<char*>(hidden_buffer_1) + v_pad_offset;
  void* v_tensor_ptr = static_cast<char*>(hidden_buffer_1) + v_tensor_offset;

  const size_t outer_dim_size = total_tokens * num_heads;
  const size_t inner_dim_size = 1;

  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   q_tensor_ptr, stream);

  // cat(k_nope, k_pe)
  Expand<SCALAR_T>(k_pe_ptr, k_pe_expanded_ptr, total_tokens, num_heads, qk_rope_head_dim, 0, stream);
  Concat<SCALAR_T>(qkv_workspace, k_pe_expanded_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   k_tensor_ptr, stream);

  // pad v
  CUDA_CHECK(cudaMemsetAsync(v_pad_ptr, 0, v_pad_size, stream));
  Concat<SCALAR_T>(hidden_buffer_1, v_pad_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   v_tensor_ptr, stream);

#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, q_seq_len, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
#else
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(v_tensor_ptr), key_cache_ptrs,
      value_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, total_tokens, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));
#endif
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  auto cache_options = options;
  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 || KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    // cache_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt8);
    KLLM_THROW("FlashAttention not support fp8 kv cache");
  }
  // kv_cache[num_blocks, 2, block_size, num_kv_heads, head_size]
  torch::Tensor k_cache_tensor =
      torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  torch::Tensor v_cache_tensor =
      torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
  c10::optional<at::Tensor> block_table_tensor =
      torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);
  c10::optional<const at::Tensor> seqlens_k_tensor =
      c10::optional<const at::Tensor>(torch::from_blob(context_lens_ptr, {batch}, int32_options));
  torch::Tensor q_tensor =
      torch::from_blob(q_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  q_tensor = q_tensor.reshape({batch, 1, num_heads, head_size});
  c10::optional<at::Tensor> out_tensor = torch::from_blob(hidden_buffer_1, {batch, 1, num_heads, head_size}, options);
  float softmax_scale = attn_scale;
  c10::optional<at::Tensor> null_tensor = c10::nullopt;
  c10::optional<const at::Tensor> const_null_tensor = c10::nullopt;
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }

  {
    ksana_llm::MhaFwdKVCacheParams fa_params;
    fa_params.q = q_tensor;              // [batch, 1, num_heads, head_size]
    fa_params.k_cache = k_cache_tensor;  // [num_blocks, block_size, num_kv_heads, head_size]
    fa_params.v_cache = v_cache_tensor;
    fa_params.seqlens_k = seqlens_k_tensor;      // [batch]
    fa_params.block_table = block_table_tensor;  // [batch, max_blocks_per_seq]
    fa_params.alibi_slopes = alibi_slopes_tensor;
    fa_params.out = out_tensor;  // [batch, 1, num_heads, head_size]
    fa_params.softmax_scale = softmax_scale;
    fa_params.is_causal = true;
    fa_params.window_size_left = -1;
    fa_params.window_size_right = -1;
    fa_params.softcap = 0.0f;
    fa_params.rotary_interleaved = true;
    fa_params.num_splits = 0;
    ksana_llm::InvokeMhaFwdKvcCache(fa_params);
  }
#else
  const float* alibi_slopes_ptr =
      reinterpret_cast<const float*>(alibi_slopes.has_value() ? alibi_slopes.value() : nullptr);
  // 可能会有softmax_scale的问题（也不调用）
  PagedAttentionOp<SCALAR_T, CACHE_T, KV_DTYPE>(num_heads, head_size, num_kv_heads, stride_size, block_size, k_scale,
                                                v_scale, hidden_buffer_1, q_tensor_ptr, key_cache_ptrs,
                                                value_cache_ptrs, cache_offsets_ptr, context_lens_ptr, max_context_len,
                                                seqs_num, stream, workspace_ptr, work_size, alibi_slopes_ptr);
#endif
  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  size_t src_pitch = head_size * sizeof(SCALAR_T);
  CUDA_CHECK(cudaMemcpy2DAsync(output_ptr, dst_pitch, hidden_buffer_1, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
}

#define RUN_MLA_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                         \
  template void InvokeMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                                \
      void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr,            \
      void* k_pe_ptr, void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale,           \
      void* v_head_weight_scale, size_t o_proj_dim, void* workspace, cublasHandle_t& cublas_handles,                 \
      cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr,    \
      int max_context_len, cudaStream_t stream, void* cache_offsets_ptr, int seqs_num, int num_heads,                \
      int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int head_size, int num_kv_heads, \
      int stride_size, int block_size, float k_scale, float v_scale, int batch, void* rotary_embedding_pos,          \
      void* rotary_embedding_mask, int total_tokens, float attn_scale,                                               \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, void* workspace_ptr,           \
      float layernorm_eps, bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, size_t work_size, int rank,   \
      const std::optional<void*>& alibi_slopes, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr,           \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, QuantMode mm_quant_mode,         \
      int q_seq_len)
RUN_MLA_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_MLA_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_MLA_PAGED_ATTENTION

// Adapted from
// [DeepSeek-V3 Project]
// https://github.com/vllm-project/vllm/blob/ed6e9075d31e32c8548b480a47d1ffb77da1f54c/vllm/attention/backends/triton_mla.py#L698
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void InvokeAbsorbMlaPagedAttention(
    void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* w_uv_weight, size_t w_uv_o_dim, void* workspace, cublasHandle_t& cublas_handles,
    cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void* context_lens_ptr, cudaStream_t stream,
    void* cache_offsets_ptr, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,
    int v_head_dim, int num_kv_heads, int block_size, float k_scale, float v_scale, int batch,
    void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda, void* tile_scheduler_metadata_ptr,
    void* num_splits_ptr, int rank, void* qkv_workspace, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr,
    int64_t kv_cache_block_num, int max_blocks_per_seq, int q_seq_len, int tail_offset_token) {
  // 修改stride_size 和 head_size
  const int stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  const int head_size = qk_nope_head_dim + qk_rope_head_dim;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(reinterpret_cast<int64_t*>(rotary_embedding_pos),
                                    reinterpret_cast<int64_t*>(rotary_embedding_mask), q_pe_ptr, k_pe_ptr, total_tokens,
                                    stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward<SCALAR_T>());
  }
  /*
  Input Parameters:
  v_tensor : compressed_kv_ptr  [total_tokens, kv_lora_rank]
  q_nope   : q_nope_ptr         [total_tokens, num_heads, kv_lora_rank]
  q_pe     : q_pe_ptr           [total_tokens, num_heads, qk_rope_head_dim]
  k_pe     : k_pe_ptr           [total_tokens, qk_rope_head_dim]

  Intermediate Tensors:
  q_tensor      : q_tensor_ptr       [total_tokens, num_heads, kv_lora_rank + qk_rope_head_dim]
  k_tensor      : k_tensor_ptr       [total_tokens, kv_lora_rank + qk_rope_head_dim]
  o_states      : output_ptr    [total_tokens, num_heads, kv_lora_rank]

  Memory Buffer Allocation(Take the maximum value):
  output_ptr : [o_states][q_tensor][k_tensor]
  */
  size_t output_head_size = std::max(kv_lora_rank + qk_rope_head_dim, v_head_dim);
  const size_t origin_size = (tail_offset_token + total_tokens) * num_heads * output_head_size * sizeof(SCALAR_T);
  const size_t q_tensor_offset = (origin_size + 1023) & ~(1023);
  const size_t q_tensor_size = total_tokens * num_heads * (kv_lora_rank + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t k_tensor_offset = (q_tensor_offset + q_tensor_size + 1023) & ~(1023);

  void* q_tensor_ptr = output_ptr + q_tensor_offset;
  void* const k_tensor_ptr = output_ptr + k_tensor_offset;

  const size_t outer_q_dim_size = total_tokens * num_heads;
  const size_t outer_k_dim_size = total_tokens;
  constexpr size_t kInnerDimSize = 1;

  const AbsorbWeightsType absorb_type = GetAbsorbWeightsType();
  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_q_dim_size, kInnerDimSize, q_tensor_ptr,
                   stream);

  // cat(v, k_pe)
  Concat<SCALAR_T>(compressed_kv_ptr, k_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_k_dim_size, kInnerDimSize,
                   k_tensor_ptr, stream);

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), key_cache_ptrs,
      key_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, q_seq_len, 1, kv_lora_rank + qk_rope_head_dim, kv_lora_rank + qk_rope_head_dim, k_scale, v_scale, stream));
  auto cache_options = options;

  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2) {
    KLLM_THROW("Flash MLA not support fp8_e5m2 KV Cache. Please use fp8_e4m3.");
  }

  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    KLLM_LOG_DEBUG << "FP8 kv cache and flash mla enabled, using FP8 inference, quantizing q tensor.";
    // hidden_buffer_1: leave storage for o [batch_size, q_seq_per_hk, num_heads, head_size_v]
    const int num_heads_q = num_heads;
    constexpr int kNumHeadsK = 1;
    const int head_size_v = kv_lora_rank;  // 512
    const int num_q_heads_per_hk = num_heads_q / kNumHeadsK;
    const int q_seq_per_hk = q_seq_len * num_q_heads_per_hk;

    const size_t o_tensor_size = batch * q_seq_per_hk * kNumHeadsK * head_size_v * sizeof(SCALAR_T);
    const size_t quant_q_tensor_offset = (o_tensor_size + 1023) & ~(1023);
    void* const quant_q_tensor_ptr = hidden_buffer_1 + quant_q_tensor_offset;

    float q_scale = k_scale;
    llm_kernels::nvidia::ConvertQToCacheType<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(q_tensor_ptr), reinterpret_cast<CACHE_T*>(quant_q_tensor_ptr), batch, q_seq_len,
        num_heads, kv_lora_rank + qk_rope_head_dim, num_heads * (kv_lora_rank + qk_rope_head_dim), q_scale, stream);
    q_tensor_ptr = quant_q_tensor_ptr;
  }

  // Absorb has two versions
  if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
    // Flash mla accepts CACHE_T type input. If KV_DTYPE is auto, SCALAR_T equals CACHE_T.
    // If KV_DTYPE is e4m3, flash mla calculates at fp8 precision and outputs at bf16 precision.
    llm_kernels::nvidia::InvokeFlashMla<SCALAR_T, CACHE_T, KV_DTYPE>(
        static_cast<CACHE_T*>(q_tensor_ptr), static_cast<CACHE_T*>(k_cache_ptr), q_seq_len, attn_scale, block_table_ptr,
        context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/, hidden_buffer_1,
        batch, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, k_scale, v_scale, max_blocks_per_seq, rank,
        kv_cache_block_num, stream);
    // tp8: num_heads:16, total_tokens:256,kv_lora_rank:512,qk_rope_head_dim:64,w_uv_o_dim:7168,v_head_dim:128
    // [256, 16, 512] => [16, 256, 512]
    InvokePermute<SCALAR_T>(hidden_buffer_1, qkv_workspace, {total_tokens, num_heads, kv_lora_rank}, {1, 0, 2}, stream);
    // [16, 256, 512] * [16, 512, 128] => [16, 256, 128]
    InvokeBatchedMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, num_heads, total_tokens, v_head_dim, kv_lora_rank,
                                  qkv_workspace, w_uv_weight, hidden_buffer_1, stream, nullptr, 0, nullptr);
    // [16, 256, 128] => [256, 16, 128];
    InvokePermute<SCALAR_T>(hidden_buffer_1, output_ptr, {num_heads, total_tokens, v_head_dim}, {1, 0, 2}, stream);
  } else {
    llm_kernels::nvidia::InvokeFlashMla<SCALAR_T, CACHE_T, KV_DTYPE>(
        static_cast<CACHE_T*>(q_tensor_ptr), static_cast<CACHE_T*>(k_cache_ptr), q_seq_len, attn_scale, block_table_ptr,
        context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/, output_ptr, batch,
        num_heads, kv_lora_rank, qk_rope_head_dim, block_size, k_scale, v_scale, max_blocks_per_seq, rank,
        kv_cache_block_num, stream);
  }
}

#define RUN_ABSORB_MLA_PAGED_ATTENTION(SCALAR_T, CACHE_T, KV_DTYPE)                                                   \
  template void InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(                                           \
      void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr,             \
      void* k_pe_ptr, void* w_uv_weight, size_t w_uv_o_dim, void* workspace, cublasHandle_t& cublas_handles,          \
      cublasLtHandle_t& cublaslt_handles, void** key_cache_ptrs, void* context_lens_ptr, cudaStream_t stream,         \
      void* cache_offsets_ptr, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,           \
      int v_head_dim, int num_kv_heads, int block_size, float k_scale, float v_scale, int batch,                      \
      void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,                    \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda>& rotary_embedding_cuda,                                 \
      void* tile_scheduler_metadata_ptr, void* num_splits_ptr, int rank, void* qkv_workspace, void* k_cache_ptr,      \
      void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int q_seq_len, \
      int tail_offset_token)
RUN_ABSORB_MLA_PAGED_ATTENTION(float, float, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, half, llm_kernels::utils::KVCacheType::kAuto);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
RUN_ABSORB_MLA_PAGED_ATTENTION(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef RUN_ABSORB_MLA_PAGED_ATTENTION

void SetMlaMetadataKernelAttribute(const int max_batch_size, cudaStream_t stream) {
  llm_kernels::nvidia::SetFlashMlaAttribute(max_batch_size, stream);
}

Status PagedMlaAttentionLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                    std::shared_ptr<Context> context, int rank) {
  AttentionLayer::Init(parameters, runtime_config, context, rank);

  // index 25 is max_batch_size in PagedMlaAttention, disgusting code, be careful.
  const size_t max_batch_size = std::any_cast<const size_t>(parameters[25]);
  SetMlaMetadataKernelAttribute(max_batch_size, context->GetComputeStreams()[rank].Get());

  return Status();
}

Status PagedMlaAttentionLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_DTYPE_AND_KVTYPE(inter_data_type_, kv_cache_dtype_, ForwardT, input_tensors, output_tensors);
}

/*
kv_list  [layers_num * (total_blocks * 2)]
|              layer1               |
| bs1 |     bs2   | bs1 |     bs2   |
|k|k|k|k|k|k|k|k|k|v|v|v|v|v|v|v|v|v|
每个k,v代表一个指针,存储的数据个数为一个block块能存的token个数
需要在model中将block按kv分开存储指针，方便后续计算
*/
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status PagedMlaAttentionLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  const Tensor& hidden_buffer1 = *input_iter++;
  const Tensor& kv_seq_len = *input_iter++;  // kv seq len (len of forwarding_tokens)
  const Tensor& kv_list = *input_iter++;
  const Tensor& cache_offset = *input_iter++;
  const Tensor& rotary_embedding_pos = *input_iter++;
  const Tensor& rotary_embedding_mask = *input_iter++;
  const Tensor& workspace = *input_iter++;
  const Tensor& forward_shape = *input_iter++;
  const Tensor& qkv_workspace = *input_iter++;
  const Tensor& query_norm_weight = *input_iter++;  // not supported, just an empty tensor
  const Tensor& key_norm_weight = *input_iter++;    // not supported, just an empty tensor
  const Tensor& layer_kv_cache = *input_iter++;
  const Tensor& block_table = *input_iter++;
  const Tensor& q_nope_tensor = *input_iter++;
  const Tensor& q_pe_tensor = *input_iter++;
  const Tensor& compressed_kv_tensor = *input_iter++;
  const Tensor& k_pe_tensor = *input_iter++;
  const Tensor& kv_b_nope_proj_weight = *input_iter++;
  const Tensor& v_head_proj_weight = *input_iter++;
  const Tensor& o_proj_weight = *input_iter++;
  const Tensor& tile_scheduler_metadata_tensor = *input_iter++;
  const Tensor& num_splits_tensor = *input_iter++;
  const Tensor& metadata = *input_iter++;
  const Tensor& w_uv_weight = *input_iter++;

  Tensor& out = output_tensors[0];

  const size_t skip_tokens_num = metadata.GetPtr<size_t>()[0];  // 在input_ids中的offset
  const size_t q_seq_len = metadata.GetPtr<size_t>()[1];

  const size_t batch_size = kv_seq_len.shape[0];
  const size_t total_tokens = batch_size * q_seq_len;

  void** const k_list_base = kv_list.GetPtr<void*>();
  const size_t layer_block_num = kv_list.shape[1] * 0.5;  // shape: [layer_num_on_node, total_block_num * 2]
  void** const k_list = k_list_base + static_cast<size_t>(this->layer_index_ * layer_block_num * 2);
  void** const v_list = k_list + layer_block_num;
  const int64_t kv_cache_block_num = *(layer_kv_cache.GetPtr<int64_t>());
  void** const layer_kv_cache_ptr = layer_kv_cache.GetPtr<void*>() + 1;
  void* const k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];  // block中每层layer的起始地址
  void* const v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* const block_table_ptr =
      block_table.GetPtr<int32_t>();                    // block id，加上layer_kv_cache_ptr后就是对应的cache block
  const int max_blocks_per_seq = block_table.shape[1];  // shape: [bs, max_num_blocks_per_query]
  // for mla
  auto skipped_q_pe_ptr =
      q_pe_tensor.GetPtr<void>() + skip_tokens_num * (q_pe_tensor.GetTotalBytes() / q_pe_tensor.shape[0]);
  auto skipped_compressed_kv_ptr =
      compressed_kv_tensor.GetPtr<void>() +
      skip_tokens_num * (compressed_kv_tensor.GetTotalBytes() / compressed_kv_tensor.shape[0]);

  auto skipped_k_pe_ptr =
      k_pe_tensor.GetPtr<void>() + skip_tokens_num * (k_pe_tensor.GetTotalBytes() / k_pe_tensor.shape[0]);

  const size_t batch_input_ids_len = q_pe_tensor.shape[0];
  const size_t batch_tail_tokens = batch_input_ids_len - skip_tokens_num - total_tokens;

  void* const kv_b_nope_proj_weight_ptr = kv_b_nope_proj_weight.GetPtr<void>();
  void* const v_head_proj_weight_ptr = v_head_proj_weight.GetPtr<void>();

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = input_tensors[17].weight_scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[18].weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = input_tensors[17].scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[18].scales->GetPtr<void>();
  }

  const size_t o_proj_dim =
      this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3 ? o_proj_weight.shape[0] : o_proj_weight.shape[1];

  auto skipped_hidden_buffer1_ptr =
      hidden_buffer1.GetPtr<void>() + skip_tokens_num * (hidden_buffer1.GetTotalBytes() / hidden_buffer1.shape[0]);
  const size_t o_proj_k_dim = this->v_head_dim_ * this->num_heads_;
  auto skipped_output_ptr = out.GetPtr<void>() + skip_tokens_num * o_proj_k_dim * out.GetDTypeSize();

  std::shared_ptr<Tensor>& work_buffer = this->workspace_buffer_;
  void* fp8_work_buffer = work_buffer == nullptr ? nullptr : work_buffer->GetPtr<void>();

  void* const w_uv_weight_ptr = w_uv_weight.GetPtr<void>();

  if (w_uv_weight_ptr) {  // Absorb
    InvokeAbsorbMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
        skipped_hidden_buffer1_ptr, skipped_output_ptr, q_nope_tensor.GetPtr<void>(), skipped_q_pe_ptr,
        skipped_compressed_kv_ptr, skipped_k_pe_ptr, w_uv_weight_ptr, o_proj_dim, fp8_work_buffer,
        this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
        k_list, kv_seq_len.GetPtr<void>(), this->context_->GetComputeStreams()[this->rank_].Get(),
        cache_offset.GetPtr<void>(), this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_,
        this->kv_lora_rank_, this->v_head_dim_, this->num_kv_heads_, this->block_token_num_, this->k_scale_,
        this->v_scale_, batch_size, rotary_embedding_pos.GetPtr<void>(), rotary_embedding_mask.GetPtr<void>(),
        total_tokens, this->attn_scale_, this->rotary_embedding_cuda_, tile_scheduler_metadata_tensor.GetPtr<void>(),
        num_splits_tensor.GetPtr<void>(), this->rank_, qkv_workspace.GetPtr<void>(), k_cache_ptr, v_cache_ptr,
        block_table_ptr, kv_cache_block_num, max_blocks_per_seq, q_seq_len, batch_tail_tokens);
    return Status();
  }

  const size_t max_tokens = forward_shape.shape[11];  // dp_single_token_request_max_tokens
  InvokeMlaPagedAttention<SCALAR_T, CACHE_T, KV_DTYPE>(
      skipped_hidden_buffer1_ptr, skipped_output_ptr, q_nope_tensor.GetPtr<void>(), skipped_q_pe_ptr,
      skipped_compressed_kv_ptr, skipped_k_pe_ptr, kv_b_nope_proj_weight_ptr, v_head_proj_weight_ptr,
      kv_b_nope_weight_scale, v_head_weight_scale, o_proj_dim, fp8_work_buffer,
      this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
      k_list, v_list, kv_seq_len.GetPtr<void>(), max_tokens, this->context_->GetComputeStreams()[this->rank_].Get(),
      cache_offset.GetPtr<void>(), batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_,
      this->kv_lora_rank_, this->v_head_dim_, this->head_size_, this->num_kv_heads_, this->stride_size_,
      this->block_token_num_, this->k_scale_, this->v_scale_, batch_size, rotary_embedding_pos.GetPtr<void>(),
      rotary_embedding_mask.GetPtr<void>(), total_tokens, this->attn_scale_, this->rotary_embedding_cuda_,
      workspace.GetPtr<void>(), this->layernorm_eps_, this->use_qk_norm_, query_norm_weight.GetPtr<void>(),
      key_norm_weight.GetPtr<void>(), workspace.GetTotalBytes(), this->rank_, this->alibi_slopes_,
      qkv_workspace.GetPtr<void>(), k_cache_ptr, v_cache_ptr, block_table_ptr, kv_cache_block_num, max_blocks_per_seq,
      this->mm_quant_mode_, q_seq_len);

  return Status();
}

}  // namespace ksana_llm
