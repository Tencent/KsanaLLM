/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/flash_mla_attention_layer.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"
#include "ksana_llm/runtime/layer_progress_tracker.h"
#include "ksana_llm/utils/string_utils.h"

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"

#include "ksana_llm/utils/singleton.h"
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)
#  include "ksana_llm/kernels/nvidia/flash_attn_cpp_wrapper.h"
#else
#  include "flash_api.h"
#endif

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

extern bool kContextDecodeUseFP8Cache;

// Adapted from
// [DeepSeek-V3 Project] https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py#L393
template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaAttenVarlen(void* output_buffer, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,
                    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale,
                    void* v_head_weight_scale, void* gemm_workspace, cublasHandle_t& cublas_handles,
                    cublasLtHandle_t& cublaslt_handles, void* rotary_embedding_pos, void* rotary_embedding_mask,
                    void* mla_workspace, void* seqlens_with_prefix_ptr, float attn_scale,
                    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                    int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim,
                    int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int num_kv_heads, float k_scale,
                    float v_scale, bool is_causal, int rank, int block_size, void** k_list, void** v_list,
                    void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes,
                    cudaStream_t stream, void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr,
                    int64_t kv_cache_block_num, int max_blocks_per_seq, int max_forwarding_tokens, int total_prefix_len,
                    void* seqlens_without_prefix_ptr, QuantMode mm_quant_mode) {
  const int stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  const int head_size = qk_nope_head_dim + qk_rope_head_dim;

  const auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  const auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);

  static thread_local void* seqlen_kv_int32 = nullptr;
  static thread_local void* seqlen_q_int32 = nullptr;
  if (seqlen_kv_int32 == nullptr) {
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    cudaMalloc(&seqlen_kv_int32, batch_scheduler_config.max_batch_size * sizeof(int));
    cudaMalloc(&seqlen_q_int32, batch_scheduler_config.max_batch_size * sizeof(int));
  }
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_with_prefix_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlen_kv_int32), stream);
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_without_prefix_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlen_q_int32), stream);

  // req total sequence length (with prefix)
  const torch::Tensor seqlen_kv_tensor = torch::from_blob(seqlen_kv_int32, {batch + 1}, int32_options);
  // req input_ids length (without prefix)
  const torch::Tensor seqlen_q_tensor = torch::from_blob(seqlen_q_int32, {batch + 1}, int32_options);

  // kv_b_proj升维矩阵乘
  // 1. kv_nope_proj
  if (kv_b_nope_weight_scale != nullptr) {
    if (gemm_workspace == nullptr) {
      KLLM_THROW("Quantized matmul has not gemm_workspace");
    }
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = gemm_workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);
      float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, output_buffer, total_tokens, kv_lora_rank,
                                num_heads * qk_nope_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_tokens, num_heads * qk_nope_head_dim,
                        kv_lora_rank, compressed_kv_ptr, kv_b_nope_proj_weight, output_buffer,
                        GetMacheteDataType<SCALAR_T>(), llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
                        std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                            static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    // 得到k_nope [token_num, head, 128]
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(kv_b_nope_proj_weight), output_buffer, stream, nullptr,
                           nullptr);
  }
  // 2. v_head_proj
  if (v_head_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      void* a_q = gemm_workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      float* b_scale = static_cast<float*>(v_head_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, mla_workspace, total_tokens, kv_lora_rank,
                                num_heads * v_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                        compressed_kv_ptr, v_head_proj_weight, mla_workspace, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    // mla_workspace得到v [token_num, head, 128]
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), mla_workspace, stream, nullptr, nullptr);
  }

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }

  /*
  Input Parameters:
  q_nope : q_nope_ptr       [total_tokens, num_heads, qk_nope_head_dim]
  k_nope : output_buffer  [total_tokens, num_heads, qk_nope_head_dim]
  q_pe   : q_pe_ptr         [total_tokens, num_heads, qk_rope_head_dim]
  k_pe   : k_pe_ptr         [total_tokens, 1, qk_rope_head_dim]
  v_pe   : mla_workspace    [total_tokens, num_heads, v_head_dim]

  Intermediate Tensors:
  k_pe_expanded : k_pe_expanded_ptr  [total_tokens, num_heads, qk_rope_head_dim]
  v_pad         : v_pad_ptr          [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim - v_head_dim]
  q_tensor      : q_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
  k_tensor      : k_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
  v_tensor      : v_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  Memory Buffer Allocation:
  output_buffer : [k_nope][q_tensor][k_tensor]
  mla_workspace   : [v_pe][k_pe_expanded][v_pad][v_tensor]
  */

  const size_t qk_combined_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t v_tensor_size = total_tokens * num_heads * v_head_dim * sizeof(SCALAR_T);
  const size_t k_pe_expanded_size = total_tokens * num_heads * qk_rope_head_dim * sizeof(SCALAR_T);
  const size_t v_pad_size =
      total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim - v_head_dim) * sizeof(SCALAR_T);
  const size_t k_nope_size = total_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);

  const size_t q_tensor_offset = (k_nope_size + 1023) & ~(1023);
  const size_t k_tensor_offset = (q_tensor_offset + qk_combined_size + 1023) & ~(1023);
  const size_t k_pe_expanded_offset = (v_tensor_size + 1023) & ~(1023);
  const size_t v_pad_offset = (k_pe_expanded_offset + k_pe_expanded_size + 1023) & ~(1023);
  const size_t v_tensor_offset = (v_pad_offset + v_pad_size + 1023) & ~(1023);

  void* const q_tensor_ptr = static_cast<char*>(output_buffer) + q_tensor_offset;
  void* const k_tensor_ptr = static_cast<char*>(output_buffer) + k_tensor_offset;
  void* const k_pe_expanded_ptr = static_cast<char*>(mla_workspace) + k_pe_expanded_offset;
  void* const v_pad_ptr = static_cast<char*>(mla_workspace) + v_pad_offset;
  void* const v_tensor_ptr = static_cast<char*>(mla_workspace) + v_tensor_offset;

  torch::Tensor q_tensor =
      torch::from_blob(q_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor k_tensor =
      torch::from_blob(k_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor v_tensor =
      torch::from_blob(v_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);

  const size_t outer_dim_size = total_tokens * num_heads;
  const size_t inner_dim_size = 1;

  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   q_tensor.data_ptr(), stream);

  // cat(k_nope, k_pe)
  Expand<SCALAR_T>(k_pe_ptr, k_pe_expanded_ptr, total_tokens, num_heads, qk_rope_head_dim, 0, stream);
  Concat<SCALAR_T>(output_buffer, k_pe_expanded_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   k_tensor.data_ptr(), stream);

  // pad v
  CUDA_CHECK(cudaMemsetAsync(v_pad_ptr, 0, v_pad_size, stream));
  Concat<SCALAR_T>(mla_workspace, v_pad_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   v_tensor.data_ptr(), stream);

  // copy new key and value to cache block
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
      v_list, reinterpret_cast<size_t*>(seqlens_with_prefix_ptr), reinterpret_cast<size_t*>(prefix_offsets),
      reinterpret_cast<size_t*>(seqlens_without_prefix_ptr), reinterpret_cast<int*>(block_offsets), block_size, batch,
      total_tokens, num_kv_heads, head_size, stride_size, k_scale, v_scale, stream));

  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  const bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(mla_workspace, {total_tokens, num_heads, head_size}, options);
  }
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }

  std::vector<at::Tensor> mha_output;
  auto cache_options = options;
  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 || KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    KLLM_THROW("FlashAttention not support fp8 kv cache");
  }
  // kv_cache[num_blocks, block_size, num_kv_heads, head_size]
  torch::Tensor k_cache_tensor =
      torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  torch::Tensor v_cache_tensor =
      torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
  c10::optional<at::Tensor> block_table = torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);

  mha_output = mha_varlen_fwd(q_tensor, k_cache_tensor, v_cache_tensor, out_tensor, seqlen_q_tensor, seqlen_kv_tensor,
                              seqused_k, block_table, alibi_slopes_tensor, max_forwarding_tokens, max_tokens, 0.f,
                              attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);

  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, "
                      "set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    const size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(output_buffer, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  const size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  const size_t src_pitch = head_size * sizeof(SCALAR_T);
  // Tensor(MEMORY_DEVICE, TYPE_FP16, {total_tokens, num_heads, qk_nope_head_dim +
  // qk_rope_head_dim}
  CUDA_CHECK(cudaMemcpy2DAsync(output_buffer, dst_pitch, mla_workspace, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
}

#define MLA_ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                 \
  template void MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                          \
      void* output_buffer, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,                 \
      void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale, \
      void* gemm_workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,                       \
      void* rotary_embedding_pos, void* rotary_embedding_mask, void* mla_workspace, void* seqlens_with_prefix_ptr,    \
      float attn_scale, std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,     \
      int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim,         \
      int kv_lora_rank, int v_head_dim, int num_kv_heads, float k_scale, float v_scale, bool is_causal, int rank,     \
      int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,                        \
      const std::optional<void*>& alibi_slopes, cudaStream_t stream, void* k_cache_ptr, void* v_cache_ptr,            \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int max_forwarding_tokens,        \
      int total_prefix_len, void* seqlens_without_prefix_ptr, QuantMode mm_quant_mode)
MLA_ATTEN_VARLEN(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
MLA_ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef MLA_ATTEN_VARLEN

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
void MlaAttenVarlenAbsorb(
    void* output_buffer, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,
    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale,
    void* gemm_workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
    void* rotary_embedding_pos, void* rotary_embedding_mask, void* mla_workspace, void* seqlens_with_prefix_ptr,
    float attn_scale, std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
    int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim,
    int kv_lora_rank, int v_head_dim, int num_kv_heads, float k_scale, float v_scale, bool is_causal, int rank,
    int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
    const std::optional<void*>& alibi_slopes, cudaStream_t stream, void* k_cache_ptr, void* v_cache_ptr,
    int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int max_forwarding_tokens,
    int total_prefix_len, void* seqlens_without_prefix_ptr, void* prefix_k_buffer, void* prefix_v_buffer,
    void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer, QuantMode mm_quant_mode) {
  const int stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  const int head_size = qk_nope_head_dim + qk_rope_head_dim;

  const auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  const auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);

  static thread_local void* seqlen_kv_int32 = nullptr;
  static thread_local void* seqlen_q_int32 = nullptr;
  if (seqlen_kv_int32 == nullptr) {
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);
    cudaMalloc(&seqlen_kv_int32, batch_scheduler_config.max_batch_size * sizeof(int));
    cudaMalloc(&seqlen_q_int32, batch_scheduler_config.max_batch_size * sizeof(int));
  }
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_with_prefix_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlen_kv_int32), stream);
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_without_prefix_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlen_q_int32), stream);

  // req total sequence length (with prefix)
  const torch::Tensor seqlen_kv_tensor = torch::from_blob(seqlen_kv_int32, {batch + 1}, int32_options);
  // req input_ids length (without prefix)
  const torch::Tensor seqlen_q_tensor = torch::from_blob(seqlen_q_int32, {batch + 1}, int32_options);

  // kv_b_proj升维矩阵乘
  // 1. kv_nope_proj
  if (kv_b_nope_weight_scale != nullptr) {
    if (gemm_workspace == nullptr) {
      KLLM_THROW("Quantized matmul has not gemm_workspace");
    }
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      SCALAR_T* a = static_cast<SCALAR_T*>(compressed_kv_ptr);
      void* a_q = gemm_workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_tokens, kv_lora_rank, true, stream);
      float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, output_buffer, total_tokens, kv_lora_rank,
                                num_heads * qk_nope_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_tokens, num_heads * qk_nope_head_dim,
                        kv_lora_rank, compressed_kv_ptr, kv_b_nope_proj_weight, output_buffer,
                        GetMacheteDataType<SCALAR_T>(), llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
                        std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                            static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    // 得到k_nope [token_num, head, 128]
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * qk_nope_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(kv_b_nope_proj_weight), output_buffer, stream, nullptr,
                           nullptr);
  }
  // 2. v_head_proj
  if (v_head_weight_scale != nullptr) {
    if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
      void* a_q = gemm_workspace;
      float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_tokens * kv_lora_rank);
      float* b_scale = static_cast<float*>(v_head_weight_scale);
      InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, mla_workspace, total_tokens, kv_lora_rank,
                                num_heads * v_head_dim, stream);
    } else if (mm_quant_mode == QUANT_GPTQ) {
      int64_t workspace_size = 0;
      std::vector<std::string> machete_schedule_map =
          Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
      std::optional<std::string> best_schedule = std::nullopt;
      if (static_cast<size_t>(total_tokens) < machete_schedule_map.size()) {
        best_schedule = std::optional<std::string>(machete_schedule_map[total_tokens]);
      }
      InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                        compressed_kv_ptr, v_head_proj_weight, mla_workspace, GetMacheteDataType<SCALAR_T>(),
                        llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                        std::optional<std::vector<size_t>>(
                            {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                        GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
    }
  } else {
    // mla_workspace得到v [token_num, head, 128]
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), mla_workspace, stream, nullptr, nullptr);
  }

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
  }

  /*
  Input Parameters:
  q_nope : q_nope_ptr       [total_tokens, num_heads, qk_nope_head_dim]
  k_nope : output_buffer  [total_tokens, num_heads, qk_nope_head_dim]
  q_pe   : q_pe_ptr         [total_tokens, num_heads, qk_rope_head_dim]
  k_pe   : k_pe_ptr         [total_tokens, 1, qk_rope_head_dim]
  v_pe   : mla_workspace    [total_tokens, num_heads, v_head_dim]

  Intermediate Tensors:
  k_pe_expanded : k_pe_expanded_ptr  [total_tokens, num_heads, qk_rope_head_dim]
  v_pad         : v_pad_ptr          [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim - v_head_dim]
  q_tensor      : q_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
  k_tensor      : k_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]
  v_tensor      : v_tensor_ptr       [total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim]

  Memory Buffer Allocation:
  output_buffer : [k_nope][q_tensor][k_tensor]
  mla_workspace   : [v_pe][k_pe_expanded][v_pad][v_tensor]
  */

  const size_t qk_combined_size = total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t v_tensor_size = total_tokens * num_heads * v_head_dim * sizeof(SCALAR_T);
  const size_t k_pe_expanded_size = total_tokens * num_heads * qk_rope_head_dim * sizeof(SCALAR_T);
  const size_t v_pad_size =
      total_tokens * num_heads * (qk_nope_head_dim + qk_rope_head_dim - v_head_dim) * sizeof(SCALAR_T);
  const size_t k_nope_size = total_tokens * num_heads * qk_nope_head_dim * sizeof(SCALAR_T);

  const size_t q_tensor_offset = (k_nope_size + 1023) & ~(1023);
  const size_t k_tensor_offset = (q_tensor_offset + qk_combined_size + 1023) & ~(1023);
  const size_t k_pe_expanded_offset = (v_tensor_size + 1023) & ~(1023);
  const size_t v_pad_offset = (k_pe_expanded_offset + k_pe_expanded_size + 1023) & ~(1023);
  const size_t v_tensor_offset = (v_pad_offset + v_pad_size + 1023) & ~(1023);

  void* const q_tensor_ptr = static_cast<char*>(output_buffer) + q_tensor_offset;
  void* const k_tensor_ptr = static_cast<char*>(output_buffer) + k_tensor_offset;
  void* const k_pe_expanded_ptr = static_cast<char*>(mla_workspace) + k_pe_expanded_offset;
  void* const v_pad_ptr = static_cast<char*>(mla_workspace) + v_pad_offset;
  void* const v_tensor_ptr = static_cast<char*>(mla_workspace) + v_tensor_offset;

  torch::Tensor q_tensor =
      torch::from_blob(q_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor k_tensor =
      torch::from_blob(k_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);
  torch::Tensor v_tensor =
      torch::from_blob(v_tensor_ptr, {total_tokens, num_heads, qk_nope_head_dim + qk_rope_head_dim}, options);

  const size_t outer_dim_size = total_tokens * num_heads;
  const size_t inner_dim_size = 1;

  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   q_tensor.data_ptr(), stream);

  // cat(k_nope, k_pe)
  Expand<SCALAR_T>(k_pe_ptr, k_pe_expanded_ptr, total_tokens, num_heads, qk_rope_head_dim, 0, stream);
  Concat<SCALAR_T>(output_buffer, k_pe_expanded_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   k_tensor.data_ptr(), stream);

  // pad v
  CUDA_CHECK(cudaMemsetAsync(v_pad_ptr, 0, v_pad_size, stream));
  Concat<SCALAR_T>(mla_workspace, v_pad_ptr, qk_nope_head_dim, qk_rope_head_dim, outer_dim_size, inner_dim_size,
                   v_tensor.data_ptr(), stream);

  // copy new k&v to kv cache block
  // Use compressed kvcache, k is [num_token, qk_rope_head_dim], v is  [num_token, kv_lora_rank]
  const int k_stride_size = qk_rope_head_dim;
  const int v_stride_size = kv_lora_rank;
  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_pe_ptr), reinterpret_cast<SCALAR_T*>(compressed_kv_ptr), k_list, v_list,
      reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(seqlens_without_prefix_ptr),
      reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, k_stride_size, v_stride_size, k_scale,
      v_scale, stream));

  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  const bool seqlenq_ngroups_swapped =
      max_tokens == 1 && num_heads > num_kv_heads && head_size % 8 == 0 && !alibi_slopes.has_value();
  c10::optional<at::Tensor> out_tensor = c10::nullopt;
  if (!seqlenq_ngroups_swapped) {
    out_tensor = torch::from_blob(mla_workspace, {total_tokens, num_heads, head_size}, options);
  }
  at::Tensor q_tmp_tensor = torch::reshape(q_tensor, {total_tokens, num_heads, head_size});
  c10::optional<at::Tensor> seqused_k = c10::nullopt;
  auto float32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kFloat32);
  c10::optional<at::Tensor> alibi_slopes_tensor = c10::nullopt;
  if (alibi_slopes.has_value()) {
    alibi_slopes_tensor = torch::from_blob(alibi_slopes.value(), {num_heads}, float32_options);
  }
  // Enables kContextDecodeUseFP8Cache to simulate the effect of KV cache quantization on flash attention,
  // intended for use in testing accuracy outcomes only.
  if constexpr (KV_DTYPE != llm_kernels::utils::KVCacheType::kAuto) {
    if (kContextDecodeUseFP8Cache) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), k_tensor.size(0), k_tensor.size(1), stride_size, k_scale,
          stream));
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::ConvertFP8AndBack<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), v_tensor.size(0), v_tensor.size(1), stride_size, v_scale,
          stream));
    }
  }
  std::vector<at::Tensor> mha_output;
  if (total_prefix_len > 0) {
    KLLM_LOG_DEBUG << "Prefix caching triggered, unique len:" << total_tokens << ", prefix len:" << total_prefix_len;

    // copy compressed_kv_cache to prefix_kv_buffer, not include k_rope_cache
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyValueBlockToBuffer<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(prefix_kv_buffer), v_list, kv_lora_rank + qk_rope_head_dim, 0, kv_lora_rank,
        kv_lora_rank, reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<int*>(block_offsets), block_size,
        total_prefix_len, stream));

    // calc cached key to prefix_k_up_buffer
    if (kv_b_nope_weight_scale != nullptr) {
      if (gemm_workspace == nullptr) {
        KLLM_THROW("FP8 quantized matmul has not gemm_workspace");
      }
      if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
        SCALAR_T* a = static_cast<SCALAR_T*>(prefix_kv_buffer);
        void* a_q = gemm_workspace;
        float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_prefix_len * kv_lora_rank);
        InvokePerTokenGroupQuantFp8E4m3<SCALAR_T>(a, a_q, a_s, total_prefix_len, kv_lora_rank, true, stream);

        float* b_scale = static_cast<float*>(kv_b_nope_weight_scale);
        InvokeBlockGemm<SCALAR_T>(a_q, a_s, kv_b_nope_proj_weight, b_scale, prefix_k_up_buffer, total_prefix_len,
                                  kv_lora_rank, num_heads * qk_nope_head_dim, stream);
      } else if (mm_quant_mode == QUANT_GPTQ) {
        int64_t workspace_size = 0;
        std::vector<std::string> machete_schedule_map =
            Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * qk_nope_head_dim,
                                                                              kv_lora_rank);
        std::optional<std::string> best_schedule = std::nullopt;
        if (static_cast<size_t>(total_prefix_len) < machete_schedule_map.size()) {
          best_schedule = std::optional<std::string>(machete_schedule_map[total_prefix_len]);
        }
        InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_prefix_len, num_heads * qk_nope_head_dim,
                          kv_lora_rank, prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer,
                          GetMacheteDataType<SCALAR_T>(), llm_kernels::nvidia::vllm_dtype::kU4B8,
                          kv_b_nope_weight_scale,
                          std::optional<std::vector<size_t>>({static_cast<size_t>(kv_lora_rank / 128),
                                                              static_cast<size_t>(num_heads * qk_nope_head_dim)}),
                          GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
      } else {
        KLLM_THROW(fmt::format("MLA not support quant mode: {}", mm_quant_mode));
      }
    } else {
      InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * qk_nope_head_dim,
                             kv_lora_rank, prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer, stream, nullptr,
                             nullptr);
    }

    // calc cached value to prefix_v_up_buffer
    if (v_head_weight_scale != nullptr) {
      if (mm_quant_mode == QUANT_BLOCK_FP8_E4M3) {
        void* a_q = gemm_workspace;
        float* a_s = static_cast<float*>(a_q + sizeof(uint8_t) * total_prefix_len * kv_lora_rank);

        float* b_scale = static_cast<float*>(v_head_weight_scale);
        InvokeBlockGemm<SCALAR_T>(a_q, a_s, v_head_proj_weight, b_scale, prefix_v_up_buffer, total_prefix_len,
                                  kv_lora_rank, num_heads * v_head_dim, stream);
      } else if (mm_quant_mode == QUANT_GPTQ) {
        int64_t workspace_size = 0;
        std::vector<std::string> machete_schedule_map =
            Singleton<MacheteSearchStatus>::GetInstance()->GetMacheteSchedule(num_heads * v_head_dim, kv_lora_rank);
        std::optional<std::string> best_schedule = std::nullopt;
        if (static_cast<size_t>(total_prefix_len) < machete_schedule_map.size()) {
          best_schedule = std::optional<std::string>(machete_schedule_map[total_prefix_len]);
        }
        InvokeMacheteGemm(workspace_size, gemm_workspace, stream, total_prefix_len, num_heads * v_head_dim,
                          kv_lora_rank, prefix_kv_buffer, v_head_proj_weight, prefix_v_up_buffer,
                          GetMacheteDataType<SCALAR_T>(), llm_kernels::nvidia::vllm_dtype::kU4B8, v_head_weight_scale,
                          std::optional<std::vector<size_t>>(
                              {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * v_head_dim)}),
                          GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
      }
    } else {
      InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * v_head_dim, kv_lora_rank,
                             prefix_kv_buffer, v_head_proj_weight, prefix_v_up_buffer, stream, nullptr, nullptr);
    }

    // copy new key/value to prefix_k_buffer/prefix_v_buffer
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaExtendKVPrefixWithEmpty<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()),
        reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
        reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(seqlens_without_prefix_ptr), num_heads,
        qk_nope_head_dim + qk_rope_head_dim, total_tokens, stream));

    // copy cached k_rope to prefix_k_up_buffer each head
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyKeyBlockWithReplication<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(prefix_k_buffer), k_list, kv_lora_rank + qk_rope_head_dim, kv_lora_rank,
        qk_rope_head_dim, num_heads, qk_nope_head_dim + qk_rope_head_dim, qk_nope_head_dim,
        reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(seqlens_without_prefix_ptr),
        reinterpret_cast<int*>(block_offsets), block_size, total_prefix_len, stream));

    // copy and concat cached key/value from prefix_k_up_buffer/prefix_v_up_buffer to prefix_k_buffer/prefix_v_buffer
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFillKVPrefix<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
        reinterpret_cast<SCALAR_T*>(prefix_k_up_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_up_buffer),
        reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(seqlens_without_prefix_ptr), num_heads,
        qk_nope_head_dim, qk_nope_head_dim + qk_rope_head_dim, total_prefix_len, stream));

    const int total_len_with_prefix = total_tokens + total_prefix_len;
    torch::Tensor prefix_k_tensor =
        torch::from_blob(prefix_k_buffer, {total_len_with_prefix, num_heads, head_size}, options);
    torch::Tensor prefix_v_tensor =
        torch::from_blob(prefix_v_buffer, {total_len_with_prefix, num_heads, head_size}, options);
    c10::optional<at::Tensor> prefix_out_tensor =
        torch::from_blob(mla_workspace, {total_tokens, num_heads, head_size}, options);

    c10::optional<at::Tensor> block_table = c10::nullopt;
    mha_output = mha_varlen_fwd(q_tensor, prefix_k_tensor, prefix_v_tensor, prefix_out_tensor, seqlen_q_tensor,
                                seqlen_kv_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens, max_tokens,
                                0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  } else {
    KLLM_LOG_INFO << "start mha";
    c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq
    mha_output =
        mha_varlen_fwd(q_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                       torch::reshape(v_tensor, {total_tokens, num_kv_heads, head_size}), out_tensor, seqlen_q_tensor,
                       seqlen_kv_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens, max_tokens, 0.f,
                       attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  }
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, "
                      "set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(output_buffer, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  const size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  const size_t src_pitch = head_size * sizeof(SCALAR_T);
  // Tensor(MEMORY_DEVICE, TYPE_FP16, {total_tokens, num_heads, qk_nope_head_dim +
  // qk_rope_head_dim}
  CUDA_CHECK(cudaMemcpy2DAsync(output_buffer, dst_pitch, mla_workspace, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
}

#define MLA_ATTEN_VARLEN_ABSORB(SCALAR_T, CACHE_T, KV_DTYPE)                                                          \
  template void MlaAttenVarlenAbsorb<SCALAR_T, CACHE_T, KV_DTYPE>(                                                    \
      void* output_buffer, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,                 \
      void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale, \
      void* gemm_workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,                       \
      void* rotary_embedding_pos, void* rotary_embedding_mask, void* mla_workspace, void* seqlens_with_prefix_ptr,    \
      float attn_scale, std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,     \
      int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim,         \
      int kv_lora_rank, int v_head_dim, int num_kv_heads, float k_scale, float v_scale, bool is_causal, int rank,     \
      int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,                        \
      const std::optional<void*>& alibi_slopes, cudaStream_t stream, void* k_cache_ptr, void* v_cache_ptr,            \
      int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int max_forwarding_tokens,        \
      int total_prefix_len, void* seqlens_without_prefix_ptr, void* prefix_k_buffer, void* prefix_v_buffer,           \
      void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer, QuantMode mm_quant_mode)
MLA_ATTEN_VARLEN_ABSORB(float, float, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN_ABSORB(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(float, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN_ABSORB(half, half, llm_kernels::utils::KVCacheType::kAuto);
MLA_ATTEN_VARLEN_ABSORB(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(half, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, __nv_bfloat16, llm_kernels::utils::KVCacheType::kAuto);
#if defined(ENABLE_FP8)
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E4M3);
MLA_ATTEN_VARLEN_ABSORB(__nv_bfloat16, uint8_t, llm_kernels::utils::KVCacheType::kFp8E5M2);
#endif
#undef MLA_ATTEN_VARLEN_ABSORB

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                                 const RuntimeConfig& runtime_config,
                                                                 std::shared_ptr<Context> context, int rank) {
#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  KLLM_THROW("MLA Only support ENABLE_FLASH_ATTN_WITH_CACHE.");
#endif
#if defined(ENABLE_FLASH_ATTN_MINOR_4) || defined(ENABLE_FLASH_ATTN_MINOR_5)
  KLLM_THROW("MLA Only support ENABLE_FLASH_ATTN_MINOR_6 or ENABLE_FLASH_ATTN_MINOR_7");
#endif
#if !defined(ENABLE_FLASH_ATTN_2) && !defined(ENABLE_VLLM_FLASH_ATTN_2)
  KLLM_THROW("MLA Only support ENABLE_FLASH_ATTN_2 or ENABLE_VLLM_FLASH_ATTN_2");
#endif
#if !defined(ENABLE_VLLM_FLASH_ATTN_MINOR_6) && !defined(ENABLE_VLLM_FLASH_ATTN_MINOR_7)
  KLLM_THROW("MLA Only support ENABLE_VLLM_FLASH_ATTN_MINOR_6 or ENABLE_VLLM_FLASH_ATTN_MINOR_7");
#endif

  return AttentionLayer<SCALAR_T>::Init(parameters, runtime_config, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) {
  auto input_iter = input_tensors.cbegin();
  const Tensor& hidden_buffer = *input_iter++;
  const Tensor& dp_input_offset = *input_iter++;
  const Tensor& kv_list = *input_iter++;
  const Tensor& dp_input_prefix = *input_iter++;
  const Tensor& dp_prefill_q_offset = *input_iter++;
  const Tensor& kv_cache_offset = *input_iter++;
  const Tensor& rotary_embedding_pos = *input_iter++;
  const Tensor& rotary_embedding_mask = *input_iter++;
  const Tensor& dp_flexible_rotary_embedding_pos = *input_iter++;   // not supported
  const Tensor& dp_flexible_rotary_embedding_mask = *input_iter++;  // not supported
  const Tensor& dp_dst_flexible_kv_cache = *input_iter++;           // not supported
  const Tensor& dp_src_flexible_kv_cache = *input_iter++;           // not supported
  const Tensor& dp_dst_flexible_token_idx = *input_iter++;          // not supported
  const Tensor& dp_src_flexible_token_idx = *input_iter++;          // not supported
  const Tensor& dp_flexible_offset_uint64 = *input_iter++;          // not supported
  const Tensor& forward_shape = *input_iter++;
  const Tensor& layer_kv_cache = *input_iter++;
  const Tensor& block_table = *input_iter++;
  const Tensor& q_nope_buffer = *input_iter++;
  const Tensor& q_rope_buffer = *input_iter++;
  const Tensor& kv_buffer = *input_iter++;
  const Tensor& k_rope_buffer = *input_iter++;
  const Tensor& kv_b_nope_proj_weight = *input_iter++;
  const Tensor& v_head_proj_weight = *input_iter++;
  const Tensor& attn_o_proj_weight = *input_iter++;
  const Tensor& prefix_k_buffer = *input_iter++;
  const Tensor& prefix_v_buffer = *input_iter++;
  const Tensor& prefix_kv_buffer = *input_iter++;
  const Tensor& prefix_k_up_buffer = *input_iter++;
  const Tensor& prefix_v_up_buffer = *input_iter++;
  const Tensor& workspace_buffer = *input_iter++;

  Tensor& out = output_tensors[0];

  const int flexible_len = dp_dst_flexible_kv_cache.shape[0];
  KLLM_CHECK_WITH_INFO(flexible_len == 0, "Not support flexible length cache.");

  const int layer_block_num = forward_shape.shape[2];
  const int max_forwarding_tokens = forward_shape.shape[6];
  const int batch_size = forward_shape.shape[8];
  const int max_tokens = forward_shape.shape[9];
  const int total_prefix_len = forward_shape.shape[12];

  const int total_tokens = q_nope_buffer.shape[0];

  void** const k_list = (kv_list.GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** const v_list = k_list + layer_block_num;

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = kv_b_nope_proj_weight.weight_scales->GetPtr<void>();
    v_head_weight_scale = v_head_proj_weight.weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = kv_b_nope_proj_weight.scales->GetPtr<void>();
    v_head_weight_scale = v_head_proj_weight.scales->GetPtr<void>();
  }

  const int64_t kv_cache_block_num = *(layer_kv_cache.GetPtr<int64_t>());
  void** const layer_kv_cache_ptr = layer_kv_cache.GetPtr<void*>() + 1;
  void* const k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
  void* const v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  const int max_blocks_per_seq = block_table.shape[1];

  void* const fp8_work_buffer = this->workspace_buffer_ ? this->workspace_buffer_->template GetPtr<void>() : nullptr;

  if (IsAbsorbWeightsEnabled()) {
    MlaAttenVarlenAbsorb<SCALAR_T, CACHE_T, KV_DTYPE>(
        out.GetPtr<void>(), q_nope_buffer.GetPtr<void>(), q_rope_buffer.GetPtr<void>(), k_rope_buffer.GetPtr<void>(),
        kv_buffer.GetPtr<void>(), kv_b_nope_proj_weight.GetPtr<void>(), v_head_proj_weight.GetPtr<void>(),
        kv_b_nope_weight_scale, v_head_weight_scale, fp8_work_buffer,
        this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
        rotary_embedding_pos.GetPtr<void>(), rotary_embedding_mask.GetPtr<void>(), workspace_buffer.GetPtr<void>(),
        dp_input_offset.GetPtr<void>(), this->attn_scale_, this->rotary_embedding_cuda_, total_tokens, max_tokens,
        batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_, this->kv_lora_rank_,
        this->v_head_dim_, this->num_kv_heads_, this->k_scale_, this->v_scale_, this->is_causal_, this->rank_,
        this->block_token_num_, k_list, v_list, dp_input_prefix.GetPtr<void>(), kv_cache_offset.GetPtr<void>(),
        this->alibi_slopes_, this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr,
        block_table.GetPtr<int32_t>(), kv_cache_block_num, max_blocks_per_seq, max_forwarding_tokens, total_prefix_len,
        dp_prefill_q_offset.GetPtr<void>(), prefix_k_buffer.GetPtr<void>(), prefix_v_buffer.GetPtr<void>(),
        prefix_kv_buffer.GetPtr<void>(), prefix_k_up_buffer.GetPtr<void>(), prefix_v_up_buffer.GetPtr<void>(),
        this->mm_quant_mode_);
  } else {
    MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
        out.GetPtr<void>(), q_nope_buffer.GetPtr<void>(), q_rope_buffer.GetPtr<void>(), k_rope_buffer.GetPtr<void>(),
        kv_buffer.GetPtr<void>(), kv_b_nope_proj_weight.GetPtr<void>(), v_head_proj_weight.GetPtr<void>(),
        kv_b_nope_weight_scale, v_head_weight_scale, fp8_work_buffer,
        this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
        rotary_embedding_pos.GetPtr<void>(), rotary_embedding_mask.GetPtr<void>(), workspace_buffer.GetPtr<void>(),
        dp_input_offset.GetPtr<void>(), this->attn_scale_, this->rotary_embedding_cuda_, total_tokens, max_tokens,
        batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_, this->kv_lora_rank_,
        this->v_head_dim_, this->num_kv_heads_, this->k_scale_, this->v_scale_, this->is_causal_, this->rank_,
        this->block_token_num_, k_list, v_list, dp_input_prefix.GetPtr<void>(), kv_cache_offset.GetPtr<void>(),
        this->alibi_slopes_, this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr,
        block_table.GetPtr<int32_t>(), kv_cache_block_num, max_blocks_per_seq, max_forwarding_tokens, total_prefix_len,
        dp_prefill_q_offset.GetPtr<void>(), this->mm_quant_mode_);
  }

  // 通知 LayerProgressTracker 该层已完成，它会在内部记录 event 并在单独的线程中监控完成情况
  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(this->rank_, this->layer_index_,
                                                                      this->context_->GetComputeStreams()[this->rank_]);

  const size_t o_proj_dim =
      this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3 ? attn_o_proj_weight.shape[0] : attn_o_proj_weight.shape[1];
  out.shape = {total_tokens, o_proj_dim};
  out.dtype = hidden_buffer.dtype;
  return Status();
}

using llm_kernels::utils::KVCacheType;
template class FlashMlaAttentionLayer<float, float, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<float, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<half, half, KVCacheType::kAuto>;
template class FlashMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<half, uint8_t, KVCacheType::kFp8E5M2>;
template class FlashMlaAttentionLayer<__nv_bfloat16, __nv_bfloat16, KVCacheType::kAuto>;
#if defined(ENABLE_FP8)
template class FlashMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E4M3>;
template class FlashMlaAttentionLayer<__nv_bfloat16, uint8_t, KVCacheType::kFp8E5M2>;
#endif

}  // namespace ksana_llm
