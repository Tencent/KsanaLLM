/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/kernels/nvidia/mla_kernel_wrapper.h"

#include <fstream>
#include <iostream>

#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

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
                    void* v_head_weight_scale, size_t o_proj_dim, void* gemm_workspace, cublasHandle_t& cublas_handles,
                    cublasLtHandle_t& cublaslt_handles, void* rotary_embedding_pos, void* rotary_embedding_mask,
                    void* mla_workspace, void* seqlen, float attn_scale,
                    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
                    int total_tokens, int max_tokens, int batch, int num_heads, int qk_rope_head_dim,
                    int qk_nope_head_dim, int kv_lora_rank, int v_head_dim, int num_kv_heads, int head_size,
                    int stride_size, float k_scale, float v_scale, size_t tensor_para_size, bool is_causal, int rank,
                    int block_size, void** k_list, void** v_list, void* prefix_offsets, void* block_offsets,
                    const std::optional<void*>& alibi_slopes, int layer_index, void* flexible_rotary_embedding_pos_ptr,
                    void* flexible_rotary_embedding_mask_ptr, void* dst_flexible_kv_cache_ptr,
                    void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr, void* src_flexible_token_idx_ptr,
                    void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps, bool use_qk_norm,
                    void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream, void* k_cache_ptr,
                    void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq,
                    size_t* without_prefix_offsets, int max_forwarding_tokens, int total_prefix_len,
                    void* seqlens_q_ptr, void* prefix_k_buffer, void* prefix_v_buffer, void* prefix_o_buffer,
                    void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer,
                    QuantMode mm_quant_mode) {
  // 修改stride_size 和 head_size
  stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  head_size = qk_nope_head_dim + qk_rope_head_dim;

  // k & v have different size in MLA condition.
  int k_stride_size = qk_rope_head_dim;
  int v_stride_size = kv_lora_rank;

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());
  auto int_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt64);
  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(torch::kInt32);

  static thread_local void* seqlen_int32 = nullptr;
  static thread_local void* seqlens_q_ptr_int32 = nullptr;
  static thread_local void* without_prefix_offsets_int32 = nullptr;
  if (seqlen_int32 == nullptr) {
    BatchSchedulerConfig batch_scheduler_config;
    Singleton<Environment>::GetInstance()->GetBatchSchedulerConfig(batch_scheduler_config);

    cudaMallocAsync(&seqlen_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
    cudaMallocAsync(&seqlens_q_ptr_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
    cudaMallocAsync(&without_prefix_offsets_int32, batch_scheduler_config.max_batch_size * sizeof(int), stream);
  }
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlen), batch + 1,
                                  reinterpret_cast<int*>(seqlen_int32), stream);
  llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(seqlens_q_ptr), batch + 1,
                                  reinterpret_cast<int*>(seqlens_q_ptr_int32), stream);

  torch::Tensor seqlen_tensor = torch::from_blob(seqlen_int32, {batch + 1}, int32_options);
  torch::Tensor prefill_seq_q_tensor = torch::from_blob(seqlens_q_ptr_int32, {batch + 1}, int32_options);

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
    InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_tokens, num_heads * v_head_dim, kv_lora_rank,
                           reinterpret_cast<const void*>(compressed_kv_ptr),
                           reinterpret_cast<const void*>(v_head_proj_weight), mla_workspace, stream, nullptr, nullptr);
  }

  if (flexible_len != 0) {
    // 暂时先不支持灵活cache，先不做处理
    KLLM_THROW("Not support flexible length cache.");
  }
#ifndef ENABLE_FLASH_ATTN_WITH_CACHE
  // MLA使用vllm_flash_attn,暂时先不做处理
  KLLM_LOG_INFO << "Only support flash attn with cache, skip";
#endif
  if (rotary_embedding_cuda.has_value()) {
    if (flexible_len != 0) {
      KLLM_THROW("Not support flexible length cache.");
    }

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

  void* q_tensor_ptr = static_cast<char*>(output_buffer) + q_tensor_offset;
  void* k_tensor_ptr = static_cast<char*>(output_buffer) + k_tensor_offset;
  void* k_pe_expanded_ptr = static_cast<char*>(mla_workspace) + k_pe_expanded_offset;
  void* v_pad_ptr = static_cast<char*>(mla_workspace) + v_pad_offset;
  void* v_tensor_ptr = static_cast<char*>(mla_workspace) + v_tensor_offset;

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

  if (use_cache) {
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
    // Use compressed kvcache, k is [num_token, qk_rope_head_dim], v is  [num_token, kv_lora_rank]
    if (IsAbsorbWeightsEnabled()) {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaKVCacheCopy<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_pe_ptr), reinterpret_cast<SCALAR_T*>(compressed_kv_ptr), k_list, v_list,
          reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets, reinterpret_cast<int*>(block_offsets),
          block_size, batch, total_tokens, k_stride_size, v_stride_size, k_scale, v_scale, stream));
    } else {
      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CacheCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()), k_list,
          v_list, reinterpret_cast<size_t*>(seqlen), reinterpret_cast<size_t*>(prefix_offsets), without_prefix_offsets,
          reinterpret_cast<int*>(block_offsets), block_size, batch, total_tokens, num_kv_heads, head_size, stride_size,
          k_scale, v_scale, stream));
    }
#else
    KLLM_THROW("Only support ENABLE_FLASH_ATTN_WITH_CACHE");
#endif
  }

// flash attention 2 or flash attention 1
#if defined(ENABLE_FLASH_ATTN_2) || defined(ENABLE_VLLM_FLASH_ATTN_2)

  // refer to github Dao-AILab/flash-attention csrc/flash_attn/flash_api.cpp#L374
  // When the flag is set to True and the output is not nullptr, calling the function mha_varlen_fwd
  // leads to a core dump.
  bool seqlenq_ngroups_swapped =
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
#  if defined(ENABLE_VLLM_FLASH_ATTN_MINOR_6) || defined(ENABLE_VLLM_FLASH_ATTN_MINOR_7)
#    ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  if (IsAbsorbWeightsEnabled()) {
    if (total_prefix_len > 0) {
      KLLM_LOG_DEBUG << "Prefix caching triggered, unique len:" << total_tokens << ", prefix len:" << total_prefix_len;

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyValueBlockToBuffer<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_kv_buffer), v_list, kv_lora_rank + qk_rope_head_dim, 0, kv_lora_rank,
          kv_lora_rank, reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<int*>(block_offsets), block_size,
          total_prefix_len, stream));

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
          InvokeMacheteGemm(
              workspace_size, gemm_workspace, stream, total_prefix_len, num_heads * qk_nope_head_dim, kv_lora_rank,
              prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer, GetMacheteDataType<SCALAR_T>(),
              llm_kernels::nvidia::vllm_dtype::kU4B8, kv_b_nope_weight_scale,
              std::optional<std::vector<size_t>>(
                  {static_cast<size_t>(kv_lora_rank / 128), static_cast<size_t>(num_heads * qk_nope_head_dim)}),
              GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128, best_schedule);
        } else {
          KLLM_THROW(fmt::format("MLA not support quant mode: {}", mm_quant_mode));
        }
      } else {
        InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * qk_nope_head_dim,
                               kv_lora_rank, prefix_kv_buffer, kv_b_nope_proj_weight, prefix_k_up_buffer, stream,
                               nullptr, nullptr);
      }
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
                            GetMacheteDataType<SCALAR_T>(), std::nullopt, std::nullopt, std::nullopt, 128,
                            best_schedule);
        }
      } else {
        InvokeMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, total_prefix_len, num_heads * v_head_dim, kv_lora_rank,
                               prefix_kv_buffer, v_head_proj_weight, prefix_v_up_buffer, stream, nullptr, nullptr);
      }

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaExtendKVPrefixWithEmpty<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(k_tensor.data_ptr()), reinterpret_cast<SCALAR_T*>(v_tensor.data_ptr()),
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets), num_heads,
          qk_nope_head_dim + qk_rope_head_dim, total_tokens, stream));

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaCopyKeyBlockWithReplication<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), k_list, kv_lora_rank + qk_rope_head_dim, kv_lora_rank,
          qk_rope_head_dim, num_heads, qk_nope_head_dim + qk_rope_head_dim, qk_nope_head_dim,
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets),
          reinterpret_cast<int*>(block_offsets), block_size, total_prefix_len, stream));

      CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::MlaFillKVPrefix<SCALAR_T, CACHE_T, KV_DTYPE>(
          reinterpret_cast<SCALAR_T*>(prefix_k_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_buffer),
          reinterpret_cast<SCALAR_T*>(prefix_k_up_buffer), reinterpret_cast<SCALAR_T*>(prefix_v_up_buffer),
          reinterpret_cast<size_t*>(prefix_offsets), reinterpret_cast<size_t*>(without_prefix_offsets), num_heads,
          qk_nope_head_dim, qk_nope_head_dim + qk_rope_head_dim, total_prefix_len, stream));

      const int total_len_with_prefix = total_tokens + total_prefix_len;
      torch::Tensor prefix_k_tensor =
          torch::from_blob(prefix_k_buffer, {total_len_with_prefix, num_heads, head_size}, options);
      torch::Tensor prefix_v_tensor =
          torch::from_blob(prefix_v_buffer, {total_len_with_prefix, num_heads, head_size}, options);
      c10::optional<at::Tensor> prefix_out_tensor =
          torch::from_blob(mla_workspace, {total_tokens, num_heads, head_size}, options);

      c10::optional<at::Tensor> block_table = c10::nullopt;
      mha_output = mha_varlen_fwd(q_tensor, prefix_k_tensor, prefix_v_tensor, prefix_out_tensor, prefill_seq_q_tensor,
                                  seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens, max_tokens,
                                  0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
    } else {
      c10::optional<at::Tensor> block_table = c10::nullopt;  // batch_size x max_num_blocks_per_seq
      mha_output = mha_varlen_fwd(q_tmp_tensor, torch::reshape(k_tensor, {total_tokens, num_kv_heads, head_size}),
                                  torch::reshape(v_tensor, {total_tokens, num_kv_heads, head_size}), out_tensor,
                                  seqlen_tensor, seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_tokens,
                                  max_tokens, 0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
    }
  } else {
    llm_kernels::nvidia::Int64ToInt(reinterpret_cast<const int64_t*>(without_prefix_offsets), batch + 1,
                                    reinterpret_cast<int*>(without_prefix_offsets_int32), stream);
    torch::Tensor seqlen_q_tensor = torch::from_blob(without_prefix_offsets_int32, {batch + 1}, int32_options);

    auto cache_options = options;
    if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 ||
        KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
      KLLM_THROW("FlashAttention not support fp8 kv cache");
    }
    // kv_cache[num_blocks, block_size, num_kv_heads, head_size]
    torch::Tensor k_cache_tensor =
        torch::from_blob(k_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    torch::Tensor v_cache_tensor =
        torch::from_blob(v_cache_ptr, {kv_cache_block_num, block_size, num_kv_heads, head_size}, cache_options);
    auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<int32_t>());
    c10::optional<at::Tensor> block_table =
        torch::from_blob(block_table_ptr, {batch, max_blocks_per_seq}, int32_options);

    mha_output = mha_varlen_fwd(q_tmp_tensor, k_cache_tensor, v_cache_tensor, out_tensor, seqlen_q_tensor,
                                seqlen_tensor, seqused_k, block_table, alibi_slopes_tensor, max_forwarding_tokens,
                                max_tokens, 0.f, attn_scale, false, is_causal, -1, -1, 0.f, false, c10::nullopt);
  }
#    else
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_WITH_CACHE");
#    endif
#  endif

#  if defined(ENABLE_FLASH_ATTN_MINOR_4) || defined(ENABLE_FLASH_ATTN_MINOR_5)
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_MINOR_6 or ENABLE_FLASH_ATTN_MINOR_7");
#  endif
  if (seqlenq_ngroups_swapped) {
    KLLM_LOG_DEBUG << "To prevent a core dump when seqlenq_ngroups_swapped is True, "
                      "set the output tensor to nullptr.";
    at::Tensor& out_data = mha_output[0];
    size_t total_size = out_data.numel() * out_data.element_size();
    CUDA_CHECK(cudaMemcpyAsync(output_buffer, out_data.data_ptr(), total_size, cudaMemcpyDeviceToDevice, stream));
  }

#else
  KLLM_THROW("Only support ENABLE_FLASH_ATTN_2 or ENABLE_VLLM_FLASH_ATTN_2");
#endif

  //  当 v_tensor 被 pad 时调用, 取out_tensor 的 v_head_dim 大小
  size_t dst_pitch = v_head_dim * sizeof(SCALAR_T);
  size_t src_pitch = head_size * sizeof(SCALAR_T);
  // Tensor(MEMORY_DEVICE, TYPE_FP16, {total_tokens, num_heads, qk_nope_head_dim +
  // qk_rope_head_dim}
  CUDA_CHECK(cudaMemcpy2DAsync(output_buffer, dst_pitch, mla_workspace, src_pitch, dst_pitch, total_tokens * num_heads,
                               cudaMemcpyDeviceToDevice, stream));
}

#define MLA_ATTEN_VARLEN(SCALAR_T, CACHE_T, KV_DTYPE)                                                                 \
  template void MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(                                                          \
      void* output_buffer, void* q_nope_ptr, void* q_pe_ptr, void* k_pe_ptr, void* compressed_kv_ptr,                 \
      void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale, \
      size_t o_proj_dim, void* gemm_workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,    \
      void* rotary_embedding_pos, void* rotary_embedding_mask, void* mla_workspace, void* seqlen, float attn_scale,   \
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, int total_tokens,     \
      int max_tokens, int batch, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,         \
      int v_head_dim, int num_kv_heads, int head_size, int stride_size, float k_scale, float v_scale,                 \
      size_t tensor_para_size, bool is_causal, int rank, int block_size, void** k_list, void** v_list,                \
      void* prefix_offsets, void* block_offsets, const std::optional<void*>& alibi_slopes, int layer_index,           \
      void* flexible_rotary_embedding_pos_ptr, void* flexible_rotary_embedding_mask_ptr,                              \
      void* dst_flexible_kv_cache_ptr, void* src_flexible_kv_cache_ptr, void* dst_flexible_token_idx_ptr,             \
      void* src_flexible_token_idx_ptr, void* flexible_offset_uint64_ptr, int flexible_len, float layernorm_eps,      \
      bool use_qk_norm, void* q_norm_weight, void* k_norm_weight, bool use_cache, cudaStream_t stream,                \
      void* k_cache_ptr, void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num,                     \
      int max_blocks_per_seq, size_t* without_prefix_offsets, int max_forwarding_tokens, int total_prefix_len,        \
      void* seqlens_q_ptr, void* prefix_k_buffer, void* prefix_v_buffer, void* prefix_o_buffer,                       \
      void* prefix_kv_buffer, void* prefix_k_up_buffer, void* prefix_v_up_buffer, QuantMode mm_quant_mode)
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
void InvokeMlaPagedAttention(
    void* hidden_buffer_1, void* output_ptr, void* q_nope_ptr, void* q_pe_ptr, void* compressed_kv_ptr, void* k_pe_ptr,
    void* kv_b_nope_proj_weight, void* v_head_proj_weight, void* kv_b_nope_weight_scale, void* v_head_weight_scale,
    size_t o_proj_dim, void* workspace, cublasHandle_t& cublas_handles, cublasLtHandle_t& cublaslt_handles,
    void** key_cache_ptrs, void** value_cache_ptrs, void* context_lens_ptr, int max_context_len, cudaStream_t stream,
    void* cache_offsets_ptr, int seqs_num, int num_heads, int qk_rope_head_dim, int qk_nope_head_dim, int kv_lora_rank,
    int v_head_dim, int head_size, int num_kv_heads, int stride_size, int block_size, float k_scale, float v_scale,
    int batch, void* rotary_embedding_pos, void* rotary_embedding_mask, int total_tokens, float attn_scale,
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr,
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
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
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

  mha_fwd_kvcache(q_tensor,             // batch_size x seqlen_q x num_heads x head_size
                  k_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                  v_cache_tensor,       // num_blocks x page_block_size x num_heads_k x head_size.
                  const_null_tensor,    // k_
                  const_null_tensor,    // v_
                  seqlens_k_tensor,     // batch_size
                  const_null_tensor,    // rotary_cos_: seqlen_ro x (rotary_dim / 2)
                  const_null_tensor,    // rotary_sin_: seqlen_ro x (rotary_dim / 2)
                  const_null_tensor,    // cache_batch_idx_: indices to index into the KV cache
                  block_table_tensor,   // batch_size x max_num_blocks_per_seq
                  alibi_slopes_tensor,  // num_heads or batch_size x num_heads
                  out_tensor,           // batch_size x seqlen_q x num_heads x head_size
                  softmax_scale, true, -1, -1, 0.0, true, 0);
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
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda, void* workspace_ptr, \
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
    std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,
    void* tile_scheduler_metadata_ptr, void* num_splits_ptr, int rank, void* qkv_workspace, void* k_cache_ptr,
    void* v_cache_ptr, int32_t* block_table_ptr, int64_t kv_cache_block_num, int max_blocks_per_seq, int q_seq_len,
    int tail_offset_token) {
  // 修改stride_size 和 head_size
  const int stride_size = num_heads * (qk_nope_head_dim + qk_rope_head_dim);
  const int head_size = qk_nope_head_dim + qk_rope_head_dim;
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchDataType<SCALAR_T>());

  if (rotary_embedding_cuda.has_value()) {
    rotary_embedding_cuda->SetInput(
        reinterpret_cast<int64_t*>(rotary_embedding_pos), reinterpret_cast<int64_t*>(rotary_embedding_mask),
        reinterpret_cast<SCALAR_T*>(q_pe_ptr), reinterpret_cast<SCALAR_T*>(k_pe_ptr), total_tokens, stream);
    CUDA_CHECK_LAST_ERROR(rotary_embedding_cuda->Forward());
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
  const size_t q_tensor_size = (total_tokens)*num_heads * (kv_lora_rank + qk_rope_head_dim) * sizeof(SCALAR_T);
  const size_t k_tensor_offset = (q_tensor_offset + q_tensor_size + 1023) & ~(1023);

  void* const q_tensor_ptr = output_ptr + q_tensor_offset;
  void* const k_tensor_ptr = output_ptr + k_tensor_offset;

  const size_t outer_q_dim_size = total_tokens * num_heads;
  const size_t outer_k_dim_size = total_tokens;
  const size_t inner_dim_size = 1;

  const AbsorbWeightsType absorb_type = GetAbsorbWeightsType();
  // cat(q_nope, q_pe)
  Concat<SCALAR_T>(q_nope_ptr, q_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_q_dim_size, inner_dim_size, q_tensor_ptr,
                   stream);

  // cat(v, k_pe)
  Concat<SCALAR_T>(compressed_kv_ptr, k_pe_ptr, kv_lora_rank, qk_rope_head_dim, outer_k_dim_size, inner_dim_size,
                   k_tensor_ptr, stream);

  CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::CachePosCopyFlashAttnLayout<SCALAR_T, CACHE_T, KV_DTYPE>(
      reinterpret_cast<SCALAR_T*>(k_tensor_ptr), reinterpret_cast<SCALAR_T*>(k_tensor_ptr), key_cache_ptrs,
      key_cache_ptrs, reinterpret_cast<int*>(context_lens_ptr), reinterpret_cast<int*>(cache_offsets_ptr), block_size,
      batch, q_seq_len, 1, kv_lora_rank + qk_rope_head_dim, kv_lora_rank + qk_rope_head_dim, k_scale, v_scale, stream));
  auto cache_options = options;
  if (KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E5M2 || KV_DTYPE == llm_kernels::utils::KVCacheType::kFp8E4M3) {
    int32_t* dst_block_table_ptr = block_table_ptr + batch * max_blocks_per_seq;
    llm_kernels::nvidia::ConvertToScalar<SCALAR_T, CACHE_T, KV_DTYPE>(
        reinterpret_cast<CACHE_T*>(k_cache_ptr), reinterpret_cast<SCALAR_T*>(v_cache_ptr), block_table_ptr,
        dst_block_table_ptr, batch * max_blocks_per_seq, block_size * (kv_lora_rank + qk_rope_head_dim), k_scale,
        v_scale, stream);
    k_cache_ptr = v_cache_ptr;
    block_table_ptr = dst_block_table_ptr;
    KLLM_LOG_DEBUG << "ConvertToScalar num:" << batch * max_blocks_per_seq;
  }

  static bool enable_flash_mla = Singleton<Environment>::GetInstance()->IsFlashMlaEnable();
  if (enable_flash_mla) {
    // Absorb has two versions
    if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
      llm_kernels::nvidia::InvokeFlashMla<SCALAR_T>(
          static_cast<SCALAR_T*>(q_tensor_ptr), static_cast<SCALAR_T*>(k_cache_ptr), q_seq_len, attn_scale,
          block_table_ptr, context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/,
          hidden_buffer_1, batch, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, rank,
          kv_cache_block_num, stream);
      // tp8: num_heads:16, total_tokens:256,kv_lora_rank:512,qk_rope_head_dim:64,w_uv_o_dim:7168,v_head_dim:128
      // [256, 16, 512] => [16, 256, 512]
      InvokePermute<SCALAR_T>(hidden_buffer_1, qkv_workspace, {total_tokens, num_heads, kv_lora_rank}, {1, 0, 2},
                              stream);
      // [16, 256, 512] * [16, 512, 128] => [16, 256, 128]
      InvokeBatchedMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, num_heads, total_tokens, v_head_dim, kv_lora_rank,
                                    qkv_workspace, w_uv_weight, hidden_buffer_1, stream, nullptr, 0, nullptr);
      // [16, 256, 128] => [256, 16, 128];
      InvokePermute<SCALAR_T>(hidden_buffer_1, output_ptr, {num_heads, total_tokens, v_head_dim}, {1, 0, 2}, stream);
    } else {
      llm_kernels::nvidia::InvokeFlashMla<SCALAR_T>(
          static_cast<SCALAR_T*>(q_tensor_ptr), static_cast<SCALAR_T*>(k_cache_ptr), q_seq_len, attn_scale,
          block_table_ptr, context_lens_ptr, tile_scheduler_metadata_ptr, num_splits_ptr, qkv_workspace /*workspace*/,
          output_ptr, batch, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, rank,
          kv_cache_block_num, stream);
    }

  } else {
    float softmax_scale = attn_scale;
    if (absorb_type == AbsorbWeightsType::kAbsorbTypeBMM) {
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage1<SCALAR_T>(
          q_tensor_ptr, k_cache_ptr, k_cache_ptr, softmax_scale, block_table_ptr, context_lens_ptr, hidden_buffer_1,
          total_tokens, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, stream);
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage2<SCALAR_T>(
          hidden_buffer_1, context_lens_ptr, qkv_workspace, total_tokens, num_heads, kv_lora_rank, stream);

      InvokePermute<SCALAR_T>(qkv_workspace, hidden_buffer_1, {total_tokens, num_heads, kv_lora_rank}, {1, 0, 2},
                              stream);
      InvokeBatchedMatMul<SCALAR_T>(cublas_handles, cublaslt_handles, num_heads, total_tokens, v_head_dim, kv_lora_rank,
                                    hidden_buffer_1, w_uv_weight, qkv_workspace, stream, nullptr, 0, nullptr);
      InvokePermute<SCALAR_T>(qkv_workspace, output_ptr, {num_heads, total_tokens, v_head_dim}, {1, 0, 2}, stream);
    } else {
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage1<SCALAR_T>(
          q_tensor_ptr, k_cache_ptr, k_cache_ptr, softmax_scale, block_table_ptr, context_lens_ptr, hidden_buffer_1,
          total_tokens, num_heads, kv_lora_rank, qk_rope_head_dim, block_size, max_blocks_per_seq, stream);
      Singleton<TritonWrapper>::GetInstance()->InvokeMlaAttenStage2<SCALAR_T>(
          hidden_buffer_1, context_lens_ptr, output_ptr, total_tokens, num_heads, kv_lora_rank, stream);
    }
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
      std::optional<llm_kernels::nvidia::RotaryEmbeddingCuda<SCALAR_T>>& rotary_embedding_cuda,                       \
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

}  // namespace ksana_llm
