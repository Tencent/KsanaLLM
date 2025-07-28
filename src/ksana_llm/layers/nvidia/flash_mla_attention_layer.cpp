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
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Init(const std::vector<std::any>& parameters,
                                                                 const RuntimeConfig& runtime_config,
                                                                 std::shared_ptr<Context> context, int rank) {
  return AttentionLayer<SCALAR_T>::Init(parameters, runtime_config, context, rank);
}

template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
Status FlashMlaAttentionLayer<SCALAR_T, CACHE_T, KV_DTYPE>::Forward(const std::vector<Tensor>& input_tensors,
                                                                    std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //     0: 临时空间
  //     1: token_offset tensor shape [max_batch_size + 1], type uint64
  //     2: kv_list shape [num_layer, max_block_num, 2], type pointer
  //     3: prefix_offset_tensor shape [max_batch_size + 1], type int32
  //     4: kv_cache_offset_tensor shape [max_batch_size + 1], type int32
  //     5: rotary embedding pos tensor shape [max_token_num], type int64
  //        mrotary embedding pos tensor shape [3, max_token_num], type int64 (only for qwen2_vl)
  //     6: rotary embedding mask tensor shape [max_token_num], type int64
  //     7: flexible_rotary_embedding_pos,
  //     8: flexible_rotary_embedding_mask,
  //     9: dst_flexible_kv_cache_tensor,
  //     10: src_flexible_kv_cache_tensor,
  //     11: dst_flexible_token_idx_tensor,
  //     12: src_flexible_token_idx_tensor,
  //     13: flexible_offset_uint64_tensor,
  //     14: forward shape: [batch_size, max_tokens, kv_cache_offset_list.back(), xx, xx, xx, max_forwarding_tokens]
  //     15: query_layernorm_weight (if not use_qk_norm, value is nullptr)
  //     16: key_layernorm_weight (if not use_qk_norm, value is nullptr)
  //     17: flag_tensor: [use_cache]
#ifdef ENABLE_FLASH_ATTN_WITH_CACHE
  //     18: kv_cache_base_ptr_tensor: [1 + layer_num * 2]
  //     19: block_table: [batch_size, max_tokens]
  //     20: input_without_prefix_offset_tensor: [batch_size + 1]
#endif
  // output_tensors:
  //     0: flash_mla_attention_output shape: [std::max(max_batch_size * vocab_size, max_token_num * hidden_units * 3)]
  int max_tokens = input_tensors[15].shape[9];
  int batch_size = input_tensors[15].shape[8];
  int layer_block_num = input_tensors[15].shape[2];
  int total_tokens = input_tensors[22].shape[0];
  bool use_cache = input_tensors[18].GetPtr<bool>()[0];

  void** k_list = (input_tensors[2].GetPtr<void*>()) + this->layer_index_ * layer_block_num * 2;
  void** v_list = k_list + layer_block_num;

  void* q_nope_ptr = input_tensors[22].GetPtr<void>();
  void* q_pe_ptr = input_tensors[23].GetPtr<void>();
  void* compressed_kv_ptr = input_tensors[24].GetPtr<void>();
  void* k_pe_ptr = input_tensors[25].GetPtr<void>();
  void* kv_b_nope_proj_weight = input_tensors[26].GetPtr<void>();
  void* v_head_proj_weight = input_tensors[27].GetPtr<void>();

  void* kv_b_nope_weight_scale = nullptr;
  void* v_head_weight_scale = nullptr;
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    kv_b_nope_weight_scale = input_tensors[26].weight_scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[27].weight_scales->GetPtr<void>();
  } else if (this->mm_quant_mode_ == QUANT_GPTQ) {
    kv_b_nope_weight_scale = input_tensors[26].scales->GetPtr<void>();
    v_head_weight_scale = input_tensors[27].scales->GetPtr<void>();
  }

  size_t o_proj_dim = input_tensors[28].shape[1];
  if (this->mm_quant_mode_ == QUANT_BLOCK_FP8_E4M3) {
    o_proj_dim = input_tensors[28].shape[0];
  }

  void* prefix_k_buffer = input_tensors[29].GetPtr<void>();
  void* prefix_v_buffer = input_tensors[30].GetPtr<void>();
  void* prefix_o_buffer = input_tensors[31].GetPtr<void>();
  void* prefix_kv_buffer = input_tensors[32].GetPtr<void>();
  void* prefix_k_up_buffer = input_tensors[33].GetPtr<void>();
  void* prefix_v_up_buffer = input_tensors[34].GetPtr<void>();
  void* workspace_buffer = input_tensors[35].GetPtr<void>();

  int64_t kv_cache_block_num = *(input_tensors[19].GetPtr<int64_t>());
  void** layer_kv_cache_ptr = input_tensors[19].GetPtr<void*>() + 1;
  void* k_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2];
  void* v_cache_ptr = layer_kv_cache_ptr[this->layer_index_ * 2 + 1];
  int32_t* block_table_ptr = input_tensors[20].GetPtr<int32_t>();
  int max_blocks_per_seq = input_tensors[20].shape[1];
  size_t* input_without_prefix_offset = input_tensors[21].GetPtr<size_t>();
  int max_forwarding_tokens = input_tensors[15].shape[6];

  // The total length of prefix part.
  int total_prefix_len = input_tensors[15].shape[12];
  std::shared_ptr<Tensor>& work_buffer = this->workspace_buffer_;
  void* fp8_work_buffer = work_buffer == nullptr ? nullptr : work_buffer->GetPtr<void>();

  void* seqlens_q_ptr = input_tensors[4].GetPtr<void>();

  MlaAttenVarlen<SCALAR_T, CACHE_T, KV_DTYPE>(
      output_tensors[0].GetPtr<void>(), q_nope_ptr, q_pe_ptr, k_pe_ptr, compressed_kv_ptr, kv_b_nope_proj_weight,
      v_head_proj_weight, kv_b_nope_weight_scale, v_head_weight_scale, o_proj_dim, fp8_work_buffer,
      this->context_->ext->GetCublasHandles()[this->rank_], this->context_->ext->GetCublasLtHandles()[this->rank_],
      input_tensors[6].GetPtr<void>(), input_tensors[7].GetPtr<void>(), workspace_buffer,
      input_tensors[1].GetPtr<void>(), this->attn_scale_, this->rotary_embedding_cuda_, total_tokens, max_tokens,
      batch_size, this->num_heads_, this->qk_rope_head_dim_, this->qk_nope_head_dim_, this->kv_lora_rank_,
      this->v_head_dim_, this->num_kv_heads_, this->head_size_, this->stride_size_, this->k_scale_, this->v_scale_,
      this->attn_dp_atp_size_, this->is_causal_, this->rank_, this->block_token_num_, k_list, v_list,
      input_tensors[3].GetPtr<void>(), input_tensors[5].GetPtr<void>(), this->alibi_slopes_, this->layer_index_,
      input_tensors[8].GetPtr<void>(), input_tensors[9].GetPtr<void>(), input_tensors[10].GetPtr<void>(),
      input_tensors[11].GetPtr<void>(), input_tensors[12].GetPtr<void>(), input_tensors[13].GetPtr<void>(),
      input_tensors[14].GetPtr<void>(), input_tensors[10].shape[0], this->layernorm_eps_, this->use_qk_norm_,
      input_tensors[16].GetPtr<void>(), input_tensors[17].GetPtr<void>(), use_cache,
      this->context_->GetComputeStreams()[this->rank_].Get(), k_cache_ptr, v_cache_ptr, block_table_ptr,
      kv_cache_block_num, max_blocks_per_seq, input_without_prefix_offset, max_forwarding_tokens, total_prefix_len,
      seqlens_q_ptr, prefix_k_buffer, prefix_v_buffer, prefix_o_buffer, prefix_kv_buffer, prefix_k_up_buffer,
      prefix_v_up_buffer, this->mm_quant_mode_);

  // 通知 LayerProgressTracker 该层已完成，它会在内部记录 event 并在单独的线程中监控完成情况
  Singleton<LayerProgressTracker>::GetInstance()->RecordLayerProgress(this->rank_, this->layer_index_,
                                                                      this->context_->GetComputeStreams()[this->rank_]);

  output_tensors[0].shape = {total_tokens, o_proj_dim};
  output_tensors[0].dtype = input_tensors[0].dtype;
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
