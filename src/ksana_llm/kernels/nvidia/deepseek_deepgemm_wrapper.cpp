/*
 * Copyright 2025 Tencent Inc.  All rights reserved.
 */

#include "ksana_llm/kernels/nvidia/deepseek_deepgemm_wrapper.h"

#include <torch/torch.h>

#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/nvidia/deepseek_deepgemm_bridge.h"

#include "csrc/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"

using namespace llm_kernels::utils;

namespace ksana_llm {
namespace nvidia {

std::shared_ptr<DeepSeekDeepGEMMWrapper> DeepSeekDeepGEMMWrapper::GetInstance(int rank) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  std::lock_guard<std::mutex> lock(instances_mutex_);

  auto it = instances_.find(rank);
  if (it != instances_.end()) {
    return it->second;
  }

  // 创建新实例
  auto instance = std::shared_ptr<DeepSeekDeepGEMMWrapper>(new DeepSeekDeepGEMMWrapper(rank));
  instances_[rank] = instance;
  return instance;
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, cannot create DeepSeekDeepGEMMWrapper instance.");
#endif
}

DeepSeekDeepGEMMWrapper::DeepSeekDeepGEMMWrapper(int id) : id_(id) {
#if ENABLE_DEEPSEEK_DEEPGEMM
  num_device_sms_ = GetSMCount();

  int current_device = -1;
  cudaGetDevice(&current_device);
  KLLM_LOG_DEBUG << "Initializing DeepSeekDeepGEMMWrapper on device " << current_device << " with " << num_device_sms_
                 << " SMs";

  deepseek::InitializeDeepGemm(num_device_sms_);
  KLLM_LOG_DEBUG << "DeepSeekDeepGEMMWrapper initialized successfully on device " << current_device;
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::Fp8MqaLogits(const torch::Tensor& q_fp8,
                                           const std::pair<torch::Tensor, torch::Tensor>& kv_fp8,
                                           const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                                           const torch::Tensor& cu_seq_len_k_end, torch::Tensor& logits,
                                           bool clean_logits) {
#if ENABLE_DEEPSEEK_DEEPGEMM

  KLLM_CHECK_WITH_INFO(q_fp8.dim() == 3, "Expect q tensor of shape [seq_len, num_heads, head_dim]");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.dim() == 2, "Expect kv tensor of shape [seq_len_kv, head_dim]");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.dim() == 1 || kv_fp8.second.dim() == 2, "Expect kv scales tensor of rank 1 or 2");
  KLLM_CHECK_WITH_INFO(weights.dim() == 2, "Expect weights tensor of shape [seq_len, num_heads]");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.dim() == 1, "Expect cu_seq_len_k_start rank 1 tensor");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.dim() == 1, "Expect cu_seq_len_k_end rank 1 tensor");

  const int seq_len = q_fp8.size(0);
  const int num_heads = q_fp8.size(1);
  const int head_dim = q_fp8.size(2);
  const int seq_len_kv = kv_fp8.first.size(0);
  const int head_dim_kv = kv_fp8.first.size(1);

  KLLM_CHECK_WITH_INFO(head_dim_kv == head_dim, "KV head dim must match query head dim");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(0)) == seq_len, "Weights first dimension must equal seq_len");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(1)) == num_heads, "Weights second dimension must equal num_heads");
  KLLM_CHECK_WITH_INFO(static_cast<int>(cu_seq_len_k_start.size(0)) == seq_len,
                       "cu_seq_len_k_start length must equal seq_len");
  KLLM_CHECK_WITH_INFO(static_cast<int>(cu_seq_len_k_end.size(0)) == seq_len,
                       "cu_seq_len_k_end length must equal seq_len");

  KLLM_CHECK_WITH_INFO(q_fp8.is_contiguous(), "q tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.is_contiguous(), "kv tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.is_contiguous(), "kv scales tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(weights.is_contiguous(), "weights tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.is_contiguous(), "cu_seq_len_k_start must be contiguous");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.is_contiguous(), "cu_seq_len_k_end must be contiguous");

  KLLM_CHECK_WITH_INFO(q_fp8.scalar_type() == torch::kFloat8_e4m3fn, "q tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(kv_fp8.first.scalar_type() == torch::kFloat8_e4m3fn, "kv tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(kv_fp8.second.scalar_type() == torch::kFloat, "kv scales tensor must be float32");
  KLLM_CHECK_WITH_INFO(weights.scalar_type() == torch::kFloat, "weights tensor must be float32");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_start.scalar_type() == torch::kInt, "cu_seq_len_k_start tensor must be int32");
  KLLM_CHECK_WITH_INFO(cu_seq_len_k_end.scalar_type() == torch::kInt, "cu_seq_len_k_end tensor must be int32");

  constexpr int seq_len_alignment = 4;
  constexpr int block_kv = 256;
  const int aligned_seq_len = RoundUp(seq_len, seq_len_alignment);
  const int aligned_seq_len_kv = RoundUp(seq_len_kv + block_kv, seq_len_alignment);
  KLLM_CHECK_WITH_INFO(logits.device() == q_fp8.device(), "logits tensor must be on the same device as q");
  KLLM_CHECK_WITH_INFO(logits.scalar_type() == torch::kFloat, "logits tensor must be float32");
  KLLM_CHECK_WITH_INFO(logits.dim() == 2, "logits tensor must have rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(0)) >= aligned_seq_len,
                       "logits rows must be at least the aligned sequence length");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(1)) >= aligned_seq_len_kv,
                       "logits cols must be at least the aligned kv length");
  KLLM_CHECK_WITH_INFO(logits.stride(1) == 1, "logits tensor must be contiguous on the last dimension");
  logits = logits.narrow(0, 0, seq_len).narrow(1, 0, seq_len_kv);
  CUDA_CHECK_LAST_ERROR(deepseek::SmxxFp8MqaLogits(q_fp8, kv_fp8.first, kv_fp8.second, weights, cu_seq_len_k_start,
                                                   cu_seq_len_k_end, logits, seq_len, seq_len_kv, aligned_seq_len_kv,
                                                   num_heads, head_dim, seq_len_alignment, clean_logits));
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::Fp8PagedMqaLogits(const torch::Tensor& q_fp8, const torch::Tensor& fused_kv_cache,
                                                const torch::Tensor& weights, const torch::Tensor& context_lens,
                                                const torch::Tensor& block_table, const torch::Tensor& schedule_meta,
                                                torch::Tensor& logits, int max_context_len, bool clean_logits) {
#if ENABLE_DEEPSEEK_DEEPGEMM

  KLLM_CHECK_WITH_INFO(q_fp8.dim() == 4, "Expect q tensor of shape [batch_size, next_n, num_heads, head_dim]");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.dim() == 4, "Expect fused_kv_cache tensor of rank 4");
  KLLM_CHECK_WITH_INFO(weights.dim() == 2, "Expect weights tensor of rank 2");
  KLLM_CHECK_WITH_INFO(context_lens.dim() == 1, "Expect context_lens tensor of rank 1");
  KLLM_CHECK_WITH_INFO(block_table.dim() == 2, "Expect block_table tensor of rank 2");
  KLLM_CHECK_WITH_INFO(schedule_meta.dim() == 2, "Expect schedule_meta tensor of rank 2");

  const int batch_size = static_cast<int>(q_fp8.size(0));
  const int next_n = static_cast<int>(q_fp8.size(1));
  const int num_heads = static_cast<int>(q_fp8.size(2));
  const int head_dim = static_cast<int>(q_fp8.size(3));

  const int num_kv_blocks = static_cast<int>(fused_kv_cache.size(0));
  const int block_kv = static_cast<int>(fused_kv_cache.size(1));
  const int num_heads_kv = static_cast<int>(fused_kv_cache.size(2));
  const int head_dim_with_sf = static_cast<int>(fused_kv_cache.size(3));

  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(0)) == batch_size * next_n,
                       "weights first dimension must equal batch_size * next_n");
  KLLM_CHECK_WITH_INFO(static_cast<int>(weights.size(1)) == num_heads, "weights second dimension must equal num_heads");
  KLLM_CHECK_WITH_INFO(static_cast<int>(context_lens.size(0)) == batch_size,
                       "context_lens length must equal batch_size");
  KLLM_CHECK_WITH_INFO(static_cast<int>(block_table.size(0)) == batch_size,
                       "block_table first dimension must equal batch_size");
  KLLM_CHECK_WITH_INFO(num_heads_kv == 1, "Expect num_heads_kv == 1 in fused_kv_cache");
  KLLM_CHECK_WITH_INFO(head_dim_with_sf == head_dim + static_cast<int>(sizeof(float)),
                       "Expect fused_kv_cache last dim to equal head_dim + sizeof(float)");

  const int num_sms = num_device_sms_;
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_meta.size(0)) == num_sms + 1,
                       "schedule_meta rows must equal num_sms + 1");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_meta.size(1)) == 2, "schedule_meta cols must equal 2");

  KLLM_CHECK_WITH_INFO(q_fp8.is_contiguous(), "q tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(weights.is_contiguous(), "weights tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(context_lens.is_contiguous(), "context_lens tensor must be contiguous");
  KLLM_CHECK_WITH_INFO(schedule_meta.is_contiguous(), "schedule_meta tensor must be contiguous");

  KLLM_CHECK_WITH_INFO(q_fp8.scalar_type() == torch::kFloat8_e4m3fn, "q tensor must be float8 e4m3");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.scalar_type() == torch::kByte, "fused_kv_cache tensor must be uint8");
  KLLM_CHECK_WITH_INFO(weights.scalar_type() == torch::kFloat, "weights tensor must be float32");
  KLLM_CHECK_WITH_INFO(context_lens.scalar_type() == torch::kInt, "context_lens tensor must be int32");
  KLLM_CHECK_WITH_INFO(block_table.scalar_type() == torch::kInt, "block_table tensor must be int32");
  KLLM_CHECK_WITH_INFO(schedule_meta.scalar_type() == torch::kInt, "schedule_meta tensor must be int32");

  const int kv_cache_stride_bytes = static_cast<int>(fused_kv_cache.stride(0));
  const int block_table_stride = static_cast<int>(block_table.stride(0));

  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(1) == head_dim_with_sf,
                       "fused_kv_cache stride(1) must equal head_dim + sizeof(float)");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(2) == head_dim_with_sf,
                       "fused_kv_cache stride(2) must equal head_dim + sizeof(float)");
  KLLM_CHECK_WITH_INFO(fused_kv_cache.stride(3) == 1, "fused_kv_cache stride(3) must equal 1");
  KLLM_CHECK_WITH_INFO(block_table.stride(1) == 1, "block_table stride(1) must equal 1");
  KLLM_CHECK_WITH_INFO(kv_cache_stride_bytes % static_cast<int>(sizeof(float)) == 0,
                       "kv cache stride must be divisible by sizeof(float), kv_cache_stride_bytes");

  const auto device = fused_kv_cache.device();
  const auto fp8_options = torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(device);
  const auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto kv_cache =
      torch::from_blob(fused_kv_cache.data_ptr(), {num_kv_blocks, block_kv, head_dim},
                       {static_cast<int64_t>(kv_cache_stride_bytes), static_cast<int64_t>(head_dim), 1}, fp8_options);

  auto kv_cache_scales =
      torch::from_blob(fused_kv_cache.data_ptr<uint8_t>() + block_kv * head_dim, {num_kv_blocks, block_kv},
                       {static_cast<int64_t>(kv_cache_stride_bytes / static_cast<int>(sizeof(float))), 1}, f32_options);

  constexpr int num_math_warp_groups = 4;
  const int aligned_max_context_len = RoundUp(max_context_len, num_math_warp_groups * block_kv);

  KLLM_CHECK_WITH_INFO(logits.device() == q_fp8.device(), "logits tensor must be on the same device as q");
  KLLM_CHECK_WITH_INFO(logits.scalar_type() == torch::kFloat, "logits tensor must be float32");
  KLLM_CHECK_WITH_INFO(logits.dim() == 2, "logits tensor must have rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(0)) >= batch_size * next_n,
                       "logits rows must be at least batch_size * next_n");
  KLLM_CHECK_WITH_INFO(static_cast<int>(logits.size(1)) >= aligned_max_context_len,
                       "logits cols must be at least the aligned max context length");
  KLLM_CHECK_WITH_INFO(logits.stride(1) == 1, "logits tensor must be contiguous on the last dimension");
  logits = logits.narrow(-1, 0, max_context_len);
  CUDA_CHECK_LAST_ERROR(deepseek::SmxxFp8PagedMqaLogits(
      q_fp8, kv_cache, kv_cache_scales, weights, context_lens, logits, block_table, schedule_meta, batch_size, next_n,
      num_heads, head_dim, num_kv_blocks, block_kv, kv_cache_stride_bytes, aligned_max_context_len, block_table_stride,
      num_sms, num_math_warp_groups, clean_logits, max_context_len));
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

void DeepSeekDeepGEMMWrapper::PagedMqaLogitsMetadata(const torch::Tensor& context_lens,
                                                     torch::Tensor& schedule_metadata, int batch_size, int block_kv) {
#if ENABLE_DEEPSEEK_DEEPGEMM

  KLLM_CHECK_WITH_INFO(context_lens.scalar_type() == torch::kInt, "context_lens tensor must be int32");
  KLLM_CHECK_WITH_INFO(context_lens.is_contiguous(), "context_lens must be contiguous");
  KLLM_CHECK_WITH_INFO(context_lens.dim() == 1, "context_lens must be rank 1");
  KLLM_CHECK_WITH_INFO(static_cast<int>(context_lens.size(0)) == batch_size, "context_lens size must equal batch_size");

  KLLM_CHECK_WITH_INFO(schedule_metadata.scalar_type() == torch::kInt, "schedule_metadata tensor must be int32");
  KLLM_CHECK_WITH_INFO(schedule_metadata.dim() == 2, "schedule_metadata must be rank 2");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_metadata.size(0)) == num_device_sms_ + 1,
                       "schedule_metadata rows must equal num_sms + 1");
  KLLM_CHECK_WITH_INFO(static_cast<int>(schedule_metadata.size(1)) == 2, "schedule_metadata cols must equal 2");

  CUDA_CHECK_LAST_ERROR(
      deepseek::SmxxPagedMqaLogitsMetadata(context_lens, schedule_metadata, batch_size, block_kv, num_device_sms_));
#else
  KLLM_KERNEL_THROW("ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM kernel.");
#endif
}

}  // namespace nvidia
}  // namespace ksana_llm
