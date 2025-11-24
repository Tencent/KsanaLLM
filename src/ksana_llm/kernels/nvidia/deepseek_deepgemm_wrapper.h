/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <torch/torch.h>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/nvidia/deepseek_deepgemm_bridge.h"

namespace ksana_llm {
namespace nvidia {

class DeepSeekDeepGEMMWrapper {
 public:
  // 获取指定 rank 的实例（线程安全的单例模式）
  static std::shared_ptr<DeepSeekDeepGEMMWrapper> GetInstance(int rank);

  // 清理所有实例和 DeepGEMM 运行时
  static void Shutdown() {
#if ENABLE_DEEPSEEK_DEEPGEMM
    std::lock_guard<std::mutex> lock(instances_mutex_);
    instances_.clear();
    deepseek::ShutdownDeepGemmRuntime();
#else
    KLLM_LOG_DEBUG << "ENABLE_DEEPSEEK_DEEPGEMM=0, skipping DeepSeek DeepGEMM shutdown.";
#endif
  }

  explicit DeepSeekDeepGEMMWrapper(int id);
  ~DeepSeekDeepGEMMWrapper() = default;

  // MQA attention functions using DeepSeek's DeepGEMM
  void Fp8MqaLogits(const torch::Tensor& q_fp8, const std::pair<torch::Tensor, torch::Tensor>& kv_fp8,
                    const torch::Tensor& weights, const torch::Tensor& cu_seq_len_k_start,
                    const torch::Tensor& cu_seq_len_k_end, torch::Tensor& logits, bool clean_logits = true);

  void Fp8PagedMqaLogits(const torch::Tensor& q_fp8, const torch::Tensor& fused_kv_cache, const torch::Tensor& weights,
                         const torch::Tensor& context_lens, const torch::Tensor& block_table,
                         const torch::Tensor& schedule_meta, torch::Tensor& logits, int max_context_len,
                         bool clean_logits = false);

  // Generate schedule metadata for paged MQA
  void PagedMqaLogitsMetadata(const torch::Tensor& context_lens, torch::Tensor& schedule_metadata, int batch_size,
                              int block_kv);

 private:
  // 静态成员变量（用于按 rank 管理实例）
  inline static std::unordered_map<int, std::shared_ptr<DeepSeekDeepGEMMWrapper>> instances_;
  inline static std::mutex instances_mutex_;

  int num_device_sms_;
  int id_{0};
};

}  // namespace nvidia
}  // namespace ksana_llm
