/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/attention_layer.h"

namespace ksana_llm {

class PagedAttentionLayer : public AttentionLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

#ifdef ENABLE_CUDA
  // Override SetWorkspaceBuffer to initialize FlashInfer workspaces.
  virtual Status SetWorkspaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  // Initialize shared FlashInfer workspaces for the given rank.
  // This function should be called after Init() if using FlashInfer backend.
  void SetFlashInferWorkspace(int num_heads, int num_kv_heads, int head_dim, int rank);
#endif

 private:
  template <typename SCALAR_T, typename CACHE_T, llm_kernels::utils::KVCacheType KV_DTYPE>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
#ifdef ENABLE_ACL
  // ATB attention implement for ascend device, for ATB run with graph, each layer need's one kernel implement instance
  std::shared_ptr<llm_kernels::ascend::ATBAttention> atb_paged_attn_;
#endif
  bool enable_blocked_multi_token_forwarding_kv_;
#ifdef ENABLE_CUDA
  bool use_flashinfer_for_decode_;
  // Shared FlashInfer workspaces across all layers on the same rank.
  void* shared_pinned_host_workspace_ = nullptr;
  std::shared_ptr<Tensor> shared_device_workspace_;
  // Shared FlashInfer prefill helper across all layers on the same rank.
  std::shared_ptr<void> flashinfer_prefill_helper_;
#endif
};

}  // namespace ksana_llm
