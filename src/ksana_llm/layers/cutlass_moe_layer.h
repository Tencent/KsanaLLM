/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/
#pragma once
#ifdef ENABLE_CUDA
#  include "csrc/kernels/nvidia/moe/cutlass_moe/cutlass_moe_wrapper.h"
#endif
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/layers/grouped_topk_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class CutlassMoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status Preprocess(const ModelConfig& model_config, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  // 执行 GroupedTopk 计算的辅助函数
  Status ExecuteGroupedTopk(const std::vector<Tensor>& input_tensors, int num_tokens);

  template <typename T>
  Status InitT(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
               std::shared_ptr<Context> context, int rank);

 protected:
  bool set_workspace_buffer_info_ = true;

  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_ws_bytes_;
  size_t max_token_num_;
  size_t expert_num_per_node_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;
  int group_size_;
  bool apply_weight_ = false;

  void* topk_weights_ptr_;
  size_t topk_weights_ptr_size;
  void* topk_ids_ptr_;
  size_t topk_ids_ptr_size;
  void* kernel_workspace_ptr_;
  size_t kernel_workspace_size;

  std::vector<std::vector<int64_t>> config_map_;

  std::shared_ptr<GroupedTopkLayer> grouped_topk_layer_;

  std::shared_ptr<llm_kernels::nvidia::CutlassMoeWrapper> cutlass_moe_wrapper_;
};
#endif
}  // namespace ksana_llm