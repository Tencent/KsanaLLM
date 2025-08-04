/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/
#pragma once
#include "csrc/utils/nvidia/workspace.h"
#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/models/common_moe/moe_config.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class MarlinMoeLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual size_t GetWorkSpaceSize() override;

  virtual Status SetWorkSpaceBuffer(const std::shared_ptr<Tensor>& workspace_buffer) override;

  virtual Status Preprocess(const ModelConfig& model_config_, const RuntimeConfig& runtime_config) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  size_t GetWorkSpaceSizeT();

 private:
  MoeScaleNormMode moe_scale_norm_mode_;
  size_t max_token_num_;
  size_t expert_num_;
  size_t expert_hidden_size_;
  size_t expert_inter_size_;
  size_t expert_topk_;
  int tp_size_;
  bool use_lora_ = false;
  bool use_vllm_moe_ = false;
  uint32_t num_expert_group_ = 1;
  uint32_t expert_groups_topk_ = 1;
  std::string scoring_func_ = "softmax";
  std::string topk_method_ = "greedy";
  bool norm_topk_prob_ = false;
  float routed_scaling_factor_ = 1.0f;
  bool use_e_score_correction_bias_ = false;
  size_t group_size_ = 128;
  bool apply_weight_ = false;
  size_t num_bits_ = 4;

  size_t max_ws_bytes_;
  size_t max_gating_size_;
};
#endif
}  // namespace ksana_llm
