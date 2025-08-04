/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/base_layer.h"
#include "ksana_llm/utils/common_device.h"

namespace ksana_llm {

#ifdef ENABLE_CUDA
class GroupedTopkLayer : public BaseLayer {
 public:
  virtual Status Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) override;

  virtual Status Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) override;

 private:
  template <typename T>
  Status ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors);

 private:
  int topk_;
  bool renormalize_;
  int num_expert_group_;
  int topk_group_;
  std::string scoring_func_;
  float routed_scaling_factor_;
  bool use_e_score_correction_bias_;

  size_t expert_para_size_;
  size_t expert_world_size_;
};
#endif

}  // namespace ksana_llm
