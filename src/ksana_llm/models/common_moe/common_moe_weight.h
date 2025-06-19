/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class CommonMoeWeight : virtual public CommonWeight<T> {
 public:
  CommonMoeWeight() {}
  explicit CommonMoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list,
                             std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::context_;
  using BaseWeight::rank_;
  using CommonWeight<T>::tensor_para_size_;
  using CommonWeight<T>::expert_world_size_;
  using CommonWeight<T>::expert_para_size_;
  using CommonWeight<T>::global_expert_para_size_;

  using CommonWeight<T>::weight_data_type_;
  using CommonWeight<T>::moe_weight_data_type_;
  using BaseWeight::weights_data_type_map_;
  using BaseWeight::weights_map_;

  using BaseWeight::model_config_;

  using CommonWeight<T>::tensor_manager_;

  using CommonWeight<T>::quant_weight_solver_;

  using BaseWeight::pipeline_config_;
  using CommonWeight<T>::required_layer_idx_;

 private:
  Status GetExpertsIdx(const std::string& expert_name, int& layer_idx, int& expert_idx);
  Status PermuteGatingWeight(Tensor& last_gating_tensor);
  Status ConvertShareMLPWeight(bool is_weight_scale);

  // represents the contiguous arrangement of actual_expert_id that each expert_id is mapped to in current rank.
  // For example, if num_experts = 4 and expert_para_size = 2, then:
  //   rank0: [0, 1, 3, 3]
  //   rank1: [3, 3, 0, 1]
  // Here, 3 is num_experts_per_rank + 1, indicating that the expert corresponding to this index is unavailable on this
  // rank.
  std::vector<int> expert_map_;
  size_t num_experts_per_rank_ = 1;
};

}  // namespace ksana_llm
