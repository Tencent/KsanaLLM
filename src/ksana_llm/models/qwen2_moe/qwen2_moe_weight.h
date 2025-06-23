/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class Qwen2MoeWeight : public CommonMoeWeight<T> {
 public:
  Qwen2MoeWeight() {}
  explicit Qwen2MoeWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::rank_;

  using BaseWeight::model_config_;
  using BaseWeight::weights_map_;

  using CommonWeight<T>::tensor_manager_;

 private:
  Status PermuteShareGatingWeight(Tensor& last_share_gating_tensor, const int num_layer);
};

}  // namespace ksana_llm
