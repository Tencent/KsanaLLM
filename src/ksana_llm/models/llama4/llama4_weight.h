/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class Llama4Weight : public CommonMoeWeight<T> {
 public:
  Llama4Weight() {}
  explicit Llama4Weight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::rank_;

  using BaseWeight::model_config_;
  using BaseWeight::weights_map_;

  using CommonWeight<T>::tensor_manager_;
  using CommonWeight<T>::context_;
  using CommonWeight<T>::required_layer_idx_;

 private:
  Status PermuteExpertsWeight();
};

}  // namespace ksana_llm
