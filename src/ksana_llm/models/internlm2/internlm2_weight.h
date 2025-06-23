/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class Internlm2Weight : public CommonWeight<T> {
 public:
  Internlm2Weight() {}
  explicit Internlm2Weight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::context_;
  using BaseWeight::rank_;

  using CommonWeight<T>::tensor_para_size_;

  using BaseWeight::weights_map_;
  using CommonWeight<T>::weight_data_type_;
  using BaseWeight::weights_data_type_map_;

  using BaseWeight::model_config_;

  using CommonWeight<T>::tensor_manager_;

 private:
  Status PermuteLoraWeight(Tensor& last_lora_a_tensor, Tensor& last_lora_b_tensor, const int num_layer,
                           const std::string& last_lora_a_proj_name, const std::string& last_lora_b_proj_name);
};

}  // namespace ksana_llm
