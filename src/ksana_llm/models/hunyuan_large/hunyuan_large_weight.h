/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class HunyuanLargeWeight : public CommonMoeWeight<T> {
 public:
  HunyuanLargeWeight() {}
  explicit HunyuanLargeWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                              std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  using BaseWeight::context_;
  using BaseWeight::rank_;
  using CommonWeight<T>::tensor_para_size_;

  using CommonWeight<T>::moe_weight_data_type_;
  using BaseWeight::weights_data_type_map_;
  using BaseWeight::weights_map_;

  using BaseWeight::model_config_;

  using CommonWeight<T>::tensor_manager_;

 private:
  Status PermuteQueryWeight(Tensor& last_q_proj_tensor, const int num_layer);
};

}  // namespace ksana_llm