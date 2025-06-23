/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common_mla/common_mla_weight.h"
#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class DeepSeekV3Weight : public CommonMlaWeight<T>, public CommonMoeWeight<T> {
 public:
  DeepSeekV3Weight() {}
  explicit DeepSeekV3Weight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);

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
  using CommonWeight<T>::quant_weight_solver_;

  using CommonWeight<T>::model_config_;
  using BaseWeight::required_layer_idx_;

  using CommonWeight<T>::tensor_manager_;
};

}  // namespace ksana_llm