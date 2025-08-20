/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/utils/config/model_config_parser.h"

namespace ksana_llm {

template <typename T>
class BgeRerankerMinicpmWeight : public CommonWeight<T> {
 public:
  BgeRerankerMinicpmWeight() {}
  explicit BgeRerankerMinicpmWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                                    std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;

 protected:
  Status ConvertCommonTensor(int hidden_units, int inter_size, int vocab_size);

  using CommonWeight<T>::context_;
  using CommonWeight<T>::rank_;
  using CommonWeight<T>::tensor_para_size_;

  using CommonWeight<T>::weights_map_;
  using CommonWeight<T>::weights_data_type_map_;
  using CommonWeight<T>::weight_data_type_;

  using CommonWeight<T>::model_config_;

  using CommonWeight<T>::tensor_manager_;

 private:
  Status PermuteLinearHeadWeight(Tensor& last_linear_head_tensor, const int num_layer);
};

}  // namespace ksana_llm