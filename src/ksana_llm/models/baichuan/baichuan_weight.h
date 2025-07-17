/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common/common_weight.h"

namespace ksana_llm {

template <typename T>
class BaichuanWeight : public BaseWeight {
 public:
  BaichuanWeight() {}
  explicit BaichuanWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                          std::shared_ptr<Context> context);

  Tensor GetModelWeights(const std::string& weight_name);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list);

  void ProcessWeights();

  std::shared_ptr<CommonWeight<T>> Getcommonweight() { return common_weight_; }

  void SetEmbeddingsConfig();

 private:
  // the common weight instance.
  std::shared_ptr<CommonWeight<T>> common_weight_ = nullptr;
};

}  // namespace ksana_llm
