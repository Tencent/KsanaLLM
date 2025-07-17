/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include "ksana_llm/models/common_moe/common_moe_weight.h"

namespace ksana_llm {

template <typename T>
class MixtralWeight : public CommonMoeWeight<T> {
 public:
  MixtralWeight() {}
  explicit MixtralWeight(const ModelConfig& model_config, const RuntimeConfig& runtime_config, int rank,
                         std::shared_ptr<Context> context);

  Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                             const std::vector<std::string>& weight_name_list,
                             const std::vector<std::string>& custom_name_list) override;

  void ProcessWeights() override;
};

}  // namespace ksana_llm
