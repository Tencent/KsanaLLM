/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_weight_impl.h"

namespace ksana_llm {
// The new deepseek loader.
class NewDeepSeekV3WeightLoader : public BaseModelWeightLoader {
 public:
  NewDeepSeekV3WeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                            std::shared_ptr<Context> context);
  virtual ~NewDeepSeekV3WeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

  // Invoked only once after ProcessModelWeights.
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

 private:
  Status InitWeightLoaderImpl(std::shared_ptr<NewDeepSeekV3Config>& new_deepseek_v3_config);

 private:
  PipelineConfig pipeline_config_;
  std::unique_ptr<NewDeepSeekV3WeightImplBase> weight_impl_;
  // rank -> weight_name
  std::vector<std::unordered_set<std::string>> to_be_permuted_weights_;
};
}  // namespace ksana_llm
