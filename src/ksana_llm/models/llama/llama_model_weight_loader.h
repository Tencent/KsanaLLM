/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include "ksana_llm/models/base/common_model_weight_loader.h"

namespace ksana_llm {

// The llama loader.
class LlamaModelWeightLoader : public BaseModelWeightLoader {
 public:
  LlamaModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config, std::shared_ptr<Environment> env,
                         std::shared_ptr<Context> context);
  virtual ~LlamaModelWeightLoader() override;

  // Do some filter on model weight names.
  virtual Status FilterWeightNames(std::vector<std::string>& weight_names) override;

  // Poseprocess
  virtual Status PostProcessModelWeights(std::unordered_map<std::string, Tensor>& dev_weights_map,
                                         int dev_rank) override;

  // Process weights, such as rename, split, merge, type convert, quantization, etc.
  virtual Status ProcessModelWeights(const std::unordered_map<std::string, Tensor>& host_model_weights, int dev_rank,
                                     std::unordered_map<std::string, Tensor>& device_model_weights,
                                     std::unordered_map<std::string, Tensor>& left_host_weights) override;

 private:
  PipelineConfig pipeline_config_;
  std::vector<std::unordered_set<std::string>> weights_to_permute_;
  std::unique_ptr<CommonModelWeightLoader> common_weight_loader_;
};

}  // namespace ksana_llm
