/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <unordered_map>

#include "ksana_llm/models/base/base_weight.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class ModelWeightLoader;

// class for model weights.
class ModelWeight : public BaseWeight {
 public:
  ModelWeight();
  virtual ~ModelWeight() override;

  friend class ModelWeightLoader;

  // Get a weight tensor by name.
  const Tensor& GetWeightTensor(const std::string& weight_name) const;

  // Get the weight name list.
  std::vector<std::string> GetWeightNames() const;

  // //////////////////////////////////////////////////////////////////////////////////////////
  // For compatible with common_model, improve it later
  // //////////////////////////////////////////////////////////////////////////////////////////
  virtual Tensor GetModelWeights(const std::string& weight_name) override { return GetWeightTensor(weight_name); }

  virtual Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                     const std::vector<std::string>& weight_name_list,
                                     const std::vector<std::string>& custom_name_list) override {
    return Status();
  }

  virtual void ProcessWeights() override {}

  virtual void SetEmbeddingsConfig() override {}

  virtual void PrintDebugMessage() override {}
  // //////////////////////////////////////////////////////////////////////////////////////////

  // currently used for deepseek v3 and will support other models in the future
  Status BindQuantWeightScales();

 private:
  using BaseWeight::weights_map_;
  Tensor empty_tensor_;
};

}  // namespace ksana_llm
