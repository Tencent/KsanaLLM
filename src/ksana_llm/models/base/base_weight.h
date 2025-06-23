/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/base_file_tensor_loader.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/tensor_manager.h"

namespace ksana_llm {

class BaseWeight {
 public:
  BaseWeight() {}
  explicit BaseWeight(const ModelConfig& model_config, int rank, std::shared_ptr<Context> context);
  virtual ~BaseWeight() {}

  // 查表,返回 weights_map_[weight_name]
  virtual Tensor GetModelWeights(const std::string& weight_name) = 0;
  // 加载权重
  virtual Status LoadWeightsFromFile(const std::shared_ptr<BaseFileTensorLoader> weights_loader,
                                     const std::vector<std::string>& weight_name_list,
                                     const std::vector<std::string>& custom_name_list) = 0;

  virtual bool TryToLoadWeightsFromCache();

  virtual bool SaveWeightsToCacheFolder();

  virtual bool IsPipelineNodeWeight(const std::string& tensor_name);

  virtual void ProcessWeights() = 0;

  virtual void SetEmbeddingsConfig() = 0;

  virtual void PrintDebugMessage() {}

  std::string GetCacheFolder();

 protected:
  int rank_ = 0;
  std::shared_ptr<Context> context_{nullptr};
  ModelConfig model_config_;
  PipelineConfig pipeline_config_;

  std::unordered_map<std::string, Tensor> weights_map_;
  std::unordered_map<std::string, DataType> weights_data_type_map_;

  std::shared_ptr<TensorManager> tensor_manager_;

  struct {
    std::unordered_set<int16_t> all, moe, dense;
  } required_layer_idx_;
};

}  // namespace ksana_llm
