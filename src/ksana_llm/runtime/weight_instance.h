/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <future>
#include <memory>
#include <string>

#include "ksana_llm/kernels/cast.h"
#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/model_loader/config_parser/model_config_parser.h"
#include "ksana_llm/model_loader/weight_loader/model_weight_loader.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/runtime/weight_instance_inferface.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class WeightInstance : public WeightInstanceInterface {
 public:
  WeightInstance(const ModelConfig& model_config, std::shared_ptr<Context> context)
      : model_config_(model_config), context_(context) {
    loader_weight_threadpool_ = std::make_shared<ThreadPool>(context->GetTensorParallelSize());
    loader_weight_threadpool_->Start();
  }
  ~WeightInstance() {
    loader_weight_threadpool_->Stop();
    weights_.clear();
  }

  // Load model with specified model config.
  void Load();

  std::shared_ptr<BaseWeight> GetWeight(int rank) override { return weights_[rank]; }

 private:
  void SetEmbeddingsConfig();

  void CheckTieEmbeddings(int weight_file_size);

  void CheckTieEmbeddings(std::vector<std::string>& custom_name_list);

  void CreateWeightInstances();

  void LoadWeightsAndModelsMap(bool& loaded_from_cache);

  void LoadWeights();

  void ProcessWeights();

  bool TryToLoadWeightsFromCache();

  bool SaveWeightsToCache();

  // tmp function to check whether current model and configs satisfy new model weight loader.
  // TODO(huicongyao): remove this function later.
  bool IsCompatibleWithNewLoader(std::shared_ptr<BaseModelConfig> model_config);

 private:
  // The model config.
  ModelConfig model_config_;

  // The global context.
  std::shared_ptr<Context> context_ = nullptr;

  // The base model and weight, shared by all model instances.
  std::vector<std::shared_ptr<BaseWeight>> weights_;

  std::shared_ptr<ThreadPool> loader_weight_threadpool_ = nullptr;
};

}  // namespace ksana_llm
