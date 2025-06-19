/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/model_loader/file_loader/model_file_loader.h"
#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/models/base/model_weight.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Used to load model weights.
class ModelWeightLoader {
 public:
  ModelWeightLoader(std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~ModelWeightLoader();

  // Load weights from file, the index of weights is device rank.
  Status LoadWeights(std::shared_ptr<BaseModelConfig>& model_config,
                     std::vector<std::shared_ptr<ModelWeight>>& dev_weights);

 private:
  std::shared_ptr<Environment> env_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  // Used to load weight, one thread for every device.
  std::shared_ptr<ThreadPool> weight_loader_threadpool_ = nullptr;
};

}  // namespace ksana_llm
