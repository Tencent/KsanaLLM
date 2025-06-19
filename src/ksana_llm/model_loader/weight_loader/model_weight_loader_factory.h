/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_weight_loader.h"
#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// The factory used to create different model weight loader.
class ModelWeightLoaderFactory {
 public:
  static Status CreateModelWeightLoader(ModelArchitecture model_arch, std::shared_ptr<BaseModelConfig> model_config,
                                        std::shared_ptr<Environment> env, std::shared_ptr<Context> context,
                                        std::shared_ptr<BaseModelWeightLoader>& model_weight_loader);
};

}  // namespace ksana_llm
