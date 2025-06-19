/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/weight_loader/model_weight_loader_factory.h"

#include <memory>

#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

#include "ksana_llm/models/llama/llama_model_weight_loader.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_weight_loader.h"

namespace ksana_llm {

Status ModelWeightLoaderFactory::CreateModelWeightLoader(ModelArchitecture model_arch,
                                                         std::shared_ptr<BaseModelConfig> model_config,
                                                         std::shared_ptr<Environment> env,
                                                         std::shared_ptr<Context> context,
                                                         std::shared_ptr<BaseModelWeightLoader>& model_weight_loader) {
  switch (model_arch) {
    case ModelArchitecture::ARCH_LLAMA: {
      model_weight_loader = std::make_shared<LlamaModelWeightLoader>(model_config, env, context);
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK: {
      model_weight_loader = std::make_shared<NewDeepSeekV3WeightLoader>(model_config, env, context);
      break;
    }
    default: {
      model_weight_loader = nullptr;
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported model arch %d", static_cast<int>(model_arch)));
    }
  }

  return Status();
}

}  // namespace ksana_llm
