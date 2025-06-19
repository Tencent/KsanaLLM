/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/config_parser/model_config_parser_factory.h"

#include "ksana_llm/models/llama/llama_model_config_parser.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config_parser.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

Status ModelConfigParserFactory::CreateModelConfigParser(ModelArchitecture model_arch,
                                                         std::shared_ptr<BaseModelConfigParser>& model_config_parser) {
  switch (model_arch) {
    case ModelArchitecture::ARCH_LLAMA: {
      model_config_parser = std::make_shared<LlamaModelConfigParser>();
      break;
    }
    case ModelArchitecture::ARCH_DEEPSEEK: {
      model_config_parser = std::make_shared<NewDeepSeekV3ConfigParser>();
      break;
    }
    default: {
      model_config_parser = nullptr;
      return Status(RET_INVALID_ARGUMENT, FormatStr("Not supported model arch %d", static_cast<int>(model_arch)));
    }
  }

  return Status();
}

}  // namespace ksana_llm
