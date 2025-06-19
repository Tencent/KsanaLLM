/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/model_arch.h"

#include <string>
#include <unordered_map>

#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// buildin model type.
static const std::unordered_map<std::string, ModelArchitecture> model_type_to_archs = {
    {"llama", ModelArchitecture::ARCH_LLAMA},       {"qwen2_moe", ModelArchitecture::ARCH_QWEN2_MOE},
    {"qwen2_vl", ModelArchitecture::ARCH_QWEN2_VL}, {"qwen", ModelArchitecture::ARCH_QWEN},
    {"baichuan", ModelArchitecture::ARCH_BAICHUAN}, {"chatglm", ModelArchitecture::ARCH_CHATGLM},
    {"gpt", ModelArchitecture::ARCH_GPT},           {"fairseq-transformer", ModelArchitecture::ARCH_GPT},
    {"mixtral", ModelArchitecture::ARCH_MIXTRAL},   {"qwen3_moe", ModelArchitecture::ARCH_QWEN3_MOE},
    {"deepseek", ModelArchitecture::ARCH_DEEPSEEK}};

Status GetModelArchitectureFromString(const std::string& model_type, ModelArchitecture& model_arch) {
  for (const auto& [key, value] : model_type_to_archs) {
    if (model_type.find(key) != std::string::npos) {
      model_arch = value;
      return Status();
    }
  }

  return Status(RET_INVALID_ARGUMENT);
}

}  // namespace ksana_llm
