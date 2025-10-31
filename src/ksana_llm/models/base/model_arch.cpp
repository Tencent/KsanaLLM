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
    {"llama", ModelArchitecture::ARCH_LLAMA},
    {"qwen2_moe", ModelArchitecture::ARCH_QWEN2_MOE},
    {"qwen2_vl", ModelArchitecture::ARCH_QWEN2_VL},
    {"qwen", ModelArchitecture::ARCH_QWEN},
    {"qwen2", ModelArchitecture::ARCH_QWEN},
    {"qwen3", ModelArchitecture::ARCH_QWEN},
    {"baichuan", ModelArchitecture::ARCH_BAICHUAN},
    {"chatglm", ModelArchitecture::ARCH_CHATGLM},
    {"gpt", ModelArchitecture::ARCH_GPT},
    {"fairseq-transformer", ModelArchitecture::ARCH_GPT},
    {"mixtral", ModelArchitecture::ARCH_MIXTRAL},
    {"qwen3_moe", ModelArchitecture::ARCH_QWEN3_MOE},
    {"deepseek_v2", ModelArchitecture::ARCH_DEEPSEEK},
    {"deepseek_v3", ModelArchitecture::ARCH_DEEPSEEK},
    {"deepseek_v32", ModelArchitecture::ARCH_DEEPSEEK},
    {"kimi_k2", ModelArchitecture::ARCH_DEEPSEEK}};

static const std::unordered_map<ModelArchitecture, std::string> arch_to_model_type = {
    {ModelArchitecture::ARCH_LLAMA, "llama"},
    {ModelArchitecture::ARCH_QWEN2_MOE, "qwen2_moe"},
    {ModelArchitecture::ARCH_QWEN2_VL, "qwen2_vl"},
    {ModelArchitecture::ARCH_QWEN, "qwen3"},
    {ModelArchitecture::ARCH_BAICHUAN, "baichuan"},
    {ModelArchitecture::ARCH_CHATGLM, "chatglm"},
    {ModelArchitecture::ARCH_GPT, "gpt"},
    {ModelArchitecture::ARCH_MIXTRAL, "mixtral"},
    {ModelArchitecture::ARCH_QWEN3_MOE, "qwen3_moe"},
    {ModelArchitecture::ARCH_DEEPSEEK, "deepseek_v3"}};

Status GetModelArchitectureFromString(const std::string& model_type, ModelArchitecture& model_arch) {
  for (const auto& [key, value] : model_type_to_archs) {
    if (model_type == key) {
      model_arch = value;
      return Status();
    }
  }

  return Status(RET_INVALID_ARGUMENT, fmt::format("Not supported model type: {}", model_type));
}

Status GetModelTypeFromArchitecture(ModelArchitecture model_arch, std::string& model_type) {
  const auto it = arch_to_model_type.find(model_arch);
  if (it != arch_to_model_type.end()) {
    model_type = it->second;
    return Status();
  }
  return Status(RET_INVALID_ARGUMENT, fmt::format("Not supported model arch: {}", static_cast<int>(model_arch)));
}

}  // namespace ksana_llm
