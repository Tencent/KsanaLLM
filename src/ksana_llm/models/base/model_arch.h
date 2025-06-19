/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

enum class ModelArchitecture {
  // llama
  ARCH_LLAMA,
  // qwen2_moe
  ARCH_QWEN2_MOE,
  // qwen3_moe,
  ARCH_QWEN3_MOE,
  // qwen2_vl
  ARCH_QWEN2_VL,
  // qwen
  ARCH_QWEN,
  // baichuan
  ARCH_BAICHUAN,
  // chatglm
  ARCH_CHATGLM,
  // gpt
  ARCH_GPT,
  // mixtral
  ARCH_MIXTRAL,
  // DeepSeek V2/V3/R1
  ARCH_DEEPSEEK,
  // unknown
  ARCH_UNKNOWN,
};

// Get model architecture.
Status GetModelArchitectureFromString(const std::string& model_type, ModelArchitecture& model_arch);

}  // namespace ksana_llm
