/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/models/base/model_format.h"

namespace ksana_llm {

// The base class of all model configs.
struct BaseModelConfig {
  virtual ~BaseModelConfig();

  // The model dir.
  std::string model_dir;

  // The file format of current model.
  ModelFormat model_format;

  // The model architecture, such as hunyuan/llama/qwen, etc.
  ModelArchitecture model_arch;
};

}  // namespace ksana_llm
