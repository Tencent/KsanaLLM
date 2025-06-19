/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/utils/status.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

// Used to parse model config in pytorch format.
class PytorchConfigParser {
 public:
  Status GetJsonConfig(const std::string& model_dir, nlohmann::json& config_json, ModelArchitecture& model_arch);
};

}  // namespace ksana_llm
