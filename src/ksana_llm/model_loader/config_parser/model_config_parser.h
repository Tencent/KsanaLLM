/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>
#include <string>

#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/utils/config/schedule_config_parser.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Used to parse model config from model dir.
class ModelConfigParser {
 public:
  Status ParseModelConfig(const std::string& model_dir, const ParallelismBasicConfig& parallel_basic_config,
                          std::shared_ptr<BaseModelConfig>& model_config);
};

}  // namespace ksana_llm
