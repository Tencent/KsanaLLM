/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "ksana_llm/models/base/base_model_config_parser.h"

namespace ksana_llm {

// Model config parser for Qwen loader.
class NewQwenConfigParser : public BaseModelConfigParser {
 public:
  NewQwenConfigParser();
  virtual ~NewQwenConfigParser() override;

  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json& config_json,
                                  const ParallelismBasicConfig& parallel_basic_config,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;
};

}  // namespace ksana_llm
