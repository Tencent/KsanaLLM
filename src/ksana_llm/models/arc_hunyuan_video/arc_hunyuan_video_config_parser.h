/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_config_parser.h"

namespace ksana_llm {

// Model config parser for ArcHunyuanVideo loader.
class ArcHunyuanVideoConfigParser : public BaseModelConfigParser {
 public:
  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json& config_json,
                                  const ParallelismBasicConfig& parallel_basic_config, const std::string& model_dir,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;
};

}  // namespace ksana_llm
