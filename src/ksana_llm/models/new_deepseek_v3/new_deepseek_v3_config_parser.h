/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_config_parser.h"
#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"

namespace ksana_llm {

// Model config parser for deepseek_v3.
class NewDeepSeekV3ConfigParser : public BaseModelConfigParser {
 public:
  NewDeepSeekV3ConfigParser();
  virtual ~NewDeepSeekV3ConfigParser() override;

  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json& config_json,
                                  const ParallelismBasicConfig& parallel_basic_config,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;

 private:
  // Parse quant config
  Status ParseQuantConfig(const nlohmann::json& quant_config_json,
                          std::shared_ptr<NewDeepSeekV3Config> new_deepseek_v3_config,
                          const std::string& yaml_weight_quant_method, const std::string& yaml_gptq_backend);
};
}  // namespace ksana_llm
