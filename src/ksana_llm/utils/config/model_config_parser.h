/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

#include "ksana_llm/utils/config/model_config.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/search_path.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/yaml_reader.h"

namespace ksana_llm {

class EnvModelConfigParser {
 public:
  EnvModelConfigParser(const std::string &weight_quant_method, const std::string &gptq_backend)
      : weight_quant_method_(weight_quant_method), gptq_backend_(gptq_backend) {}

  // Parse model config from model dir.
  Status ParseModelConfig(const std::string &model_dir, const std::string &tokenizer_dir,
                          const std::string &model_config_filename, ModelConfig &model_config);

  // Parse Model Quant Config
  void ParseModelQuantConfig(const nlohmann::json &config_json, ModelConfig &model_config,
                             std::string &yaml_weight_quant_method, std::string &yaml_gptq_backend);

 private:
  // Parse model config from GGUF file.
  Status ParseModelConfigFromGGUF(const std::string &meta_file_path, ModelConfig &model_config);

 private:
  // The config of quantization.
  std::string weight_quant_method_;

  // The backend of gptq/awq quantization.
  std::string gptq_backend_;
};

}  // namespace ksana_llm
