/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/base_model_config_parser.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

BaseModelConfigParser::~BaseModelConfigParser() {}

Status BaseModelConfigParser::ParseModelConfig(const nlohmann::json& config_json,
                                               const ParallelismBasicConfig& parallel_basic_config,
                                               std::shared_ptr<BaseModelConfig>& model_config) {
  KLLM_THROW("ParseModelConfig from json config not implemented.");
  return Status();
}

// Parse config from gguf files.
Status BaseModelConfigParser::ParseModelConfig(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                                               std::shared_ptr<BaseModelConfig>& model_config) {
  KLLM_THROW("ParseModelConfig from gguf meta not implemented.");
  return Status();
}

}  // namespace ksana_llm
