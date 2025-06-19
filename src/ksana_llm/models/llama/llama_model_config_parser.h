/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_config_parser.h"

namespace ksana_llm {

// Model config parser for llama.
class LlamaModelConfigParser : public BaseModelConfigParser {
 public:
  LlamaModelConfigParser();
  virtual ~LlamaModelConfigParser() override;

  // Parse config from config.json
  virtual Status ParseModelConfig(const nlohmann::json& config_json,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;

  // Parse config from gguf files.
  virtual Status ParseModelConfig(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                                  std::shared_ptr<BaseModelConfig>& model_config) override;
};

}  // namespace ksana_llm
