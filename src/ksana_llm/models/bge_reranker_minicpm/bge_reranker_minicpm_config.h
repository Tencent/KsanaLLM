/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/config/model_config_parser.h"
#include "ksana_llm/utils/logger.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

inline void PrepareBgeRerankerMinicpmAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.exist_tie_embeddings_param = true;
  model_config.emb_scale = config_json.value("scale_emb", 1.0f);
  model_config.scale_depth = config_json.value("scale_depth", 1.0f);
  model_config.start_layer = config_json.value("start_layer", 1);
}

}  // namespace ksana_llm