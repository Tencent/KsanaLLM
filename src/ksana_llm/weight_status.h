/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include <string>
#include <unordered_map>

#include "ksana_llm/utils/config/model_config_parser.h"

namespace ksana_llm {

// Weight status information for model weights
// Stores quantization mode and layout information (e.g., dimensions like "k", "n")
struct WeightStatus {
  QuantMode quant_mode = QuantMode::QUANT_NONE;
  std::unordered_map<std::string, size_t> layout;
};

}  // namespace ksana_llm