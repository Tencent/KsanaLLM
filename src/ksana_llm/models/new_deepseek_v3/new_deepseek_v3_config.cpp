/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/new_deepseek_v3/new_deepseek_v3_config.h"

namespace ksana_llm {

bool NewDeepSeekV3Config::ContainGptqWeights() const {
  return (quant_config.method == QUANT_GPTQ || quant_config.enable_moe_int4);
}

}  // namespace ksana_llm
