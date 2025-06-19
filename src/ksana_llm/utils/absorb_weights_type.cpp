/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

AbsorbWeightsType ReadAbsorbWeightsType() {
  // maybe change the env variable name
  static const char* enable_absort = std::getenv("ENABLE_COMPRESSED_KV");
  int absorb_type = 0;
  if (enable_absort != nullptr) {
    if (strcmp(enable_absort, "1") == 0) {
      absorb_type = 1;
    } else if (strcmp(enable_absort, "2") == 0) {
      absorb_type = 2;
    } else if (strcmp(enable_absort, "0") == 0) {
      absorb_type = 0;
    } else {
      KLLM_THROW("ENABLE_COMPRESSED_KV must be 1, 2, or 0");
    }
  }
  return AbsorbWeightsType(absorb_type);
}

static AbsorbWeightsType g_absorb_weights_type = ReadAbsorbWeightsType();

bool IsAbsorbWeightsEnabled() {
  return GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeUKV ||
         GetAbsorbWeightsType() == AbsorbWeightsType::kAbsorbTypeBMM;
}

AbsorbWeightsType GetAbsorbWeightsType() { return g_absorb_weights_type; }

void SetAbsorbWeightsType(AbsorbWeightsType type) { g_absorb_weights_type = type; }

}  // namespace ksana_llm
