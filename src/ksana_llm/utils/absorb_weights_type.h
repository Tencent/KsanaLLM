/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

namespace ksana_llm {

enum AbsorbWeightsType {
  kAbsorbDisabled = 0,
  kAbsorbTypeUKV = 1,  // deprecated, use kAbsorbTypeBMM instead
  kAbsorbTypeBMM = 2,
};

AbsorbWeightsType GetAbsorbWeightsType();

bool IsAbsorbWeightsEnabled();

void SetAbsorbWeightsType(AbsorbWeightsType type);

static const char* enable_absort = std::getenv("ENABLE_COMPRESSED_KV");

}  // namespace ksana_llm
