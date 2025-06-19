/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>

#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Model file format.
enum class ModelFormat { PYTORCH_BIN, PYTORCH_SAFETENSOR, GGUF };

}  // namespace ksana_llm
