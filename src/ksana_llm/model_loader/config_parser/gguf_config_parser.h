/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "ksana_llm/models/base/model_arch.h"
#include "ksana_llm/utils/gguf_file_utils.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// Config parser for gguf format.
class GGUFConfigParser {
 public:
  Status GetGGUFMeta(const std::string& model_dir, std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                     ModelArchitecture& model_arch);
};

}  // namespace ksana_llm
