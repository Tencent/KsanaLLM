/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/gguf_file_utils.h"

#include <stdexcept>

#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

std::any GetValueFromGGUFMeta(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                              const std::string& key) {
  auto it = gguf_meta.find(key);
  if (it != gguf_meta.end()) {
    return it->second.value;
  }
  throw std::runtime_error(FormatStr("The key %s not found in gguf meta.", key.c_str()));
}

// Get value from meta, return default value if not found.
std::any GetValueFromGGUFMeta(const std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                              const std::string& key, const std::any& default_value) {
  auto it = gguf_meta.find(key);
  if (it != gguf_meta.end()) {
    return it->second.value;
  } else {
    return default_value;
  }
}

}  // namespace ksana_llm
