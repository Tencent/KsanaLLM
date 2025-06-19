/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <filesystem>
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {

inline std::string GetTestConfigFile() {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  return std::filesystem::absolute(config_path_relate).string();
}

inline std::string GetTestTPConfigFile() {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm_tp.yaml";
  return std::filesystem::absolute(config_path_relate).string();
}

}  // namespace ksana_llm
