/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <filesystem>
#include <numeric>

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "tensor.h"

namespace ksana_llm {

inline void InitializeTestEnvironment() {
  std::filesystem::path current_path = __FILE__;
  std::filesystem::path parent_path = current_path.parent_path();
  std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
  std::string config_path = std::filesystem::absolute(config_path_relate).string();
  Singleton<Environment>::GetInstance()->ParseConfig(config_path);
}

inline void ClearTestBlockManager() {}

}  // namespace ksana_llm
