/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/json_config_utils.h"

#include "fmt/format.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

DataType GetModelDataType(const nlohmann::json &config_json) {
  std::string dtype_raw = config_json.value("torch_dtype", "float16");
  std::string dtype_str = dtype_raw;

  // unify it to lower case
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (dtype_str == "float16") {
    return DataType::TYPE_FP16;
  } else if (dtype_str == "bfloat16") {
    return DataType::TYPE_BF16;
  } else {
    KLLM_THROW(fmt::format("Not supported model data type: {}.", dtype_str));
  }
}

}  // namespace ksana_llm
