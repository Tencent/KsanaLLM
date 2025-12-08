/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/device_types.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

// Get data type from json config.
DataType GetModelDataType(const nlohmann::json &config_json);

// Read JSON from file.
nlohmann::json ReadJsonFromFile(const std::string &file_path);

}  // namespace ksana_llm
