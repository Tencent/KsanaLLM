/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// This method will remove unnecessary files in dir.
Status GetModelFileList(const std::string& model_dir, std::vector<std::string>& model_file_list);

// Get model format from model dir.
Status GetModelFormat(const std::string& model_dir, ModelFormat& model_format);

// Filter model file List for specific file format.
Status FilterModelFormatFiles(ModelFormat model_format, std::vector<std::string>& model_file_list);

// Get python path used to search weight maps.
Status GetPythonWeightMapPath(std::vector<std::string>& paths);

int GetLayerIdxFromName(const std::string& weight_name);
}  // namespace ksana_llm
