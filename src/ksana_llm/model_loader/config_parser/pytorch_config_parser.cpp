/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/config_parser/pytorch_config_parser.h"

#include <fstream>

#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

Status PytorchConfigParser::GetJsonConfig(const std::string& model_dir, nlohmann::json& config_json,
                                          ModelArchitecture& model_arch) {
  // Get config path.
  std::filesystem::path abs_model_dir = std::filesystem::absolute(model_dir);
  std::string model_config_file = abs_model_dir.u8string() + "/config.json";

  if (!std::filesystem::exists(model_config_file)) {
    return Status(RET_INVALID_ARGUMENT, "GetJsonConfig error, config.json not found.");
  }

  std::ifstream config_file(model_config_file);
  if (!config_file.is_open()) {
    return Status(RET_INVALID_ARGUMENT,
                  FormatStr("Load model config config_file %s error.", model_config_file.c_str()));
  }

  config_file >> config_json;
  config_file.close();

  // Get model arch
  std::string model_type = config_json.at("model_type");
  Status status = GetModelArchitectureFromString(model_type, model_arch);
  if (!status.OK()) {
    return status;
  }

  return Status();
}

}  // namespace ksana_llm
