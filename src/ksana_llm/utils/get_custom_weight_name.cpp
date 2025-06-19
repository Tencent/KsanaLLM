/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <fstream>
#include <iostream>
#include <regex>

#include "ksana_llm/utils/get_custom_weight_name.h"
#include "ksana_llm/utils/optional_file.h"

#include "nlohmann/json.hpp"

namespace ksana_llm {

std::string GetWeightMapPath(const std::string& model_path, const std::string& model_type,
                             ModelFileFormat model_file_format) {
  auto optional_file = Singleton<OptionalFile>::GetInstance();
  std::string weight_map_suffix = "_weight_map.json";
  if (model_file_format == GGUF) {
    weight_map_suffix = "_gguf_weight_map.json";
  }
  std::string& weight_path = optional_file->GetOptionalFile(model_path, "weight_map", model_type + weight_map_suffix);
  if (weight_path == "") {  // searching weight_map under ksana_llm
    std::filesystem::path filePath = __FILE__;
    std::filesystem::path directory = filePath.parent_path();
    const std::string kRelativePath = "../python";
    std::string ksana_weight_map_path = directory / kRelativePath / "weight_map";
    weight_path = optional_file->GetOptionalFile(ksana_weight_map_path, "weight_map", model_type + weight_map_suffix);
  }
  return weight_path;
}

Status GetCustomNameList(const std::string& model_path, const std::string& model_type,
                         const std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list,
                         ModelFileFormat model_file_format) {
  // In the default case, the tensor name is consistent with the weight name.
  custom_name_list.assign(weight_name_list.begin(), weight_name_list.end());

  // Search for the optional_weight_map.json file
  const std::string weight_path = GetWeightMapPath(model_path, model_type, model_file_format);

  if (weight_path == "") {
    return Status();
  }

  nlohmann::json weight_map_json;
  std::ifstream file(weight_path);
  if (!file.is_open()) {
    KLLM_LOG_ERROR << fmt::format("Load weight map json: {} error.", weight_path) << std::endl;
    return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Load weight map json: {} error.", weight_path));
  } else {
    file >> weight_map_json;
    file.close();
  }
  for (size_t idx = 0; idx < weight_name_list.size(); ++idx) {
    std::string weight_name = weight_name_list[idx];
    for (auto it = weight_map_json.begin(); it != weight_map_json.end(); ++it) {
      std::regex pattern(it.key());
      std::string format = it.value();
      if (std::regex_search(weight_name, pattern)) {
        custom_name_list[idx] = std::regex_replace(weight_name, pattern, format);
        break;
      }
    }
  }
  return Status();
}

}  // namespace ksana_llm
