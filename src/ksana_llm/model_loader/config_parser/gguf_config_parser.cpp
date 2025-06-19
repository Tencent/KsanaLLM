/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/config_parser/gguf_config_parser.h"

#include "ksana_llm/model_loader/file_loader/gguf_file_loader.h"
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

Status GGUFConfigParser::GetGGUFMeta(const std::string& model_dir,
                                     std::unordered_map<std::string, NewGGUFMetaValue>& gguf_meta,
                                     ModelArchitecture& model_arch) {
  std::vector<std::string> model_file_list;
  Status status = GetModelFileList(model_dir, model_file_list);
  if (!status.OK()) {
    return status;
  }

  status = FilterModelFormatFiles(ModelFormat::GGUF, model_file_list);
  if (!status.OK()) {
    return status;
  }

  if (model_file_list.empty()) {
    return Status(RET_INVALID_ARGUMENT, "No valid gguf file found.");
  }

  // Only the first file has required meta data.
  GGUFFileLoader gguf_file_loader(model_file_list.front());
  status = gguf_file_loader.GetMetaDict(gguf_meta);
  if (!status.OK()) {
    return status;
  }

  // Get model arch
  constexpr const char* MODEL_ARCH_KEY = "general.architecture";
  auto it = gguf_meta.find(MODEL_ARCH_KEY);
  if (it == gguf_meta.end()) {
    return Status(RET_INVALID_ARGUMENT, "No general.architecture found in GGUF meta.");
  }

  std::string model_type = std::any_cast<std::string>(it->second.value);
  if (model_type != "llama") {
    return Status(RET_INVALID_ARGUMENT, "Only support llama architecture in GGUF format.");
  }

  status = GetModelArchitectureFromString(model_type, model_arch);
  if (!status.OK()) {
    return status;
  }

  return Status();
}

}  // namespace ksana_llm
