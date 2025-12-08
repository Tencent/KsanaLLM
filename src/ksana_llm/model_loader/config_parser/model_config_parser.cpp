/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/model_loader/config_parser/model_config_parser.h"

#include "ksana_llm/model_loader/config_parser/gguf_config_parser.h"
#include "ksana_llm/model_loader/config_parser/model_config_parser_factory.h"
#include "ksana_llm/model_loader/config_parser/pytorch_config_parser.h"
#include "ksana_llm/model_loader/model_loader_utils.h"
#include "ksana_llm/models/base/model_format.h"
#include "ksana_llm/utils/ret_code.h"

namespace ksana_llm {

Status ModelConfigParser::ParseModelConfig(const std::string& model_dir,
                                           const ParallelismBasicConfig& parallel_basic_config,
                                           std::shared_ptr<BaseModelConfig>& model_config) {
  // Get model file format.
  ModelFormat model_format;
  Status status = GetModelFormat(model_dir, model_format);
  if (!status.OK()) {
    return status;
  }

  nlohmann::json config_json;
  ModelArchitecture model_arch;
  std::unordered_map<std::string, NewGGUFMetaValue> gguf_meta;

  // Get model arch and config.
  if (model_format == ModelFormat::PYTORCH_BIN || model_format == ModelFormat::PYTORCH_SAFETENSOR) {
    PytorchConfigParser pytorch_config_parser;
    status = pytorch_config_parser.GetJsonConfig(model_dir, config_json, model_arch);
    if (!status.OK()) {
      return status;
    }
  } else if (model_format == ModelFormat::GGUF) {
    GGUFConfigParser gguf_config_parser;
    Status status = gguf_config_parser.GetGGUFMeta(model_dir, gguf_meta, model_arch);
    if (!status.OK()) {
      return status;
    }
  } else {
    return Status(RET_INVALID_ARGUMENT, "Not supported model file format.");
  }

  // Create model parser.
  std::shared_ptr<BaseModelConfigParser> model_config_parser;
  status = ModelConfigParserFactory::CreateModelConfigParser(model_arch, model_config_parser);
  if (!status.OK()) {
    return status;
  }

  if (model_format == ModelFormat::PYTORCH_BIN || model_format == ModelFormat::PYTORCH_SAFETENSOR) {
    status = model_config_parser->ParseModelConfig(config_json, parallel_basic_config, model_dir, model_config);
  } else if (model_format == ModelFormat::GGUF) {
    status = model_config_parser->ParseModelConfig(gguf_meta, model_config);
  }
  if (!status.OK()) {
    return status;
  }

  // Set base info.
  model_config->model_dir = model_dir;
  model_config->model_arch = model_arch;
  model_config->model_format = model_format;

  return Status();
}

}  // namespace ksana_llm
