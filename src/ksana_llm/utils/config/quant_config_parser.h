/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#pragma once

#include "ksana_llm/utils/config/model_config_parser.h"
#include "nlohmann/json.hpp"

namespace ksana_llm {

// TODO(jinxcwu) 需要抽象成类

// Parse GPTQ quantization configuration
void ParseGPTQQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config);

// Parse AWQ quantization configuration
void ParseAWQQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config);

// Parse FP8 quantization configuration
void ParseFP8QuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config);

// Parse W4A8_AWQ quantization configuration
void ParseModelOptQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config);

// Parse and convert mixed precision quantization configuration from hf_quant_config.json
nlohmann::json ParseAndConvertQuantConfig(const nlohmann::json &hf_quant_config);

}  // namespace ksana_llm