/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/config/quant_config_parser.h"

#include "fmt/core.h"
#include "ksana_llm/utils/logger.h"

namespace ksana_llm {

void ParseGPTQQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config) {
  quant_config.method = QUANT_GPTQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  quant_config.desc_act = config_json.value("desc_act", false);
  quant_config.input_scale = config_json.value("input_scale", false);
  KLLM_LOG_INFO << fmt::format(
      "using quant model, quant method gptq, bits: {}, group_size: {}, desc_act: {}, input_scale:{}", quant_config.bits,
      quant_config.group_size, quant_config.desc_act, quant_config.input_scale);
}

void ParseAWQQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config) {
  if (is_moe) {
    KLLM_THROW(fmt::format("Not support quant_method awq for moe model."));
  }
  quant_config.method = QUANT_AWQ;
  quant_config.bits = config_json.at("bits");
  quant_config.group_size = config_json.at("group_size");
  KLLM_LOG_INFO << fmt::format("using quant model, quant method awq, bits: {}, group_size: {}", quant_config.bits,
                               quant_config.group_size);
}

void ParseFP8QuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config) {
  quant_config.method = QUANT_FP8_E4M3;
  quant_config.is_checkpoint_fp8_serialized = true;
  quant_config.is_activation_scheme_static = (config_json.at("activation_scheme") == "static");
  if (config_json.contains("weight_block_size") && config_json["weight_block_size"].is_array()) {
    quant_config.is_fp8_blockwise = true;
    quant_config.method = QUANT_BLOCK_FP8_E4M3;
    quant_config.weight_block_size = config_json["weight_block_size"].get<std::vector<size_t>>();
  }
  if (is_moe && quant_config.is_fp8_blockwise == false && !quant_config.is_activation_scheme_static) {
    KLLM_THROW(fmt::format("Not support dyanmic fp8 quant_method for moe model."));
  }
  KLLM_LOG_INFO << fmt::format(
      "using quant model, quant method fp8, method type: {}, is_checkpoint_fp8_serialized: {}, "
      "is_activation_scheme_static: {}",
      quant_config.method, quant_config.is_checkpoint_fp8_serialized, quant_config.is_activation_scheme_static);
}

void ParseModelOptQuantConfig(const nlohmann::json &config_json, bool is_moe, QuantConfig &quant_config) {
  std::string algo = config_json.value("quant_algo", "");
  if (algo == "W4A8_AWQ") {
    quant_config.method = QUANT_W4A8_AWQ;
    KLLM_LOG_INFO << "using quant model, quant method W4A8_AWQ";
  } else {
    KLLM_THROW("only support W4A8_AWQ algo in modelopt");
  }
}

nlohmann::json ParseAndConvertQuantConfig(const nlohmann::json &hf_quant_config) {
  // 获取顶层quant_algo
  std::string top_level_quant_algo = hf_quant_config["quantization"].value("quant_algo", "");
  KLLM_CHECK_WITH_INFO(top_level_quant_algo == "MIXED_PRECISION",
                       "only support MIXED_PRECISION in hf_quant_config.json");
  // 参数构建
  nlohmann::json quantization_config;
  quantization_config["quant_method"] = "mixed";
  quantization_config["configs"]["fp8"]["method"] = "fp8";
  quantization_config["configs"]["fp8"]["activation_scheme"] = "dynamic";
  quantization_config["configs"]["fp8"]["fmt"] = "e4m3";
  quantization_config["configs"]["fp8"]["weight_block_size"] = nlohmann::json::array({128, 128});
  quantization_config["configs"]["w4a8"]["method"] = "rtn";
  quantization_config["configs"]["w4a8"]["bits"] = 4;
  quantization_config["configs"]["w4a8"]["group_size"] = 128;
  quantization_config["configs"]["w4a8"]["input_scale"] = true;
  quantization_config["layer_mapping"]["fp8"]["default_config"] = true;
  quantization_config["layer_mapping"]["fp8"]["pattern_layers"] = nlohmann::json::array();
  quantization_config["layer_mapping"]["fp8"]["ignored_layers"] = nlohmann::json::array();
  quantization_config["layer_mapping"]["w4a8"]["default_config"] = false;
  quantization_config["layer_mapping"]["w4a8"]["pattern_layers"] = nlohmann::json::array({".mlp.experts."});
  quantization_config["layer_mapping"]["w4a8"]["ignored_layers"] = nlohmann::json::array();
  return quantization_config;
}

}  // namespace ksana_llm