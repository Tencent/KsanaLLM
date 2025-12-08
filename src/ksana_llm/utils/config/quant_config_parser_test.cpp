/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/config/quant_config_parser.h"
#include <gtest/gtest.h>

namespace ksana_llm {

TEST(QuantConfigParser, TestParseAndConvertQuantConfig) {
  // 构建数据
  std::string json_str = R"(
    {
        "quantization": {
            "quant_algo": "MIXED_PRECISION",
            "kv_cache_quant_algo": null
        }
    })";
  nlohmann::json hf_quant_config = nlohmann::json::parse(json_str);

  // 解析
  nlohmann::json quantization_config = ParseAndConvertQuantConfig(hf_quant_config);

  // 校验
  EXPECT_EQ(quantization_config["quant_method"], "mixed");
  EXPECT_EQ(quantization_config["configs"]["fp8"]["method"], "fp8");
  EXPECT_EQ(quantization_config["configs"]["w4a8"]["method"], "rtn");
  EXPECT_EQ(quantization_config["layer_mapping"]["fp8"]["default_config"], true);
  EXPECT_EQ(quantization_config["layer_mapping"]["w4a8"]["default_config"], false);
}

}  // namespace ksana_llm
