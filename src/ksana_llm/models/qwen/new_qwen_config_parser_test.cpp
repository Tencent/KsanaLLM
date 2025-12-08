/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen/new_qwen_config_parser.h"

#include <memory>
#include <string>

#include "ksana_llm/models/qwen/new_qwen_config.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/singleton.h"
#include "nlohmann/json.hpp"
#include "test.h"

using namespace ksana_llm;

class NewQwenConfigParserTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();
    parser_ = std::make_unique<NewQwenConfigParser>();
  }

  void TearDown() override { parser_.reset(); }

  // Helper: Create realistic config based on Qwen3-32B-W4A8-AWQ
  nlohmann::json CreateBaseConfig() {
    return {{"architectures", {"Qwen3ForCausalLM"}},
            {"bos_token_id", 151643},
            {"eos_token_id", 151645},
            {"head_dim", 128},
            {"hidden_size", 5120},
            {"intermediate_size", 25600},
            {"max_position_embeddings", 40960},
            {"num_attention_heads", 64},
            {"num_hidden_layers", 64},
            {"num_key_value_heads", 8},
            {"rms_norm_eps", 1e-06},
            {"rope_theta", 1000000},
            {"tie_word_embeddings", false},
            {"vocab_size", 151936}};
  }

  // Helper: Parse config and return NewQwenConfig
  std::shared_ptr<NewQwenConfig> ParseConfig(const nlohmann::json& config_json) {
    ParallelismBasicConfig parallel_basic_config;
    parallel_basic_config.tensor_parallel_size = 1;
    std::shared_ptr<BaseModelConfig> model_config;
    Status status = parser_->ParseModelConfig(config_json, parallel_basic_config, "", model_config);
    EXPECT_TRUE(status.OK());
    return std::dynamic_pointer_cast<NewQwenConfig>(model_config);
  }

  std::unique_ptr<NewQwenConfigParser> parser_;
};

// Test basic model config parsing with and without quantization
TEST_F(NewQwenConfigParserTest, ParseModelConfigBasic) {
  // Test with quantization
  nlohmann::json config_with_quant = CreateBaseConfig();
  config_with_quant["quantization_config"] = {{"quant_method", "modelopt"},
                                              {"quant_algo", "W4A8_AWQ"},
                                              {"producer", {{"name", "modelopt"}, {"version", "0.33.1"}}}};

  auto qwen_config = ParseConfig(config_with_quant);
  ASSERT_NE(qwen_config, nullptr);

  // Verify basic parameters
  EXPECT_EQ(qwen_config->head_num, 64);
  EXPECT_EQ(qwen_config->num_key_value_heads, 8);
  EXPECT_EQ(qwen_config->inter_size, 25600);
  EXPECT_EQ(qwen_config->vocab_size, 151936);
  EXPECT_EQ(qwen_config->num_layer, 64);
  EXPECT_EQ(qwen_config->hidden_units, 5120);
  EXPECT_FLOAT_EQ(qwen_config->rope_theta, 1000000.0f);
  EXPECT_FLOAT_EQ(qwen_config->layernorm_eps, 1e-6);
  EXPECT_EQ(qwen_config->start_id, 151643);
  EXPECT_EQ(qwen_config->end_id, 151645);
  EXPECT_EQ(qwen_config->max_position_embeddings, 40960);
  EXPECT_EQ(qwen_config->size_per_head, 128);
  EXPECT_FALSE(qwen_config->tie_word_embeddings);
  EXPECT_FALSE(qwen_config->is_visual);

  // Verify quantization
  EXPECT_TRUE(qwen_config->is_quant);
  EXPECT_EQ(qwen_config->quant_config.method, QUANT_W4A8_AWQ);

  // Test without quantization
  nlohmann::json config_no_quant = CreateBaseConfig();
  auto qwen_config_no_quant = ParseConfig(config_no_quant);
  EXPECT_FALSE(qwen_config_no_quant->is_quant);
}

// Test special field handling: head_dim, tie_word_embeddings, visual
TEST_F(NewQwenConfigParserTest, ParseModelConfigSpecialFields) {
  // Test missing head_dim
  nlohmann::json config = CreateBaseConfig();
  config.erase("head_dim");
  auto qwen_config = ParseConfig(config);
  EXPECT_EQ(qwen_config->size_per_head, 0);

  // Test tie_word_embeddings present
  config = CreateBaseConfig();
  config["tie_word_embeddings"] = true;
  qwen_config = ParseConfig(config);
  EXPECT_TRUE(qwen_config->tie_word_embeddings);
  EXPECT_TRUE(qwen_config->exist_tie_embeddings_param);

  // Test tie_word_embeddings absent
  config = CreateBaseConfig();
  config.erase("tie_word_embeddings");
  qwen_config = ParseConfig(config);
  EXPECT_FALSE(qwen_config->exist_tie_embeddings_param);
  EXPECT_FALSE(qwen_config->tie_word_embeddings);

  // Test visual field
  config = CreateBaseConfig();
  config["visual"] = {{"config", "test"}};
  qwen_config = ParseConfig(config);
  EXPECT_TRUE(qwen_config->is_visual);
}

// Test quantization config parsing
TEST_F(NewQwenConfigParserTest, ParseQuantConfig) {
  std::shared_ptr<NewQwenConfig> qwen_config = std::make_shared<NewQwenConfig>();

  // Test W4A8_AWQ quantization
  nlohmann::json config_w4a8 = {
      {"quantization_config",
       {{"quant_method", "modelopt"}, {"quant_algo", "W4A8_AWQ"}, {"producer", {{"name", "modelopt"}}}}}};
  Status status = parser_->ParseQuantConfig(config_w4a8, qwen_config, "", "");
  EXPECT_TRUE(status.OK());
  EXPECT_TRUE(qwen_config->is_quant);
  EXPECT_EQ(qwen_config->quant_config.method, QUANT_W4A8_AWQ);

  // Test no quantization
  qwen_config = std::make_shared<NewQwenConfig>();
  nlohmann::json config_no_quant = {{"model_type", "qwen3"}};
  status = parser_->ParseQuantConfig(config_no_quant, qwen_config, "", "");
  EXPECT_TRUE(status.OK());
  EXPECT_FALSE(qwen_config->is_quant);

  // Test unsupported quant_method
  qwen_config = std::make_shared<NewQwenConfig>();
  nlohmann::json config_unsupported = {{"quantization_config", {{"quant_method", "unsupported"}}}};
  EXPECT_THROW(parser_->ParseQuantConfig(config_unsupported, qwen_config, "", ""), std::exception);

  // Test unsupported quant_algo in modelopt
  qwen_config = std::make_shared<NewQwenConfig>();
  nlohmann::json config_bad_algo = {
      {"quantization_config", {{"quant_method", "modelopt"}, {"quant_algo", "UNSUPPORTED"}}}};
  EXPECT_THROW(parser_->ParseQuantConfig(config_bad_algo, qwen_config, "", ""), std::exception);

  // Test missing quant_algo in modelopt
  qwen_config = std::make_shared<NewQwenConfig>();
  nlohmann::json config_missing_algo = {{"quantization_config", {{"quant_method", "modelopt"}}}};
  EXPECT_THROW(parser_->ParseQuantConfig(config_missing_algo, qwen_config, "", ""), std::exception);
}

// Test default values and comprehensive config
TEST_F(NewQwenConfigParserTest, ParseModelConfigDefaultsAndComprehensive) {
  // Test with minimal config (should use defaults)
  nlohmann::json minimal_config = {{"model_type", "qwen3"}};
  auto qwen_config = ParseConfig(minimal_config);
  EXPECT_EQ(qwen_config->head_num, 64);
  EXPECT_EQ(qwen_config->vocab_size, 151936);
  EXPECT_FLOAT_EQ(qwen_config->rope_theta, 1000000.0f);

  // Test comprehensive config with custom values
  nlohmann::json comprehensive_config = {
      {"bos_token_id", 100},
      {"eos_token_id", 101},
      {"pad_token_id", 102},
      {"head_dim", 256},
      {"hidden_size", 1024},
      {"intermediate_size", 4096},
      {"max_position_embeddings", 8192},
      {"num_attention_heads", 16},
      {"num_hidden_layers", 24},
      {"num_key_value_heads", 4},
      {"rms_norm_eps", 1e-5},
      {"rope_theta", 10000.0},
      {"tie_word_embeddings", true},
      {"vocab_size", 50000},
      {"visual", {{"config", "test"}}},
      {"quantization_config",
       {{"quant_method", "modelopt"}, {"quant_algo", "W4A8_AWQ"}, {"producer", {{"name", "modelopt"}}}}}};

  qwen_config = ParseConfig(comprehensive_config);
  EXPECT_EQ(qwen_config->start_id, 100);
  EXPECT_EQ(qwen_config->end_id, 101);
  EXPECT_EQ(qwen_config->pad_id, 102);
  EXPECT_EQ(qwen_config->size_per_head, 256);
  EXPECT_EQ(qwen_config->hidden_units, 1024);
  EXPECT_EQ(qwen_config->inter_size, 4096);
  EXPECT_EQ(qwen_config->max_position_embeddings, 8192);
  EXPECT_EQ(qwen_config->head_num, 16);
  EXPECT_EQ(qwen_config->num_layer, 24);
  EXPECT_EQ(qwen_config->num_key_value_heads, 4);
  EXPECT_FLOAT_EQ(qwen_config->layernorm_eps, 1e-5);
  EXPECT_FLOAT_EQ(qwen_config->rope_theta, 10000.0f);
  EXPECT_TRUE(qwen_config->tie_word_embeddings);
  EXPECT_EQ(qwen_config->vocab_size, 50000);
  EXPECT_TRUE(qwen_config->is_visual);
  EXPECT_TRUE(qwen_config->is_quant);
  EXPECT_EQ(qwen_config->quant_config.method, QUANT_W4A8_AWQ);
}