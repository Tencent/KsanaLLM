/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/utils.h"

#include <unordered_map>
#include <vector>

#include "ksana_llm/utils/tensor.h"
#include "tests/test.h"

using namespace ksana_llm;

// 定义一个 UtilsTest 类，继承自 testing::Test
class UtilsTest : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

// 测试 CheckWeightNameMatched 函数（bool full_match 版本）
TEST_F(UtilsTest, CheckWeightNameMatchedBoolTest) {
  std::vector<std::string> name_list = {"model.layers.0.weight", "model.layers.1.bias", "model.embed.weight"};

  // 测试完全匹配模式
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.0.weight", name_list, true));
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.1.bias", name_list, true));
  EXPECT_FALSE(CheckWeightNameMatched("model.layers.0", name_list, true));
  EXPECT_FALSE(CheckWeightNameMatched("model.layers.2.weight", name_list, true));

  // 测试部分匹配模式
  EXPECT_TRUE(CheckWeightNameMatched("model.layers.0.weight", name_list, false));
  EXPECT_TRUE(CheckWeightNameMatched("prefix.model.layers.0.weight.suffix", name_list, false));
  EXPECT_TRUE(CheckWeightNameMatched("model.embed.weight.extra", name_list, false));
  EXPECT_TRUE(CheckWeightNameMatched("test.model.layers.1.bias.test", name_list, false));
  EXPECT_FALSE(CheckWeightNameMatched("model.layers.2.weight", name_list, false));
  EXPECT_FALSE(CheckWeightNameMatched("completely.different.name", name_list, false));

  // 测试空列表
  std::vector<std::string> empty_list;
  EXPECT_FALSE(CheckWeightNameMatched("any.weight", empty_list, true));
  EXPECT_FALSE(CheckWeightNameMatched("any.weight", empty_list, false));

  // 测试空字符串
  EXPECT_FALSE(CheckWeightNameMatched("", name_list, true));
  EXPECT_FALSE(CheckWeightNameMatched("", name_list, false));
}

// 测试 CheckWeightNameMatched 函数（MatchMode 版本）
TEST_F(UtilsTest, CheckWeightNameMatchedMatchModeTest) {
  std::vector<std::string> name_list = {"attention.weight", "mlp.gate", "norm.bias"};

  // 测试 FullMatch 模式
  EXPECT_TRUE(CheckWeightNameMatched("attention.weight", name_list, MatchMode::FullMatch));
  EXPECT_TRUE(CheckWeightNameMatched("mlp.gate", name_list, MatchMode::FullMatch));
  EXPECT_FALSE(CheckWeightNameMatched("attention", name_list, MatchMode::FullMatch));
  EXPECT_FALSE(CheckWeightNameMatched("prefix.attention.weight", name_list, MatchMode::FullMatch));

  // 测试 PartialMatch 模式
  EXPECT_TRUE(CheckWeightNameMatched("attention.weight", name_list, MatchMode::PartialMatch));
  EXPECT_TRUE(CheckWeightNameMatched("layer.attention.weight.extra", name_list, MatchMode::PartialMatch));
  EXPECT_TRUE(CheckWeightNameMatched("prefix.mlp.gate.suffix", name_list, MatchMode::PartialMatch));
  EXPECT_FALSE(CheckWeightNameMatched("different.name", name_list, MatchMode::PartialMatch));

  // 测试边界情况
  std::vector<std::string> single_item = {"test"};
  EXPECT_TRUE(CheckWeightNameMatched("test", single_item, MatchMode::FullMatch));
  EXPECT_TRUE(CheckWeightNameMatched("prefix.test.suffix", single_item, MatchMode::PartialMatch));
}

// 测试 CheckAllWeightsExist 函数
TEST_F(UtilsTest, CheckAllWeightsExistTest) {
  // 创建测试用的权重映射
  std::unordered_map<std::string, Tensor> host_model_weights;
  host_model_weights["weight1"] = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {2, 3}, 0);
  host_model_weights["weight2"] = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP16, {4, 5}, 0);
  host_model_weights["weight3"] = Tensor(MemoryLocation::LOCATION_HOST, TYPE_FP32, {1, 1}, 0);

  // 测试所有权重都存在的情况
  std::vector<std::string> existing_weights = {"weight1", "weight2", "weight3"};
  EXPECT_TRUE(CheckAllWeightsExist(host_model_weights, existing_weights));

  // 测试部分权重存在的情况
  std::vector<std::string> partial_weights = {"weight1", "weight2"};
  EXPECT_TRUE(CheckAllWeightsExist(host_model_weights, partial_weights));

  // 测试有不存在的权重
  std::vector<std::string> missing_weights = {"weight1", "weight4"};
  EXPECT_FALSE(CheckAllWeightsExist(host_model_weights, missing_weights));

  // 测试全部不存在的权重
  std::vector<std::string> all_missing = {"weight4", "weight5"};
  EXPECT_FALSE(CheckAllWeightsExist(host_model_weights, all_missing));

  // 测试空列表（应该返回 true，因为没有需要检查的权重）
  std::vector<std::string> empty_list;
  EXPECT_TRUE(CheckAllWeightsExist(host_model_weights, empty_list));

  // 测试空的权重映射
  std::unordered_map<std::string, Tensor> empty_weights;
  EXPECT_FALSE(CheckAllWeightsExist(empty_weights, existing_weights));
  EXPECT_TRUE(CheckAllWeightsExist(empty_weights, empty_list));
}

// 测试 CheckWeightNameEndMatched 函数
TEST_F(UtilsTest, CheckWeightNameEndMatchedTest) {
  std::vector<std::string> suffix_list = {".weight", ".bias", ".scale"};

  // 测试匹配的后缀
  EXPECT_TRUE(CheckWeightNameEndMatched("model.layer.weight", suffix_list));
  EXPECT_TRUE(CheckWeightNameEndMatched("attention.bias", suffix_list));
  EXPECT_TRUE(CheckWeightNameEndMatched("norm.scale", suffix_list));

  // 测试不匹配的后缀
  EXPECT_FALSE(CheckWeightNameEndMatched("model.layer.param", suffix_list));
  EXPECT_FALSE(CheckWeightNameEndMatched("weight.model", suffix_list));
  EXPECT_FALSE(CheckWeightNameEndMatched("bias.layer", suffix_list));

  // 测试部分匹配但不在末尾的情况
  EXPECT_FALSE(CheckWeightNameEndMatched("model.weight.extra", suffix_list));
  EXPECT_FALSE(CheckWeightNameEndMatched("bias.layer.param", suffix_list));

  // 测试后缀比权重名长的情况
  EXPECT_FALSE(CheckWeightNameEndMatched("w", suffix_list));
  EXPECT_FALSE(CheckWeightNameEndMatched("", suffix_list));

  // 测试完全匹配后缀的情况
  EXPECT_TRUE(CheckWeightNameEndMatched(".weight", suffix_list));
  EXPECT_TRUE(CheckWeightNameEndMatched(".bias", suffix_list));

  // 测试空后缀列表
  std::vector<std::string> empty_suffix;
  EXPECT_FALSE(CheckWeightNameEndMatched("any.weight", empty_suffix));

  // 测试包含空字符串的后缀列表
  std::vector<std::string> suffix_with_empty = {".weight", ""};
  EXPECT_TRUE(CheckWeightNameEndMatched("model.layer.weight", suffix_with_empty));
  EXPECT_TRUE(CheckWeightNameEndMatched("any.string", suffix_with_empty));  // 空字符串总是匹配末尾
}

// 测试 WeightNameReplace 函数
TEST_F(UtilsTest, WeightNameReplaceTest) {
  // 测试基本替换
  EXPECT_EQ(WeightNameReplace("model.layers.0.weight", "layers", "blocks"), "model.blocks.0.weight");
  EXPECT_EQ(WeightNameReplace("attention.qkv.weight", "qkv", "query_key_value"), "attention.query_key_value.weight");

  // 测试多次出现的情况
  EXPECT_EQ(WeightNameReplace("model.layers.0.layers.1.weight", "layers", "blocks"), "model.blocks.0.blocks.1.weight");

  // 测试不存在的匹配
  EXPECT_EQ(WeightNameReplace("model.layers.weight", "attention", "attn"), "model.layers.weight");

  // 测试特殊正则字符的转义
  EXPECT_EQ(WeightNameReplace("model.layers.0.weight", ".", "_"), "model_layers_0_weight");
  EXPECT_EQ(WeightNameReplace("model[0].weight", "[0]", "[1]"), "model[1].weight");
  EXPECT_EQ(WeightNameReplace("model(0).weight", "(0)", "(1)"), "model(1).weight");
  EXPECT_EQ(WeightNameReplace("model{0}.weight", "{0}", "{1}"), "model{1}.weight");

  // 测试包含多个特殊字符的情况
  // 注意：regex_replace 会替换所有匹配的子串，".*+?" 作为字面字符串被替换为 "test"
  EXPECT_EQ(WeightNameReplace("model.*+?.weight", ".*+?", "test"), "modeltest.weight");
  EXPECT_EQ(WeightNameReplace("model.^$|.weight", "^$|", "test"), "model.test.weight");

  // 测试替换为空字符串
  EXPECT_EQ(WeightNameReplace("model.prefix.weight", "prefix.", ""), "model.weight");

  // 测试空字符串匹配（应该在每个位置插入替换字符串）
  std::string result = WeightNameReplace("abc", "", "X");
  EXPECT_EQ(result, "XaXbXcX");

  // 测试完全替换
  EXPECT_EQ(WeightNameReplace("old_name", "old_name", "new_name"), "new_name");

  // 测试反斜杠转义
  EXPECT_EQ(WeightNameReplace("model\\layer.weight", "\\", "/"), "model/layer.weight");

  // 测试复杂的实际场景
  EXPECT_EQ(WeightNameReplace("model.layers.0.self_attn.q_proj.weight", "self_attn", "attention"),
            "model.layers.0.attention.q_proj.weight");
  // 注意：regex_replace 会替换所有 "h"，包括 "weight" 中的 "h"
  EXPECT_EQ(WeightNameReplace("transformer.h.0.mlp.c_fc.weight", "h", "layers"),
            "transformer.layers.0.mlp.c_fc.weiglayerst");
  // 更精确的替换方式：使用 ".h." 来避免替换其他位置的 "h"
  EXPECT_EQ(WeightNameReplace("transformer.h.0.mlp.c_fc.weight", ".h.", ".layers."),
            "transformer.layers.0.mlp.c_fc.weight");
}

// 测试 CutPrefix 函数
TEST_F(UtilsTest, CutPrefixTest) {
  // 测试基本前缀移除
  EXPECT_EQ(CutPrefix("model.layers.0.weight", "model."), "layers.0.weight");
  EXPECT_EQ(CutPrefix("prefix.attention.weight", "prefix."), "attention.weight");

  // 测试前缀不匹配的情况（返回原始字符串）
  EXPECT_EQ(CutPrefix("model.layers.0.weight", "other."), "model.layers.0.weight");
  EXPECT_EQ(CutPrefix("attention.weight", "model."), "attention.weight");

  // 测试空前缀（应该返回原始字符串）
  EXPECT_EQ(CutPrefix("model.layers.weight", ""), "model.layers.weight");

  // 测试空字符串输入
  EXPECT_EQ(CutPrefix("", "prefix."), "");
  EXPECT_EQ(CutPrefix("", ""), "");

  // 测试前缀等于整个字符串的情况
  EXPECT_EQ(CutPrefix("model.weight", "model.weight"), "");

  // 测试前缀比字符串长的情况（不匹配，返回原始字符串）
  EXPECT_EQ(CutPrefix("model", "model.layers.weight"), "model");

  // 测试部分匹配但不是前缀的情况
  EXPECT_EQ(CutPrefix("mymodel.weight", "model"), "mymodel.weight");
  EXPECT_EQ(CutPrefix("prefix_model.weight", "model"), "prefix_model.weight");

  // 测试实际使用场景
  EXPECT_EQ(CutPrefix("transformer.encoder.layer.0.weight", "transformer."), "encoder.layer.0.weight");
  EXPECT_EQ(CutPrefix("bert.embeddings.word_embeddings.weight", "bert."), "embeddings.word_embeddings.weight");
  EXPECT_EQ(CutPrefix("language_model.model.layers.0.self_attn.q_proj.weight", "language_model."),
            "model.layers.0.self_attn.q_proj.weight");

  // 测试多层前缀移除（需要多次调用）
  std::string name = "prefix1.prefix2.weight";
  name = CutPrefix(name, "prefix1.");
  EXPECT_EQ(name, "prefix2.weight");
  name = CutPrefix(name, "prefix2.");
  EXPECT_EQ(name, "weight");

  // 测试特殊字符
  EXPECT_EQ(CutPrefix("model[0].weight", "model[0]."), "weight");
  EXPECT_EQ(CutPrefix("model_v2.layers.weight", "model_v2."), "layers.weight");
}