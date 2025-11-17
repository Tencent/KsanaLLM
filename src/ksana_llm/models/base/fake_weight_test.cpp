/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/models/base/fake_weight_for_test.h"
#include "ksana_llm/models/common/common_config.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/tensor_test_helper.h"

#include "tests/test.h"

using namespace ksana_llm;

// 定义一个 LlamaTest 类,继承自 testing::Test
class FakeWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    nlohmann::json config_json = nlohmann::json::parse("{}");
    PrepareCommonModelAttributes(config_json, model_config_);
    runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    model_config_.weight_data_type = TYPE_FP16;
  }

  void TearDown() override { ClearTestBlockManager(); }

 protected:
  int rank_ = 0;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
};

TEST_F(FakeWeightTest, FakeWeightBasic) {
  FakeWeight fake_weight(rank_);

  // 测试创建HOST内存的权重
  const std::vector<size_t> shape{2, 3};
  EXPECT_TRUE(fake_weight.CreateWeight("test.weight", shape, TYPE_FP32, MemoryLocation::LOCATION_HOST));

  // 验证权重属性
  Tensor weight = fake_weight.GetModelWeights("test.weight");
  EXPECT_EQ(std::vector<size_t>(weight.shape), shape);
  EXPECT_EQ(weight.dtype, TYPE_FP32);
  EXPECT_EQ(weight.location, MemoryLocation::LOCATION_HOST);
}

TEST_F(FakeWeightTest, GetCacheFolder) {
  FakeWeight fake_weight(rank_);
  std::string cache_folder = fake_weight.GetCacheFolder();
  EXPECT_TRUE(cache_folder.find("cached_model_") != std::string::npos);
}

bool IsVectorSame(const std::vector<size_t>& a, const std::vector<size_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// Helper function to compare tensor values with epsilon tolerance
bool AreTensorValuesSame(const Tensor& a, const Tensor& b) {
  if (a.GetElementNumber() != b.GetElementNumber() || a.dtype != b.dtype) {
    KLLM_LOG_ERROR << "Tensor dimensions or data types don't match";
    return false;
  }

  // Use a small epsilon for floating point comparisons
  const float epsilon = 1e-6f;
  bool all_values_same = true;

  switch (a.dtype) {
    case TYPE_FP32: {
      float* data_a = a.GetPtr<float>();
      float* data_b = b.GetPtr<float>();
      for (size_t i = 0; i < a.GetElementNumber(); ++i) {
        if (std::abs(data_a[i] - data_b[i]) > epsilon) {
          KLLM_LOG_ERROR << "Values differ at index " << i << ": " << data_a[i] << " vs " << data_b[i]
                         << " (diff: " << std::abs(data_a[i] - data_b[i]) << ")";
          all_values_same = false;
        }
      }
      break;
    }
    case TYPE_FP16: {
      float16* data_a = a.GetPtr<float16>();
      float16* data_b = b.GetPtr<float16>();
      for (size_t i = 0; i < a.GetElementNumber(); ++i) {
#ifdef ENABLE_ACL
        float val_a = aclFloat16ToFloat(data_a[i]);
        float val_b = aclFloat16ToFloat(data_b[i]);
#else
        float val_a = static_cast<float>(data_a[i]);
        float val_b = static_cast<float>(data_b[i]);
#endif
        if (std::abs(val_a - val_b) > epsilon) {
          KLLM_LOG_ERROR << "Values differ at index " << i << ": " << val_a << " vs " << val_b
                         << " (diff: " << std::abs(val_a - val_b) << ")";
          all_values_same = false;
        }
      }
      break;
    }
    case TYPE_BF16: {
      bfloat16* data_a = a.GetPtr<bfloat16>();
      bfloat16* data_b = b.GetPtr<bfloat16>();
      for (size_t i = 0; i < a.GetElementNumber(); ++i) {
        float val_a = static_cast<float>(data_a[i]);
        float val_b = static_cast<float>(data_b[i]);
        if (std::abs(val_a - val_b) > epsilon) {
          KLLM_LOG_ERROR << "Values differ at index " << i << ": " << val_a << " vs " << val_b
                         << " (diff: " << std::abs(val_a - val_b) << ")";
          all_values_same = false;
        }
      }
      break;
    }
    default:
      KLLM_LOG_ERROR << "Unsupported data type";
      return false;
  }
  return all_values_same;
}

TEST_F(FakeWeightTest, DefaultWeightValueInitializerConsistency) {
  // Test that DefaultWeightValueInitializer produces consistent results across multiple initializations
  DefaultWeightValueInitializer initializer1, initializer2;

  // Create multiple tensors with different weight names to test all initialization cases
  const size_t elements = 100;
  const std::vector<std::string> weight_names = {
      "test.normal.weight",     // Normal distribution case
      "test.layernorm.weight",  // All 1s case
      "test.bias"               // All 0s case
  };

  // For each weight name, create two tensors and verify they have identical values
  for (const auto& weight_name : weight_names) {
    KLLM_LOG_INFO << "Testing consistency for weight: " << weight_name;

    // Create first tensor
    Tensor tensor1(MemoryLocation::LOCATION_HOST, TYPE_FP32, {elements}, rank_);
    initializer1.InitValues(weight_name, rank_, tensor1);

    // Create second tensor
    Tensor tensor2(MemoryLocation::LOCATION_HOST, TYPE_FP32, {elements}, rank_);
    initializer2.InitValues(weight_name, rank_, tensor2);

    // Verify tensors have identical values
    EXPECT_TRUE(AreTensorValuesSame(tensor1, tensor2)) << "Values differ for weight: " << weight_name;
  }
}

TEST_F(FakeWeightTest, FakeLlamaWeightStructure) {
  FakeSimpleWeight llama_weight(model_config_, runtime_config_, rank_, false, false, false);

  // 验证基础权重形状
  const auto& embed_shape = llama_weight.GetModelWeights("model.embed_tokens.weight").shape;
  EXPECT_TRUE(IsVectorSame(embed_shape, std::vector<size_t>({model_config_.vocab_size, model_config_.hidden_units})));

  const auto& lm_head_shape = llama_weight.GetModelWeights("lm_head.weight").shape;
  EXPECT_TRUE(IsVectorSame(lm_head_shape, std::vector<size_t>({model_config_.hidden_units, model_config_.vocab_size})));

  // 验证层结构
  const uint32_t head_dim = model_config_.hidden_units / model_config_.head_num;

  for (uint32_t layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
    const std::string prefix = fmt::format("model.layers.{}", layer_idx);

    // 验证注意力层
    const auto& qkv_shape = llama_weight.GetModelWeights(prefix + ".self_attn.query_key_value.weight").shape;
    EXPECT_TRUE(
        IsVectorSame(qkv_shape, std::vector<size_t>({model_config_.hidden_units, head_dim * model_config_.head_num})));

    const auto& o_proj_shape = llama_weight.GetModelWeights(prefix + ".self_attn.o_proj.weight").shape;
    EXPECT_TRUE(IsVectorSame(o_proj_shape,
                             std::vector<size_t>({head_dim * model_config_.head_num, model_config_.hidden_units})));

    // 验证MLP层
    const auto& gate_shape = llama_weight.GetModelWeights(prefix + ".mlp.gate_proj.weight").shape;
    EXPECT_TRUE(IsVectorSame(gate_shape, std::vector<size_t>({model_config_.hidden_units, model_config_.inter_size})));

    const auto& up_shape = llama_weight.GetModelWeights(prefix + ".mlp.up_proj.weight").shape;
    EXPECT_TRUE(IsVectorSame(up_shape, std::vector<size_t>({model_config_.hidden_units, model_config_.inter_size})));

    const auto& down_shape = llama_weight.GetModelWeights(prefix + ".mlp.down_proj.weight").shape;
    EXPECT_TRUE(IsVectorSame(down_shape, std::vector<size_t>({model_config_.inter_size, model_config_.hidden_units})));

    // 验证LayerNorm
    const auto& input_ln_shape = llama_weight.GetModelWeights(prefix + ".input_layernorm.weight").shape;
    EXPECT_TRUE(IsVectorSame(input_ln_shape, std::vector<size_t>({model_config_.hidden_units})));

    const auto& post_ln_shape = llama_weight.GetModelWeights(prefix + ".post_attention_layernorm.weight").shape;
    EXPECT_TRUE(IsVectorSame(post_ln_shape, std::vector<size_t>({model_config_.hidden_units})));
  }
}
