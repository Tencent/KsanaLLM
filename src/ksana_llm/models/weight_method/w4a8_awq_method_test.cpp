/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/weight_method/w4a8_awq_method.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "tests/test.h"

#include "ksana_llm/models/base/common_model_weight_loader.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/tensor.h"

using namespace ksana_llm;

class W4A8AWQMethodTest : public testing::Test {
 protected:
  void SetUp() override {
    // 初始化环境和上下文
    context_ = std::make_shared<Context>(1, 1, 1);
    env_ = std::make_shared<Environment>();

    // 创建模型配置
    model_config_ = std::make_shared<BaseModelConfig>();

    // 创建 CommonModelWeightLoader
    common_weight_loader_ = std::make_shared<CommonModelWeightLoader>(model_config_, env_, context_);

    // 创建 W4A8AWQMethod 实例
    w4a8_awq_method_ = std::make_shared<W4A8AWQMethod>(common_weight_loader_, tp_);
  }

  void TearDown() override { device_model_weights_.clear(); }

 protected:
  std::shared_ptr<Context> context_{nullptr};
  std::shared_ptr<Environment> env_{nullptr};
  std::shared_ptr<BaseModelConfig> model_config_{nullptr};
  std::shared_ptr<CommonModelWeightLoader> common_weight_loader_{nullptr};
  std::shared_ptr<W4A8AWQMethod> w4a8_awq_method_{nullptr};
  std::unordered_map<std::string, Tensor> device_model_weights_;

  int dev_rank_{0};
  int tp_{2};
  size_t group_size_{128};  // AWQ 量化的 group size
};

// 测试 load_attn_q_k_v_proj 和 process_attn_qkv_proj 的综合测试
TEST_F(W4A8AWQMethodTest, TestLoadAndProcessAttnQKVProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 1024;  // 第一维
  const size_t K = 768;   // 第二维

  // 创建 q_proj, k_proj, v_proj 的权重名称
  std::string q_proj_name = "model.layers.0.self_attn.q_proj.";
  std::string k_proj_name = "model.layers.0.self_attn.k_proj.";
  std::string v_proj_name = "model.layers.0.self_attn.v_proj.";

  // 为每个投影创建所需的权重张量
  // 注意：weight 是 uint8 类型，形状为 (N/2, K)，因为 W4 量化将两个权重打包到一个 uint8 中
  std::vector<std::string> proj_names = {q_proj_name, k_proj_name, v_proj_name};

  for (const auto& proj_name : proj_names) {
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_UINT8, {N / 2, K}, dev_rank_);
    Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {N, K / group_size_}, dev_rank_);
    Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
    Tensor weight_scale_2(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);

    // 测试 load_attn_q_k_v_proj 方法
    w4a8_awq_method_->load_attn_q_k_v_proj(device_model_weights_, proj_name + "weight", weight, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight") != device_model_weights_.end());

    w4a8_awq_method_->load_attn_q_k_v_proj(device_model_weights_, proj_name + "weight_scale", weight_scale, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight_scale") != device_model_weights_.end());

    w4a8_awq_method_->load_attn_q_k_v_proj(device_model_weights_, proj_name + "input_scale", input_scale, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "input_scale") != device_model_weights_.end());

    w4a8_awq_method_->load_attn_q_k_v_proj(device_model_weights_, proj_name + "weight_scale_2", weight_scale_2,
                                           dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight_scale_2") != device_model_weights_.end());
  }

  // 验证加载后的张量形状
  // .weight 和 .weight_scale 应该被 RowPara 模式分割（第一维被分割）
  size_t expected_N_split = N / tp_;
  size_t expected_weight_N_split = (N / 2) / tp_;  // weight 的第一维是 N/2

  for (const auto& proj_name : proj_names) {
    const Tensor& loaded_weight = device_model_weights_[proj_name + "weight"];
    const Tensor& loaded_weight_scale = device_model_weights_[proj_name + "weight_scale"];

    // 验证 .weight 的形状 (N/2 被 tp 分割)
    EXPECT_EQ(loaded_weight.shape[0], expected_weight_N_split);
    EXPECT_EQ(loaded_weight.shape[1], K);
    EXPECT_EQ(loaded_weight.dtype, DataType::TYPE_UINT8);

    // 验证 .weight_scale 的形状 (N 被 tp 分割)
    EXPECT_EQ(loaded_weight_scale.shape[0], expected_N_split);
    EXPECT_EQ(loaded_weight_scale.shape[1], K / group_size_);
    EXPECT_EQ(loaded_weight_scale.dtype, DataType::TYPE_FP32);

    // 验证 .input_scale, .weight_scale_2 没有被分割
    const Tensor& loaded_input_scale = device_model_weights_[proj_name + "input_scale"];
    const Tensor& loaded_weight_scale_2 = device_model_weights_[proj_name + "weight_scale_2"];

    EXPECT_EQ(loaded_input_scale.shape[0], 1);
    EXPECT_EQ(loaded_input_scale.dtype, DataType::TYPE_FP32);

    EXPECT_EQ(loaded_weight_scale_2.shape[0], 1);
    EXPECT_EQ(loaded_weight_scale_2.dtype, DataType::TYPE_FP32);
  }

  // 测试 process_attn_qkv_proj 方法
  std::string weight_prefix_name = "model.layers.0.self_attn.q_proj.";
  w4a8_awq_method_->process_attn_qkv_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证合并后的 qkv 权重存在
  std::string qkv_proj_name = "model.layers.0.self_attn.query_key_value.";
  EXPECT_TRUE(device_model_weights_.find(qkv_proj_name + "weight") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(qkv_proj_name + "weight_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(qkv_proj_name + "input_scale") != device_model_weights_.end());

  // 验证原始的 q/k/v 权重的 input_scale 已被删除
  EXPECT_TRUE(device_model_weights_.find(q_proj_name + "input_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(k_proj_name + "input_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(v_proj_name + "input_scale") == device_model_weights_.end());

  // 验证 weight_scale_2 已被删除
  EXPECT_TRUE(device_model_weights_.find(qkv_proj_name + "weight_scale_2") == device_model_weights_.end());

  // 验证合并后的张量形状
  const Tensor& merged_qkv_weight = device_model_weights_[qkv_proj_name + "weight"];
  const Tensor& merged_qkv_weight_scale = device_model_weights_[qkv_proj_name + "weight_scale"];

  // 验证权重已经过转换和转置
  EXPECT_EQ(merged_qkv_weight.shape.size(), 2);
  EXPECT_EQ(merged_qkv_weight_scale.shape.size(), 2);

  // 验证量化模式和 layout 信息已设置到 context
  const WeightStatus& weight_status = context_->GetWeightStatus(qkv_proj_name + "weight");
  EXPECT_EQ(weight_status.quant_mode, QuantMode::QUANT_W4A8_AWQ);
  EXPECT_TRUE(weight_status.layout.find("k") != weight_status.layout.end());
  EXPECT_TRUE(weight_status.layout.find("n") != weight_status.layout.end());

  // 验证 input_scales, weight_scales 指针已绑定
  EXPECT_TRUE(merged_qkv_weight.input_scales != nullptr);
  EXPECT_TRUE(merged_qkv_weight.weight_scales != nullptr);

  // Q/K/V 权重没有 pre_quant_scale，所以 pre_quant_scales 指针应该为 nullptr
  EXPECT_TRUE(merged_qkv_weight.pre_quant_scales == nullptr);

  // 验证 weight_scale 的数据类型已转换为 FP16
  EXPECT_EQ(merged_qkv_weight_scale.dtype, DataType::TYPE_FP16);
#endif
}

// 测试 load_mlp_gate_up_proj 和 process_mlp_gate_up_proj 的综合测试
TEST_F(W4A8AWQMethodTest, TestLoadAndProcessMlpGateUpProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 2048;  // 第一维
  const size_t K = 1024;  // 第二维

  // 创建 gate_proj, up_proj 的权重名称
  std::string gate_proj_name = "model.layers.0.mlp.gate_proj.";
  std::string up_proj_name = "model.layers.0.mlp.up_proj.";

  // 为每个投影创建所需的权重张量
  // 注意：weight 是 uint8 类型，形状为 (N/2, K)，因为 W4 量化将两个权重打包到一个 uint8 中
  std::vector<std::string> proj_names = {gate_proj_name, up_proj_name};

  for (const auto& proj_name : proj_names) {
    Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_UINT8, {N / 2, K}, dev_rank_);
    Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {N, K / group_size_}, dev_rank_);
    Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
    Tensor weight_scale_2(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);

    // 测试 load_mlp_gate_up_proj 方法
    w4a8_awq_method_->load_mlp_gate_up_proj(device_model_weights_, proj_name + "weight", weight, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight") != device_model_weights_.end());

    w4a8_awq_method_->load_mlp_gate_up_proj(device_model_weights_, proj_name + "weight_scale", weight_scale, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight_scale") != device_model_weights_.end());

    w4a8_awq_method_->load_mlp_gate_up_proj(device_model_weights_, proj_name + "input_scale", input_scale, dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "input_scale") != device_model_weights_.end());

    w4a8_awq_method_->load_mlp_gate_up_proj(device_model_weights_, proj_name + "weight_scale_2", weight_scale_2,
                                            dev_rank_);
    EXPECT_TRUE(device_model_weights_.find(proj_name + "weight_scale_2") != device_model_weights_.end());
  }

  // 验证加载后的张量形状
  // .weight 和 .weight_scale 应该被 RowPara 模式分割（第一维被分割）
  size_t expected_N_split = N / tp_;
  size_t expected_weight_N_split = (N / 2) / tp_;  // weight 的第一维是 N/2

  for (const auto& proj_name : proj_names) {
    const Tensor& loaded_weight = device_model_weights_[proj_name + "weight"];
    const Tensor& loaded_weight_scale = device_model_weights_[proj_name + "weight_scale"];

    // 验证 .weight 的形状 (N/2 被 tp 分割)
    EXPECT_EQ(loaded_weight.shape[0], expected_weight_N_split);
    EXPECT_EQ(loaded_weight.shape[1], K);
    EXPECT_EQ(loaded_weight.dtype, DataType::TYPE_UINT8);

    // 验证 .weight_scale 的形状 (N 被 tp 分割)
    EXPECT_EQ(loaded_weight_scale.shape[0], expected_N_split);
    EXPECT_EQ(loaded_weight_scale.shape[1], K / group_size_);
    EXPECT_EQ(loaded_weight_scale.dtype, DataType::TYPE_FP32);

    // 验证 .input_scale, .weight_scale_2 没有被分割
    const Tensor& loaded_input_scale = device_model_weights_[proj_name + "input_scale"];
    const Tensor& loaded_weight_scale_2 = device_model_weights_[proj_name + "weight_scale_2"];

    EXPECT_EQ(loaded_input_scale.shape[0], 1);
    EXPECT_EQ(loaded_input_scale.dtype, DataType::TYPE_FP32);

    EXPECT_EQ(loaded_weight_scale_2.shape[0], 1);
    EXPECT_EQ(loaded_weight_scale_2.dtype, DataType::TYPE_FP32);
  }

  // 测试 process_mlp_gate_up_proj 方法
  std::string weight_prefix_name = "model.layers.0.mlp.gate_proj.";
  w4a8_awq_method_->process_mlp_gate_up_proj(device_model_weights_, weight_prefix_name, dev_rank_);

  // 验证合并后的 gate_up 权重存在
  std::string gate_up_proj_name = "model.layers.0.mlp.gate_up_proj.";
  EXPECT_TRUE(device_model_weights_.find(gate_up_proj_name + "weight") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(gate_up_proj_name + "weight_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(gate_up_proj_name + "input_scale") != device_model_weights_.end());

  // 验证原始的 gate/up 权重的 input_scale 已被删除
  EXPECT_TRUE(device_model_weights_.find(gate_proj_name + "input_scale") == device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(up_proj_name + "input_scale") == device_model_weights_.end());

  // 验证 weight_scale_2 已被删除
  EXPECT_TRUE(device_model_weights_.find(gate_up_proj_name + "weight_scale_2") == device_model_weights_.end());

  // 验证合并后的张量形状
  const Tensor& merged_gate_up_weight = device_model_weights_[gate_up_proj_name + "weight"];
  const Tensor& merged_gate_up_weight_scale = device_model_weights_[gate_up_proj_name + "weight_scale"];

  // 验证权重已经过转换和转置
  EXPECT_EQ(merged_gate_up_weight.shape.size(), 2);
  EXPECT_EQ(merged_gate_up_weight_scale.shape.size(), 2);

  // 验证量化模式和 layout 信息已设置到 context
  const WeightStatus& gate_up_weight_status = context_->GetWeightStatus(gate_up_proj_name + "weight");
  EXPECT_EQ(gate_up_weight_status.quant_mode, QuantMode::QUANT_W4A8_AWQ);
  EXPECT_TRUE(gate_up_weight_status.layout.find("k") != gate_up_weight_status.layout.end());
  EXPECT_TRUE(gate_up_weight_status.layout.find("n") != gate_up_weight_status.layout.end());

  // 验证 input_scales, weight_scales 指针已绑定
  EXPECT_TRUE(merged_gate_up_weight.input_scales != nullptr);
  EXPECT_TRUE(merged_gate_up_weight.weight_scales != nullptr);

  // Gate/Up 权重没有 pre_quant_scale，所以 pre_quant_scales 指针应该为 nullptr
  EXPECT_TRUE(merged_gate_up_weight.pre_quant_scales == nullptr);

  // 验证 weight_scale 的数据类型已转换为 FP16
  EXPECT_EQ(merged_gate_up_weight_scale.dtype, DataType::TYPE_FP16);
#endif
}

// 测试 load_attn_o_proj 和 process_attn_o_proj 的综合测试
TEST_F(W4A8AWQMethodTest, TestLoadAndProcessAttnOProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 768;   // 第一维
  const size_t K = 1024;  // 第二维

  // 创建 o_proj 的权重名称
  std::string o_proj_name = "model.layers.0.self_attn.o_proj.";

  // 创建所需的权重张量
  // 注意：weight 是 uint8 类型，形状为 (N/2, K)，因为 W4 量化将两个权重打包到一个 uint8 中
  Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_UINT8, {N / 2, K}, dev_rank_);
  Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {N, K / group_size_}, dev_rank_);
  Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
  Tensor weight_scale_2(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
  Tensor pre_quant_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {K}, dev_rank_);

  // 测试 load_attn_o_proj 方法
  w4a8_awq_method_->load_attn_o_proj(device_model_weights_, o_proj_name + "weight", weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight") != device_model_weights_.end());

  w4a8_awq_method_->load_attn_o_proj(device_model_weights_, o_proj_name + "weight_scale", weight_scale, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight_scale") != device_model_weights_.end());

  w4a8_awq_method_->load_attn_o_proj(device_model_weights_, o_proj_name + "input_scale", input_scale, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "input_scale") != device_model_weights_.end());

  w4a8_awq_method_->load_attn_o_proj(device_model_weights_, o_proj_name + "weight_scale_2", weight_scale_2, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight_scale_2") != device_model_weights_.end());

  w4a8_awq_method_->load_attn_o_proj(device_model_weights_, o_proj_name + "pre_quant_scale", pre_quant_scale,
                                     dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "pre_quant_scale") != device_model_weights_.end());

  // 验证加载后的张量形状
  // .weight 和 .weight_scale 应该被 ColPara 模式分割（第二维被分割）
  size_t expected_K_split = K / tp_;
  size_t expected_weight_N_split = N / 2;  // weight 的第一维是 N/2，不被分割

  const Tensor& loaded_weight = device_model_weights_[o_proj_name + "weight"];
  const Tensor& loaded_weight_scale = device_model_weights_[o_proj_name + "weight_scale"];

  // 验证 .weight 的形状 (第二维 K 被 tp 分割)
  EXPECT_EQ(loaded_weight.shape[0], expected_weight_N_split);
  EXPECT_EQ(loaded_weight.shape[1], expected_K_split);
  EXPECT_EQ(loaded_weight.dtype, DataType::TYPE_UINT8);

  // 验证 .weight_scale 的形状 (第二维 K/group_size 被 tp 分割)
  EXPECT_EQ(loaded_weight_scale.shape[0], N);
  EXPECT_EQ(loaded_weight_scale.shape[1], (K / group_size_) / tp_);
  EXPECT_EQ(loaded_weight_scale.dtype, DataType::TYPE_FP32);

  // 验证 .input_scale, .weight_scale_2 没有被分割
  const Tensor& loaded_input_scale = device_model_weights_[o_proj_name + "input_scale"];
  const Tensor& loaded_weight_scale_2 = device_model_weights_[o_proj_name + "weight_scale_2"];

  EXPECT_EQ(loaded_input_scale.shape[0], 1);
  EXPECT_EQ(loaded_input_scale.dtype, DataType::TYPE_FP32);

  EXPECT_EQ(loaded_weight_scale_2.shape[0], 1);
  EXPECT_EQ(loaded_weight_scale_2.dtype, DataType::TYPE_FP32);

  // 验证 .pre_quant_scale 被 RowPara 模式分割（第一维被分割）
  const Tensor& loaded_pre_quant_scale = device_model_weights_[o_proj_name + "pre_quant_scale"];
  EXPECT_EQ(loaded_pre_quant_scale.shape[0], expected_K_split);
  EXPECT_EQ(loaded_pre_quant_scale.dtype, DataType::TYPE_BF16);

  // 测试 process_attn_o_proj 方法
  w4a8_awq_method_->process_attn_o_proj(device_model_weights_, o_proj_name, dev_rank_);

  // 验证处理后的权重存在
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "input_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "pre_quant_scale") != device_model_weights_.end());

  // 验证 weight_scale_2 已被删除
  EXPECT_TRUE(device_model_weights_.find(o_proj_name + "weight_scale_2") == device_model_weights_.end());

  // 验证处理后的张量形状
  const Tensor& processed_weight = device_model_weights_[o_proj_name + "weight"];
  const Tensor& processed_weight_scale = device_model_weights_[o_proj_name + "weight_scale"];

  // 验证权重已经过转换和转置
  EXPECT_EQ(processed_weight.shape.size(), 2);
  EXPECT_EQ(processed_weight_scale.shape.size(), 2);

  // 验证量化模式和 layout 信息已设置到 context
  const WeightStatus& o_proj_weight_status = context_->GetWeightStatus(o_proj_name + "weight");
  EXPECT_EQ(o_proj_weight_status.quant_mode, QuantMode::QUANT_W4A8_AWQ);
  EXPECT_TRUE(o_proj_weight_status.layout.find("k") != o_proj_weight_status.layout.end());
  EXPECT_TRUE(o_proj_weight_status.layout.find("n") != o_proj_weight_status.layout.end());

  // 验证 input_scales, weight_scales, pre_quant_scales 指针已绑定
  EXPECT_TRUE(processed_weight.input_scales != nullptr);
  EXPECT_TRUE(processed_weight.weight_scales != nullptr);
  EXPECT_TRUE(processed_weight.pre_quant_scales != nullptr);

  // 验证 weight_scale 的数据类型已转换为 FP16
  EXPECT_EQ(processed_weight_scale.dtype, DataType::TYPE_FP16);
#endif
}

// 测试 load_mlp_down_proj 和 process_mlp_down_proj 的综合测试
TEST_F(W4A8AWQMethodTest, TestLoadAndProcessMlpDownProj) {
#ifdef ENABLE_CUDA
  // 定义测试参数 - 使用不同的 N 和 K 来避免取巧
  const size_t N = 1024;  // 第一维
  const size_t K = 2048;  // 第二维

  // 创建 down_proj 的权重名称
  std::string down_proj_name = "model.layers.0.mlp.down_proj.";

  // 创建所需的权重张量
  // 注意：weight 是 uint8 类型，形状为 (N/2, K)，因为 W4 量化将两个权重打包到一个 uint8 中
  Tensor weight(MemoryLocation::LOCATION_HOST, DataType::TYPE_UINT8, {N / 2, K}, dev_rank_);
  Tensor weight_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {N, K / group_size_}, dev_rank_);
  Tensor input_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
  Tensor weight_scale_2(MemoryLocation::LOCATION_HOST, DataType::TYPE_FP32, {1}, dev_rank_);
  Tensor pre_quant_scale(MemoryLocation::LOCATION_HOST, DataType::TYPE_BF16, {K}, dev_rank_);

  // 测试 load_mlp_down_proj 方法
  w4a8_awq_method_->load_mlp_down_proj(device_model_weights_, down_proj_name + "weight", weight, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight") != device_model_weights_.end());

  w4a8_awq_method_->load_mlp_down_proj(device_model_weights_, down_proj_name + "weight_scale", weight_scale, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight_scale") != device_model_weights_.end());

  w4a8_awq_method_->load_mlp_down_proj(device_model_weights_, down_proj_name + "input_scale", input_scale, dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "input_scale") != device_model_weights_.end());

  w4a8_awq_method_->load_mlp_down_proj(device_model_weights_, down_proj_name + "weight_scale_2", weight_scale_2,
                                       dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight_scale_2") != device_model_weights_.end());

  w4a8_awq_method_->load_mlp_down_proj(device_model_weights_, down_proj_name + "pre_quant_scale", pre_quant_scale,
                                       dev_rank_);
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "pre_quant_scale") != device_model_weights_.end());

  // 验证加载后的张量形状
  // .weight 和 .weight_scale 应该被 ColPara 模式分割（第二维被分割）
  size_t expected_K_split = K / tp_;
  size_t expected_weight_N_split = N / 2;  // weight 的第一维是 N/2，不被分割

  const Tensor& loaded_weight = device_model_weights_[down_proj_name + "weight"];
  const Tensor& loaded_weight_scale = device_model_weights_[down_proj_name + "weight_scale"];

  // 验证 .weight 的形状 (第二维 K 被 tp 分割)
  EXPECT_EQ(loaded_weight.shape[0], expected_weight_N_split);
  EXPECT_EQ(loaded_weight.shape[1], expected_K_split);
  EXPECT_EQ(loaded_weight.dtype, DataType::TYPE_UINT8);

  // 验证 .weight_scale 的形状 (第二维 K/group_size 被 tp 分割)
  EXPECT_EQ(loaded_weight_scale.shape[0], N);
  EXPECT_EQ(loaded_weight_scale.shape[1], (K / group_size_) / tp_);
  EXPECT_EQ(loaded_weight_scale.dtype, DataType::TYPE_FP32);

  // 验证 .input_scale, .weight_scale_2 没有被分割
  const Tensor& loaded_input_scale = device_model_weights_[down_proj_name + "input_scale"];
  const Tensor& loaded_weight_scale_2 = device_model_weights_[down_proj_name + "weight_scale_2"];

  EXPECT_EQ(loaded_input_scale.shape[0], 1);
  EXPECT_EQ(loaded_input_scale.dtype, DataType::TYPE_FP32);

  EXPECT_EQ(loaded_weight_scale_2.shape[0], 1);
  EXPECT_EQ(loaded_weight_scale_2.dtype, DataType::TYPE_FP32);

  // 验证 .pre_quant_scale 被 RowPara 模式分割（第一维被分割）
  const Tensor& loaded_pre_quant_scale = device_model_weights_[down_proj_name + "pre_quant_scale"];
  EXPECT_EQ(loaded_pre_quant_scale.shape[0], expected_K_split);
  EXPECT_EQ(loaded_pre_quant_scale.dtype, DataType::TYPE_BF16);

  // 测试 process_mlp_down_proj 方法
  w4a8_awq_method_->process_mlp_down_proj(device_model_weights_, down_proj_name, dev_rank_);

  // 验证处理后的权重存在
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "input_scale") != device_model_weights_.end());
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "pre_quant_scale") != device_model_weights_.end());

  // 验证 weight_scale_2 已被删除
  EXPECT_TRUE(device_model_weights_.find(down_proj_name + "weight_scale_2") == device_model_weights_.end());

  // 验证处理后的张量形状
  const Tensor& processed_weight = device_model_weights_[down_proj_name + "weight"];
  const Tensor& processed_weight_scale = device_model_weights_[down_proj_name + "weight_scale"];

  // 验证权重已经过转换和转置
  EXPECT_EQ(processed_weight.shape.size(), 2);
  EXPECT_EQ(processed_weight_scale.shape.size(), 2);

  // 验证量化模式和 layout 信息已设置到 context
  const WeightStatus& down_proj_weight_status = context_->GetWeightStatus(down_proj_name + "weight");
  EXPECT_EQ(down_proj_weight_status.quant_mode, QuantMode::QUANT_W4A8_AWQ);
  EXPECT_TRUE(down_proj_weight_status.layout.find("k") != down_proj_weight_status.layout.end());
  EXPECT_TRUE(down_proj_weight_status.layout.find("n") != down_proj_weight_status.layout.end());

  // 验证 input_scales, weight_scales, pre_quant_scales 指针已绑定
  EXPECT_TRUE(processed_weight.input_scales != nullptr);
  EXPECT_TRUE(processed_weight.weight_scales != nullptr);
  EXPECT_TRUE(processed_weight.pre_quant_scales != nullptr);

  // 验证 weight_scale 的数据类型已转换为 FP16
  EXPECT_EQ(processed_weight_scale.dtype, DataType::TYPE_FP16);
#endif
}