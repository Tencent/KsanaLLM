/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <Python.h>
#include <stdlib.h>
#include <filesystem>

#include "ksana_llm/models/quant/quant_weight.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/singleton.h"
#include "test.h"

using namespace ksana_llm;

#ifdef ENABLE_CUDA
class QuantWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1);
    // 解析 config.json,初始化 ModelConfig 以及 BlockManager
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    const auto& env = Singleton<Environment>::GetInstance();
    env->ParseConfig(config_path);
    env->batch_scheduler_config_.max_token_len = 256;
    env->ParseModelConfig("/model/hunyuan_large", "/model/hunyuan_large");
    env->GetModelConfig("", model_config);
    BlockManagerConfig block_manager_config;
    env->InitializeBlockManagerConfig();
    env->GetBlockManagerConfig(block_manager_config);
    KLLM_LOG_DEBUG << fmt::format("block_size {}", block_manager_config.device_allocator_config.block_size);

    CUDA_CHECK(cudaGetDevice(&device_rank));
    tensor_manager = std::make_shared<TensorManager>(device_rank, weights_map);
  }

#  ifdef ENABLE_FP8
  void AddFp8WeightTensor(std::string tensor_name, std::vector<size_t>& shape, float value) {
    tensor_manager->AddWeightTensor(tensor_name, shape, TYPE_FP8_E4M3);
    Tensor& tensor = weights_map[tensor_name];
    __nv_fp8_e4m3 fp8_value = __nv_fp8_e4m3(value);
    std::vector<__nv_fp8_e4m3> fp8_data(tensor.GetElementNumber(), fp8_value);
    MemcpyAsync(tensor.GetPtr<void>(), fp8_data.data(), tensor.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[device_rank]);
  }

  void AddFp8ScaleTensor(std::string tensor_name, float value) {
    std::vector<size_t> scale_shape = {1};
    tensor_manager->AddWeightTensor(tensor_name, scale_shape, TYPE_FP32);
    MemcpyAsync(weights_map[tensor_name].GetPtr<void>(), &value, sizeof(float), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[device_rank]);
  }
  // Passing different values into the tensor.
  void AddFp8BlockWiseWeightTensor(std::string tensor_name, std::vector<size_t>& shape, float value1, float value2) {
    tensor_manager->AddWeightTensor(tensor_name, shape, TYPE_FP8_E4M3);
    Tensor& tensor = weights_map[tensor_name];
    __nv_fp8_e4m3 fp8_value1 = __nv_fp8_e4m3(value1);
    std::vector<__nv_fp8_e4m3> fp8_data1((tensor.GetElementNumber() / 4 * 3), fp8_value1);
    MemcpyAsync(tensor.GetPtr<void>(), fp8_data1.data(), (tensor.GetTotalBytes() / 4 * 3), MEMCPY_HOST_TO_DEVICE,
                context_->GetMemoryManageStreams()[device_rank]);
    __nv_fp8_e4m3 fp8_value2 = __nv_fp8_e4m3(value2);
    std::vector<__nv_fp8_e4m3> fp8_data2((tensor.GetElementNumber() / 4), fp8_value2);
    MemcpyAsync(tensor.GetPtr<void>() + (tensor.GetTotalBytes() / 4 * 3), fp8_data2.data(), tensor.GetTotalBytes() / 4,
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[device_rank]);
  }

  void AddFp8BlockWiseScaleTensor(std::string tensor_name, std::vector<size_t>& scale_shape, float value) {
    tensor_manager->AddWeightTensor(tensor_name, scale_shape, TYPE_FP32);
    Tensor& tensor = weights_map[tensor_name];
    std::vector<float> fp32_data(tensor.GetElementNumber(), value);
    MemcpyAsync(weights_map[tensor_name].GetPtr<void>(), fp32_data.data(), tensor.GetTotalBytes(),
                MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[device_rank]);
  }

  std::vector<float> CopyFp8InputScales(std::string tensor_name) {
    Tensor& tensor = weights_map[tensor_name];
    size_t n = tensor.input_scales->shape[0];
    std::vector<float> input_scales(n);
    MemcpyAsync(input_scales.data(), tensor.input_scales->GetPtr<void>(), n * sizeof(float), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[device_rank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[device_rank]);
    return input_scales;
  }

  std::vector<float> CopyFp8WeightScales(std::string tensor_name) {
    Tensor& tensor = weights_map[tensor_name];
    size_t n = tensor.weight_scales->shape[0];
    std::vector<float> weight_scales(n);
    MemcpyAsync(weight_scales.data(), tensor.weight_scales->GetPtr<void>(), n * sizeof(float), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[device_rank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[device_rank]);
    return weight_scales;
  }

#  endif
  std::vector<half> CopyFp16Weight(std::string tensor_name) {
    Tensor& tensor = weights_map[tensor_name];
    std::vector<half> fp16_weight(tensor.GetElementNumber());
    MemcpyAsync(fp16_weight.data(), tensor.GetPtr<void>(), tensor.GetTotalBytes(), MEMCPY_DEVICE_TO_HOST,
                context_->GetMemoryManageStreams()[device_rank]);
    StreamSynchronize(context_->GetMemoryManageStreams()[device_rank]);
    return fp16_weight;
  }

  void TearDown() override {}

 protected:
  ModelConfig model_config;
  std::shared_ptr<Context> context_{nullptr};
  int32_t device_rank = 0;
  std::unordered_map<std::string, Tensor> weights_map;
  std::unordered_map<std::string, DataType> weights_data_type_map;
  std::shared_ptr<TensorManager> tensor_manager;
};

#  ifdef ENABLE_FP8
TEST_F(QuantWeightTest, BindFp8E4m3ScaleOfMoeWeight) {
  int num_layers = model_config.num_layer;
  // model_config.moe_config.num_experts;
  size_t num_experts = 4;
  // model_config.hidden_units;
  size_t hidden_units = 8;
  // model_config.moe_config.moe_inter_size;
  size_t inter_size = 8;

  // create weights
  std::vector<size_t> q_shape = {hidden_units, hidden_units};
  std::vector<size_t> shared_gate_shape = {inter_size, hidden_units};
  std::vector<size_t> shared_down_shape = {hidden_units, inter_size};
  std::vector<size_t> down_shape = {num_experts, hidden_units, inter_size};
  std::vector<size_t> up_gate_shape = {num_experts, 2 * inter_size, hidden_units};
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx);
    if (layer_idx % model_config.cla_share_factor == 0) {
      AddFp8WeightTensor(prefix + ".self_attn.q_proj.weight", q_shape, 1.0f);
    }
    AddFp8WeightTensor(prefix + ".mlp.shared_expert.gate_proj.weight", shared_gate_shape, 1.0f);
    AddFp8WeightTensor(prefix + ".mlp.shared_expert.up_proj.weight", shared_gate_shape, 1.0f);
    AddFp8WeightTensor(prefix + ".mlp.shared_expert.down_proj.weight", shared_down_shape, 1.0f);
    AddFp8WeightTensor(prefix + ".mlp.experts.down_proj.weight", down_shape, 1.0f);
    AddFp8WeightTensor(prefix + ".mlp.experts.up_gate_proj.weight", up_gate_shape, 1.0f);
  }

  // create scales
  std::vector<std::string> scale_names = {"input_scale", "weight_scale"};
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx);
    for (std::string scale_name : scale_names) {
      if (layer_idx % model_config.cla_share_factor == 0) {
        AddFp8ScaleTensor(prefix + ".self_attn.q_proj." + scale_name, 1.0f);
      }
      AddFp8ScaleTensor(prefix + ".mlp.shared_expert.gate_proj." + scale_name, 1.0f);
      AddFp8ScaleTensor(prefix + ".mlp.shared_expert.up_proj." + scale_name, 1.0f);
      AddFp8ScaleTensor(prefix + ".mlp.shared_expert.down_proj." + scale_name, 1.0f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.0.down_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.1.down_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.2.down_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.3.down_proj." + scale_name, 1.0f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.0.gate_proj." + scale_name, 0.25f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.1.gate_proj." + scale_name, 0.25f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.2.gate_proj." + scale_name, 0.25f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.3.gate_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.0.up_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.1.up_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.2.up_proj." + scale_name, 0.5f);
      AddFp8ScaleTensor(prefix + ".mlp.experts.3.up_proj." + scale_name, 1.0f);
    }
  }

  StreamSynchronize(context_->GetMemoryManageStreams()[device_rank]);

  QuantWeight<half> quant_weight =
      QuantWeight<half>(model_config, device_rank, context_, weights_map, weights_data_type_map);
  EXPECT_TRUE(quant_weight.BindFp8E4m3ScaleOfMoeWeight().OK());

  // check scales
  for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    std::string prefix = "model.layers." + std::to_string(layer_idx);

    if (layer_idx % model_config.cla_share_factor == 0) {
      EXPECT_EQ(CopyFp8InputScales(prefix + ".self_attn.q_proj.weight")[0], 1.0f);
      EXPECT_EQ(CopyFp8WeightScales(prefix + ".self_attn.q_proj.weight")[0], 1.0f);
    }

    EXPECT_EQ(CopyFp8InputScales(prefix + ".mlp.shared_expert.gate_proj.weight")[0], 1.0f);
    EXPECT_EQ(CopyFp8WeightScales(prefix + ".mlp.shared_expert.gate_proj.weight")[0], 1.0f);

    EXPECT_EQ(CopyFp8InputScales(prefix + ".mlp.shared_expert.up_proj.weight")[0], 1.0f);
    EXPECT_EQ(CopyFp8WeightScales(prefix + ".mlp.shared_expert.up_proj.weight")[0], 1.0f);

    EXPECT_EQ(CopyFp8InputScales(prefix + ".mlp.shared_expert.down_proj.weight")[0], 1.0f);
    EXPECT_EQ(CopyFp8WeightScales(prefix + ".mlp.shared_expert.down_proj.weight")[0], 1.0f);

    std::vector<float> scales;
    EXPECT_EQ(CopyFp8InputScales(prefix + ".mlp.experts.down_proj.weight")[0], 1.0f);
    scales = CopyFp8WeightScales(prefix + ".mlp.experts.down_proj.weight");
    EXPECT_EQ(scales[0], 0.5f);
    EXPECT_EQ(scales[1], 0.5f);
    EXPECT_EQ(scales[2], 0.5f);
    EXPECT_EQ(scales[3], 1.0f);

    EXPECT_EQ(CopyFp8InputScales(prefix + ".mlp.experts.up_gate_proj.weight")[0], 1.0f);
    scales = CopyFp8WeightScales(prefix + ".mlp.experts.up_gate_proj.weight");
    EXPECT_EQ(scales[0], 0.5f);
    EXPECT_EQ(scales[1], 0.5f);
    EXPECT_EQ(scales[2], 0.5f);
    EXPECT_EQ(scales[3], 1.0f);
  }
}
#    ifdef ENABLE_FP8_TORCH
TEST_F(QuantWeightTest, ProcessMlaFp8E4m3BlockWiseScaleOfWeightTest) {
  model_config.weight_data_type = TYPE_FP16;
  model_config.mla_config.qk_rope_head_dim = 2;
  model_config.mla_config.qk_nope_head_dim = 2;
  model_config.mla_config.v_head_dim = 2;
  model_config.head_num = 1;
  model_config.mla_config.kv_lora_rank = 2;
  model_config.quant_config.weight_block_size = {2, 2};

  std::vector<size_t> kv_b_weight_shape = {
      model_config.head_num * (model_config.mla_config.qk_nope_head_dim + model_config.mla_config.v_head_dim),
      model_config.mla_config.kv_lora_rank};
  // kv_b_proj weight value: [1,1,1,1,1,1,0,0]
  AddFp8BlockWiseWeightTensor("model.layers.0.self_attn.kv_b_proj.weight", kv_b_weight_shape, 1.0f, 0.0f);
  std::vector<size_t> kv_b_scale_shape = {kv_b_weight_shape[0] / model_config.quant_config.weight_block_size[0],
                                          kv_b_weight_shape[1] / model_config.quant_config.weight_block_size[1]};
  // kv_b_proj scale value: [1,1]
  AddFp8BlockWiseScaleTensor("model.layers.0.self_attn.kv_b_proj.weight_scale_inv", kv_b_scale_shape, 1.0f);
  QuantWeight<half> quant_weight =
      QuantWeight<half>(model_config, device_rank, context_, weights_map, weights_data_type_map);
  // new compressed_kv process
  SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbTypeBMM);
  quant_weight.ProcessMlaFp8E4m3BlockWiseScaleOfWeight();
  std::string prefix = "model.layers.0.self_attn.";
  EXPECT_TRUE(weights_map.find(prefix + "kv_b_nope_proj.weight") != weights_map.end());
  EXPECT_TRUE(weights_map.find(prefix + "kv_b_nope_proj.weight_scale_inv") != weights_map.end());
  EXPECT_TRUE(weights_map.find(prefix + "v_head_proj.weight") != weights_map.end());
  EXPECT_TRUE(weights_map.find(prefix + "v_head_proj.weight_scale_inv") != weights_map.end());
  EXPECT_TRUE(weights_map.find(prefix + "w_uk_t.weight") != weights_map.end());
  EXPECT_TRUE(weights_map.find(prefix + "w_uv.weight") != weights_map.end());

  // w_uk_t weight value: [1,1,1,1]
  std::vector<half> w_uk_t_data = CopyFp16Weight("model.layers.0.self_attn.w_uk_t.weight");
  // w_uv weight value: from [1,1,0,0] permute(0,2,1) to [1,0,1,0]
  std::vector<half> w_uv_data = CopyFp16Weight("model.layers.0.self_attn.w_uv.weight");
  EXPECT_EQ(w_uk_t_data.size(), w_uv_data.size());
  for (int i = 0; i < w_uk_t_data.size(); ++i) {
    EXPECT_EQ(static_cast<float>(w_uk_t_data[i]), 1.0f);
    if (i % 2 == 0) {
      EXPECT_EQ(static_cast<float>(w_uv_data[i]), 1.0f);
    } else {
      EXPECT_EQ(static_cast<int>(w_uv_data[i]), 0);
    }
  }
  SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbDisabled);
}
#    endif
#  endif
#endif
