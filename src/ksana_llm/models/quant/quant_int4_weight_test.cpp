/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/quant/quant_weight.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

using namespace ksana_llm;

struct ArrayInfo {
  void* weight_ptr = nullptr;
  std::vector<size_t> weight_shape;
  DataType weight_data_type;
  std::string tensor_name;
};

// 定义一个 QuantWeightLoadTest 类,继承自 testing::Test
class QuantWeightLoadTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config_.moe_config.moe_inter_size = 128;
    runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    runtime_config_.parallel_basic_config.expert_parallel_size = 1;
    runtime_config_.parallel_basic_config.expert_world_size = 1;
    runtime_config_.parallel_basic_config.moe_tensor_para_size = 1;
    model_config_.moe_config.num_experts = 4;
    model_config_.hidden_units = 128;
    model_config_.quant_config.bits = 4;  // fake bits
    model_config_.quant_config.group_size = 128;
    model_config_.weight_data_type = TYPE_FP16;
    model_config_.num_layer = 1;
    model_config_.mla_config.q_lora_rank = 128;
    model_config_.mla_config.kv_lora_rank = 128;
    model_config_.mla_config.qk_rope_head_dim = 64;
    model_config_.mla_config.qk_nope_head_dim = 128;
    model_config_.mla_config.v_head_dim = 128;
    model_config_.head_num = runtime_config_.parallel_basic_config.tensor_parallel_size * 2;
    runtime_config_.inter_data_type = model_config_.weight_data_type;
    context_ = std::make_shared<Context>(1, 1, 1);

    PipelineConfig pipeline_config;
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = 0;
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);
  }

  void TearDown() override {}

 protected:
  int rank = 0;
  size_t pack_factor = 8;
  std::vector<ArrayInfo> gate_qweight_arrays_;
  std::vector<ArrayInfo> up_qweight_arrays_;
  std::vector<ArrayInfo> down_qweight_arrays_;
  std::vector<ArrayInfo> mla_weight_arrays_;
  std::vector<ArrayInfo> gate_g_idx_arrays_;
  std::vector<ArrayInfo> up_g_idx_arrays_;
  std::vector<ArrayInfo> down_g_idx_arrays_;
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_{nullptr};
  std::unordered_map<std::string, Tensor> weights_map_;
  std::unordered_map<std::string, DataType> weights_data_type_map_;

  void InitOriginMoeArrays() {
    gate_qweight_arrays_.resize(model_config_.moe_config.num_experts);
    up_qweight_arrays_.resize(model_config_.moe_config.num_experts);
    down_qweight_arrays_.resize(model_config_.moe_config.num_experts);
    const size_t gate_up_array_size =
        model_config_.hidden_units / pack_factor * model_config_.moe_config.moe_inter_size;
    const size_t down_array_size = model_config_.moe_config.moe_inter_size / pack_factor * model_config_.hidden_units;
    for (size_t i = 0; i < model_config_.moe_config.num_experts; i++) {
      int32_t* gate_qweight_array = new int32_t[gate_up_array_size];
      int32_t* up_qweight_array = new int32_t[gate_up_array_size];
      int32_t* down_qweight_array = new int32_t[down_array_size];
      for (size_t j = 0; j < (model_config_.moe_config.moe_inter_size / pack_factor * model_config_.hidden_units);
           ++j) {
        gate_qweight_array[j] = i + 1;
        up_qweight_array[j] = i + 1;
        down_qweight_array[j] = i + 1;
      }
      gate_qweight_arrays_[i].weight_ptr = reinterpret_cast<void*>(gate_qweight_array);
      gate_qweight_arrays_[i].weight_shape = {model_config_.hidden_units / pack_factor,
                                              model_config_.moe_config.moe_inter_size};
      gate_qweight_arrays_[i].weight_data_type = TYPE_INT32;
      gate_qweight_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".gate_proj.qweight";

      up_qweight_arrays_[i].weight_ptr = reinterpret_cast<void*>(up_qweight_array);
      up_qweight_arrays_[i].weight_shape = {model_config_.hidden_units / pack_factor,
                                            model_config_.moe_config.moe_inter_size};
      up_qweight_arrays_[i].weight_data_type = TYPE_INT32;
      up_qweight_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".up_proj.qweight";

      down_qweight_arrays_[i].weight_ptr = reinterpret_cast<void*>(down_qweight_array);
      down_qweight_arrays_[i].weight_shape = {model_config_.moe_config.moe_inter_size / pack_factor,
                                              model_config_.hidden_units};
      down_qweight_arrays_[i].weight_data_type = TYPE_INT32;
      down_qweight_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".down_proj.qweight";
    }

    gate_g_idx_arrays_.resize(model_config_.moe_config.num_experts);
    up_g_idx_arrays_.resize(model_config_.moe_config.num_experts);
    down_g_idx_arrays_.resize(model_config_.moe_config.num_experts);
    const size_t gate_up_g_idx_array_size = model_config_.hidden_units;
    const size_t down_g_idx_array_size = model_config_.moe_config.moe_inter_size;
    for (size_t i = 0; i < model_config_.moe_config.num_experts; i++) {
      int32_t* gate_g_idx_array = new int32_t[gate_up_g_idx_array_size];
      int32_t* up_g_idx_array = new int32_t[gate_up_g_idx_array_size];
      int32_t* down_g_idx_array = new int32_t[down_g_idx_array_size];
      for (size_t j = 0; j < gate_up_g_idx_array_size; ++j) {
        gate_g_idx_array[j] = i + 1;
        up_g_idx_array[j] = i + 1;
      }
      for (size_t j = 0; j < down_g_idx_array_size; ++j) {
        down_g_idx_array[j] = i + 1;
      }
      gate_g_idx_arrays_[i].weight_ptr = reinterpret_cast<void*>(gate_g_idx_array);
      gate_g_idx_arrays_[i].weight_shape = {gate_up_g_idx_array_size};
      gate_g_idx_arrays_[i].weight_data_type = TYPE_INT32;
      gate_g_idx_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".gate_proj.g_idx";

      up_g_idx_arrays_[i].weight_ptr = reinterpret_cast<void*>(up_g_idx_array);
      up_g_idx_arrays_[i].weight_shape = {gate_up_g_idx_array_size};
      up_g_idx_arrays_[i].weight_data_type = TYPE_INT32;
      up_g_idx_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".up_proj.g_idx";

      down_g_idx_arrays_[i].weight_ptr = reinterpret_cast<void*>(down_g_idx_array);
      down_g_idx_arrays_[i].weight_shape = {down_g_idx_array_size};
      down_g_idx_arrays_[i].weight_data_type = TYPE_INT32;
      down_g_idx_arrays_[i].tensor_name = "model.layers.0.mlp.experts." + std::to_string(i) + ".down_proj.g_idx";
    }
  }

  template <typename T>
  void TestGPTQMoeQuantWeightload() {
    // Init quant weight
    std::shared_ptr<QuantWeight<T>> quant_weight_solver = std::make_shared<QuantWeight<T>>(
        model_config_, runtime_config_, rank, context_, weights_map_, weights_data_type_map_);

    // load moe quant weight
    for (size_t i = 0; i < model_config_.moe_config.num_experts; i++) {
      quant_weight_solver->LoadMoeIntQuantWeight(
          gate_qweight_arrays_[i].tensor_name, gate_qweight_arrays_[i].weight_shape,
          gate_qweight_arrays_[i].weight_data_type, gate_qweight_arrays_[i].weight_ptr);
      quant_weight_solver->LoadMoeIntQuantWeight(up_qweight_arrays_[i].tensor_name, up_qweight_arrays_[i].weight_shape,
                                                 up_qweight_arrays_[i].weight_data_type,
                                                 up_qweight_arrays_[i].weight_ptr);
      quant_weight_solver->LoadMoeIntQuantWeight(
          down_qweight_arrays_[i].tensor_name, down_qweight_arrays_[i].weight_shape,
          down_qweight_arrays_[i].weight_data_type, down_qweight_arrays_[i].weight_ptr);

      quant_weight_solver->LoadMoeIntQuantWeight(gate_g_idx_arrays_[i].tensor_name, gate_g_idx_arrays_[i].weight_shape,
                                                 gate_g_idx_arrays_[i].weight_data_type,
                                                 gate_g_idx_arrays_[i].weight_ptr);
      quant_weight_solver->LoadMoeIntQuantWeight(up_g_idx_arrays_[i].tensor_name, up_g_idx_arrays_[i].weight_shape,
                                                 up_g_idx_arrays_[i].weight_data_type, up_g_idx_arrays_[i].weight_ptr);
      quant_weight_solver->LoadMoeIntQuantWeight(down_g_idx_arrays_[i].tensor_name, down_g_idx_arrays_[i].weight_shape,
                                                 down_g_idx_arrays_[i].weight_data_type,
                                                 down_g_idx_arrays_[i].weight_ptr);
    }
  }
};

TEST_F(QuantWeightLoadTest, GPTQMoeQuantWeightloadTest) {
  InitOriginMoeArrays();
  TestGPTQMoeQuantWeightload<half>();
  size_t num_experts = model_config_.moe_config.num_experts;
  size_t moe_inter_size = model_config_.moe_config.moe_inter_size;
  size_t tp = runtime_config_.parallel_basic_config.tensor_parallel_size;
  size_t hidden_units = model_config_.hidden_units;

  std::string gate_up_name = "model.layers.0.mlp.experts.up_gate_proj.weight";
  EXPECT_TRUE(weights_map_.find(gate_up_name) != weights_map_.end());
  std::vector<size_t> gate_up_shape = {num_experts, moe_inter_size * 2 / tp, hidden_units / pack_factor * 4};
  EXPECT_EQ(static_cast<std::vector<size_t>>(weights_map_[gate_up_name].shape), gate_up_shape);
  EXPECT_EQ(weights_map_[gate_up_name].dtype, TYPE_UINT8);

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchTypeFromDataType(TYPE_UINT8));
  torch::Tensor gate_up_tensor =
      torch::from_blob(weights_map_[gate_up_name].GetPtr<void>(),
                       std::vector<int64_t>(gate_up_shape.begin(), gate_up_shape.end()), options);
  torch::Tensor gate_up_cpu_tensor = gate_up_tensor.to(torch::kCPU);
  // compare value
  for (size_t i = 0; i < gate_up_shape[0]; i++) {
    for (size_t j = 0; j < gate_up_shape[1]; j++) {
      for (size_t k = 0; k < gate_up_shape[2]; k += 4) {
        EXPECT_EQ(gate_up_cpu_tensor[i][j][k].item<uint8_t>(), static_cast<uint8_t>(i + 1));
      }
      for (size_t k = 0; k < gate_up_shape[2]; k++) {
        if (k % 4 == 0) {
          continue;
        }
        EXPECT_EQ(gate_up_cpu_tensor[i][j][k].item<uint8_t>(), static_cast<uint8_t>(0));
      }
    }
  }

  std::string down_name = "model.layers.0.mlp.experts.down_proj.weight";
  EXPECT_TRUE(weights_map_.find(down_name) != weights_map_.end());
  std::vector<size_t> down_shape = {num_experts, hidden_units, moe_inter_size / pack_factor * 4 / tp};
  EXPECT_EQ(static_cast<std::vector<size_t>>(weights_map_[down_name].shape), down_shape);
  EXPECT_EQ(weights_map_[down_name].dtype, TYPE_UINT8);

  torch::Tensor down_tensor = torch::from_blob(weights_map_[down_name].GetPtr<void>(),
                                               std::vector<int64_t>(down_shape.begin(), down_shape.end()), options);
  torch::Tensor down_cpu_tensor = down_tensor.to(torch::kCPU);
  // compare value
  for (size_t i = 0; i < down_shape[0]; i++) {
    for (size_t j = 0; j < down_shape[1]; j++) {
      for (size_t k = 0; k < down_shape[2]; k += 4) {
        EXPECT_EQ(down_cpu_tensor[i][j][k].item<uint8_t>(), static_cast<uint8_t>(i + 1));
      }
      for (size_t k = 0; k < down_shape[2]; k++) {
        if (k % 4 == 0) {
          continue;
        }
        EXPECT_EQ(down_cpu_tensor[i][j][k].item<uint8_t>(), static_cast<uint8_t>(0));
      }
    }
  }

  std::string gate_up_g_idx_name = "model.layers.0.mlp.experts.up_gate_proj.g_idx";
  EXPECT_TRUE(weights_map_.find(gate_up_g_idx_name) != weights_map_.end());
  std::vector<size_t> gate_up_g_idx_shape = {num_experts, hidden_units};
  EXPECT_EQ(static_cast<std::vector<size_t>>(weights_map_[gate_up_g_idx_name].shape), gate_up_g_idx_shape);
  EXPECT_EQ(weights_map_[gate_up_g_idx_name].dtype, TYPE_INT32);

  auto int32_options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchTypeFromDataType(TYPE_INT32));
  torch::Tensor gate_up_g_idx_tensor =
      torch::from_blob(weights_map_[gate_up_g_idx_name].GetPtr<void>(),
                       std::vector<int64_t>(gate_up_g_idx_shape.begin(), gate_up_g_idx_shape.end()), int32_options);
  torch::Tensor gate_up_g_idx_cpu_tensor = gate_up_g_idx_tensor.to(torch::kCPU);
  // compare value
  for (size_t i = 0; i < gate_up_g_idx_shape[0]; i++) {
    for (size_t j = 0; j < gate_up_g_idx_shape[1]; j++) {
      EXPECT_EQ(gate_up_g_idx_cpu_tensor[i][j].item<int32_t>(), static_cast<int32_t>(i + 1));
    }
  }

  std::string down_g_idx_name = "model.layers.0.mlp.experts.down_proj.g_idx";
  EXPECT_TRUE(weights_map_.find(down_g_idx_name) != weights_map_.end());
  std::vector<size_t> down_g_idx_shape = {num_experts, moe_inter_size / tp};
  EXPECT_EQ(static_cast<std::vector<size_t>>(weights_map_[down_g_idx_name].shape), down_g_idx_shape);
  EXPECT_EQ(weights_map_[down_g_idx_name].dtype, TYPE_INT32);

  torch::Tensor down_g_idx_tensor =
      torch::from_blob(weights_map_[down_g_idx_name].GetPtr<void>(),
                       std::vector<int64_t>(down_g_idx_shape.begin(), down_g_idx_shape.end()), int32_options);
  torch::Tensor down_g_idx_cpu_tensor = down_g_idx_tensor.to(torch::kCPU);
  // compare value
  for (size_t i = 0; i < down_g_idx_shape[0]; i++) {
    for (size_t j = 0; j < down_g_idx_shape[1]; j++) {
      EXPECT_EQ(down_g_idx_cpu_tensor[i][j].item<int32_t>(), static_cast<int32_t>(i + 1));
    }
  }
}

TEST_F(QuantWeightLoadTest, CommonDeQuantTest) {
  model_config_.quant_config.method = QUANT_GPTQ;
  model_config_.quant_config.backend = MACHETE_BACKEND;

  std::shared_ptr<QuantWeight<half>> quant_weight_solver = std::make_shared<QuantWeight<half>>(
      model_config_, runtime_config_, rank, context_, weights_map_, weights_data_type_map_);
  // Create quant weight
  torch::Tensor qweight_tensor = torch::zeros(
      {model_config_.hidden_units / pack_factor, static_cast<int64_t>(model_config_.moe_config.moe_inter_size)},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  std::string qweight_name = "quant.weight";
  quant_weight_solver->AddWeightFromTorchTensor(qweight_name, qweight_tensor);
  // Create quant scales
  torch::Tensor scales_tensor = torch::full({model_config_.hidden_units / model_config_.quant_config.group_size,
                                             static_cast<int64_t>(model_config_.moe_config.moe_inter_size)},
                                            1.0, torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
  std::string scales_name = "quant.scales";
  quant_weight_solver->AddWeightFromTorchTensor(scales_name, scales_tensor);
  weights_map_[qweight_name].scales = &weights_map_[scales_name];
  // Dequant
  Tensor dequant_weight = quant_weight_solver->CommonDequantTensor(qweight_name);

  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchTypeFromDataType(TYPE_FP16));
  torch::Tensor dequant_tensor =
      torch::from_blob(dequant_weight.GetPtr<void>(), {dequant_weight.shape[0], dequant_weight.shape[1]}, options);
  torch::Tensor dequant_cpu_tensor = dequant_tensor.to(torch::kCPU);
  // Check dequant result
  EXPECT_TRUE(torch::all(dequant_cpu_tensor == -8).item<bool>());
}
