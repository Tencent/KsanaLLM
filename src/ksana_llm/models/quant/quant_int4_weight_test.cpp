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
    model_config_.tensor_para_size = 1;
    model_config_.expert_para_size = 1;
    model_config_.expert_world_size = 1;
    model_config_.moe_tensor_para_size = 1;
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
    model_config_.head_num = model_config_.tensor_para_size * 2;
    context_ = std::make_shared<Context>(1, 1);

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
  ModelConfig model_config_;
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
  }

  void InitOriginMlaArrays() {
    size_t q_lora_rank = static_cast<size_t>(model_config_.mla_config.q_lora_rank);
    size_t kv_lora_rank = static_cast<size_t>(model_config_.mla_config.kv_lora_rank);
    size_t qk_rope_head_dim = static_cast<size_t>(model_config_.mla_config.qk_rope_head_dim);
    size_t qk_nope_head_dim = static_cast<size_t>(model_config_.mla_config.qk_nope_head_dim);
    size_t v_head_dim = static_cast<size_t>(model_config_.mla_config.v_head_dim);
    size_t head_num = model_config_.head_num;
    size_t hidden_units = static_cast<size_t>(model_config_.hidden_units);
    size_t group_size = model_config_.quant_config.group_size;
    std::vector<std::string> mla_weight_name = {"kv_a_proj_with_mqa", "q_a_proj", "kv_b_proj", "q_b_proj"};
    Tensor empty_tensor;
    for (auto& name : mla_weight_name) {
      if (name == "kv_a_proj_with_mqa") {
        ArrayInfo kv_a_qweight_info;
        std::vector<int32_t> kv_a_qweight_data(hidden_units / pack_factor * (kv_lora_rank + qk_rope_head_dim), 1);
        kv_a_qweight_info.weight_ptr = reinterpret_cast<void*>(kv_a_qweight_data.data());
        kv_a_qweight_info.weight_shape = {hidden_units / pack_factor, kv_lora_rank + qk_rope_head_dim};
        kv_a_qweight_info.weight_data_type = TYPE_INT32;
        kv_a_qweight_info.tensor_name = "model.layers.0.self_attn." + name + ".qweight";
        mla_weight_arrays_.push_back(kv_a_qweight_info);

        ArrayInfo kv_a_scales_info;
        std::vector<half> kv_a_scales_data(hidden_units / group_size * (kv_lora_rank + qk_rope_head_dim), half(1.0f));
        kv_a_scales_info.weight_ptr = reinterpret_cast<void*>(kv_a_scales_data.data());
        kv_a_scales_info.weight_shape = {hidden_units / group_size, kv_lora_rank + qk_rope_head_dim};
        kv_a_scales_info.weight_data_type = TYPE_FP16;
        kv_a_scales_info.tensor_name = "model.layers.0.self_attn." + name + ".scales";
        mla_weight_arrays_.push_back(kv_a_scales_info);
      }
      if (name == "q_a_proj") {
        ArrayInfo q_a_qweight_info;
        std::vector<int32_t> q_a_qweight_data(hidden_units / pack_factor * q_lora_rank, 1);
        q_a_qweight_info.weight_ptr = reinterpret_cast<void*>(q_a_qweight_data.data());
        q_a_qweight_info.weight_shape = {hidden_units / pack_factor, q_lora_rank};
        q_a_qweight_info.weight_data_type = TYPE_INT32;
        q_a_qweight_info.tensor_name = "model.layers.0.self_attn." + name + ".qweight";
        mla_weight_arrays_.push_back(q_a_qweight_info);
        weights_map_["model.layers.0.self_attn." + name + ".scales"] = empty_tensor;
      }
      if (name == "kv_b_proj") {
        ArrayInfo kv_b_qweight_info;
        std::vector<int32_t> kv_b_qweight_data(kv_lora_rank / pack_factor * head_num * (qk_nope_head_dim + v_head_dim),
                                               1);
        kv_b_qweight_info.weight_ptr = reinterpret_cast<void*>(kv_b_qweight_data.data());
        kv_b_qweight_info.weight_shape = {kv_lora_rank / pack_factor, head_num * (qk_nope_head_dim + v_head_dim)};
        kv_b_qweight_info.weight_data_type = TYPE_INT32;
        kv_b_qweight_info.tensor_name = "model.layers.0.self_attn." + name + ".qweight";
        mla_weight_arrays_.push_back(kv_b_qweight_info);

        ArrayInfo kv_b_scales_info;
        std::vector<half> kv_b_scales_data(kv_lora_rank / group_size * head_num * (qk_nope_head_dim + v_head_dim),
                                           half(1.0f));
        kv_b_scales_info.weight_ptr = reinterpret_cast<void*>(kv_b_scales_data.data());
        kv_b_scales_info.weight_shape = {kv_lora_rank / group_size, head_num * (qk_nope_head_dim + v_head_dim)};
        kv_b_scales_info.weight_data_type = TYPE_FP16;
        kv_b_scales_info.tensor_name = "model.layers.0.self_attn." + name + ".scales";
        mla_weight_arrays_.push_back(kv_b_scales_info);
      }
      if (name == "q_b_proj") {
        ArrayInfo q_b_qweight_info;
        std::vector<int32_t> q_b_qweight_data(
            q_lora_rank / pack_factor * head_num * (qk_nope_head_dim + qk_rope_head_dim), 1);
        q_b_qweight_info.weight_ptr = reinterpret_cast<void*>(q_b_qweight_data.data());
        q_b_qweight_info.weight_shape = {q_lora_rank / pack_factor, head_num * (qk_nope_head_dim + qk_rope_head_dim)};
        q_b_qweight_info.weight_data_type = TYPE_INT32;
        q_b_qweight_info.tensor_name = "model.layers.0.self_attn." + name + ".qweight";
        mla_weight_arrays_.push_back(q_b_qweight_info);

        ArrayInfo q_b_scales_info;
        std::vector<half> q_b_scales_data(q_lora_rank / group_size * head_num * (qk_nope_head_dim + qk_rope_head_dim),
                                          half(1.0f));
        q_b_scales_info.weight_ptr = reinterpret_cast<void*>(q_b_scales_data.data());
        q_b_scales_info.weight_shape = {q_lora_rank / group_size, head_num * (qk_nope_head_dim + qk_rope_head_dim)};
        q_b_scales_info.weight_data_type = TYPE_FP16;
        q_b_scales_info.tensor_name = "model.layers.0.self_attn." + name + ".scales";
        mla_weight_arrays_.push_back(q_b_scales_info);
      }
    }
  }

  template <typename T>
  void TestGPTQMoeQuantWeightload() {
    // Init quant weight
    std::shared_ptr<QuantWeight<T>> quant_weight_solver =
        std::make_shared<QuantWeight<T>>(model_config_, rank, context_, weights_map_, weights_data_type_map_);

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
    }
  }
};

TEST_F(QuantWeightLoadTest, GPTQMoeQuantWeightloadTest) {
  InitOriginMoeArrays();
  TestGPTQMoeQuantWeightload<half>();
  size_t num_experts = model_config_.moe_config.num_experts;
  size_t moe_inter_size = model_config_.moe_config.moe_inter_size;
  size_t tp = model_config_.tensor_para_size;
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
}

TEST_F(QuantWeightLoadTest, CommonDeQuantTest) {
  model_config_.quant_config.method = QUANT_GPTQ;
  model_config_.quant_config.backend = MACHETE_BACKEND;

  std::shared_ptr<QuantWeight<half>> quant_weight_solver =
      std::make_shared<QuantWeight<half>>(model_config_, rank, context_, weights_map_, weights_data_type_map_);
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

TEST_F(QuantWeightLoadTest, LoadAndProcessMacheteWeightTest) {
  model_config_.is_quant = true;
  model_config_.quant_config.method = QUANT_GPTQ;
  model_config_.quant_config.backend = MACHETE_BACKEND;
  model_config_.use_mla = true;
  model_config_.is_moe = true;
  model_config_.has_shared_experts = false;
  model_config_.moe_config.first_k_dense_replace = 0;
  // Init weight
  InitOriginMlaArrays();
  if (model_config_.is_moe) {
    InitOriginMoeArrays();
    mla_weight_arrays_.insert(mla_weight_arrays_.end(), gate_qweight_arrays_.begin(), gate_qweight_arrays_.end());
    mla_weight_arrays_.insert(mla_weight_arrays_.end(), up_qweight_arrays_.begin(), up_qweight_arrays_.end());
    mla_weight_arrays_.insert(mla_weight_arrays_.end(), down_qweight_arrays_.begin(), down_qweight_arrays_.end());
  }
  std::shared_ptr<QuantWeight<half>> quant_weight_solver =
      std::make_shared<QuantWeight<half>>(model_config_, rank, context_, weights_map_, weights_data_type_map_);
  EXPECT_TRUE(quant_weight_solver->IsEnable());
  // Load quant weight
  for (auto& mla_weight_array : mla_weight_arrays_) {
    quant_weight_solver->LoadQuantWeight(mla_weight_array.tensor_name, mla_weight_array.weight_shape,
                                         mla_weight_array.weight_data_type, mla_weight_array.weight_ptr);
  }
  // Check load result
  std::vector<std::string> split_weight_names = {"kv_a_lora_proj", "kv_a_rope_proj", "kv_b_nope_proj", "v_head_proj",
                                                 "q_b_nope_proj",  "q_b_rope_proj",  "q_a_proj"};
  std::vector<std::string> weight_types = {".qweight", ".scales"};
  for (auto& split_weight_name : split_weight_names) {
    for (auto& weight_type : weight_types) {
      EXPECT_TRUE(weights_map_.find("model.layers.0.self_attn." + split_weight_name + weight_type) !=
                  weights_map_.end());
    }
  }
  for (auto& weight_type : weight_types) {
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.kv_a_lora_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.mla_config.kv_lora_rank));
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.kv_a_rope_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.mla_config.qk_rope_head_dim));
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.kv_b_nope_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.head_num * model_config_.mla_config.qk_nope_head_dim));
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.v_head_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.head_num * model_config_.mla_config.v_head_dim));
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.q_b_nope_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.head_num * model_config_.mla_config.qk_nope_head_dim));
    EXPECT_EQ(weights_map_["model.layers.0.self_attn.q_b_rope_proj" + weight_type].shape[1],
              static_cast<size_t>(model_config_.head_num * model_config_.mla_config.qk_rope_head_dim));
    // temp use other Tensor for process quant weight test
    weights_map_["model.layers.0.self_attn.o_proj" + weight_type] =
        weights_map_["model.layers.0.self_attn.kv_a_lora_proj" + weight_type];
  }
  // temp use other Tensor for process moe quant weight test
  if (model_config_.is_moe) {
    weights_map_["model.layers.0.mlp.experts.up_gate_proj.scales"] =
        weights_map_["model.layers.0.self_attn.kv_a_lora_proj.scales"];
    weights_map_["model.layers.0.mlp.experts.down_proj.scales"] =
        weights_map_["model.layers.0.self_attn.kv_a_lora_proj.scales"];
  }
  // Process quant weight
  quant_weight_solver->ConvertGroupTensor();
  // Check process result
  for (auto& split_weight_name : split_weight_names) {
    EXPECT_FALSE(weights_map_.find("model.layers.0.self_attn." + split_weight_name + ".qweight") != weights_map_.end());
    EXPECT_TRUE(weights_map_.find("model.layers.0.self_attn." + split_weight_name + ".weight") != weights_map_.end());
    EXPECT_NE(weights_map_["model.layers.0.self_attn." + split_weight_name + ".weight"].scales, nullptr);
  }
  if (model_config_.is_moe) {
    EXPECT_NE(weights_map_["model.layers.0.mlp.experts.up_gate_proj.weight"].scales, nullptr);
    EXPECT_NE(weights_map_["model.layers.0.mlp.experts.down_proj.weight"].scales, nullptr);
  }
}
