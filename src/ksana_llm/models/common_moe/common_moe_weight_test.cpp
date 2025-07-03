/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common_moe/common_moe_weight.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "ksana_llm/cache_manager/block_allocator/block_allocator_manager.h"
#include "ksana_llm/cache_manager/cache_manager_interface.h"
#include "ksana_llm/cache_manager/prefix_cache_manager.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_allocator.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader_test_helper.h"
#include "ksana_llm/utils/singleton.h"

using namespace ksana_llm;
using json = nlohmann::json;

// Create a MoeMockSafeTensorsLoader class to simulate the behavior of SafeTensorsLoader
class MoeMockSafeTensorsLoader : public MockSafeTensorsLoader {
 public:
  explicit MoeMockSafeTensorsLoader(const std::string& file_name, const bool load_bias)
      : MockSafeTensorsLoader(file_name, load_bias) {
    InitMockData();
  }

 private:
  void InitMockData() override {
    const int num_layers = 2;
    const int hidden_size = 5;
    const int inter_size = 6;
    const int num_experts = 2;

    // Create Mock Data
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
        // up_proj
        CreateMockTensor(fmt::format("model.layers.{}.mlp.experts.{}.up_proj.weight", layer_idx, expert_idx),
                         {inter_size, hidden_size}, TYPE_FP16, expert_idx);

        // gate_proj
        CreateMockTensor(fmt::format("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_idx, expert_idx),
                         {inter_size, hidden_size}, TYPE_FP16, expert_idx);

        // down_proj
        CreateMockTensor(fmt::format("model.layers.{}.mlp.experts.{}.down_proj.weight", layer_idx, expert_idx),
                         {hidden_size, inter_size}, TYPE_FP16, expert_idx);
      }
    }
  }

  void CreateMockTensor(const std::string& tensor_name, const std::vector<size_t>& shape, DataType data_type,
                        size_t expert_idx) override {
    tensor_name_list_.push_back(tensor_name);
    tensor_shape_map_[tensor_name] = shape;
    tensor_data_type_map_[tensor_name] = data_type;

    // get element size
    size_t element_count = 1;
    for (const auto& dim : shape) {
      element_count *= dim;
    }
    size_t tensor_size = element_count * GetTypeSize(data_type);
    tensor_size_map_[tensor_name] = tensor_size;

    // Allocate and fill in random values.
    void* tensor_data = malloc(tensor_size);
    if (data_type == TYPE_FP16) {
      float16* data_ptr = static_cast<float16*>(tensor_data);
      for (size_t i = 0; i < element_count; ++i) {
        float value = (i + expert_idx * element_count) * 0.01f;
        data_ptr[i] = static_cast<float16>(value);
      }
    } else if (data_type == TYPE_FP32) {
      float* data_ptr = static_cast<float*>(tensor_data);
      for (size_t i = 0; i < element_count; ++i) {
        float value = (i + expert_idx * element_count) * 0.01f;
        data_ptr[i] = value;
      }
    }

    tensor_ptr_map_[tensor_name] = tensor_data;
  }
};

class CommonMoeWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(2, 1, 1);

    // Initialize ModelConfig
    model_config.hidden_units = 5;
    model_config.inter_size = 6;
    model_config.moe_config.num_experts = 2;
    model_config.moe_config.moe_inter_size = 6;
    model_config.moe_config.first_k_dense_replace = 0;
    model_config.has_shared_experts = false;
    model_config.weight_data_type = TYPE_FP16;

    PipelineConfig pipeline_config;
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = 2;
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);

    loader = std::make_shared<MoeMockSafeTensorsLoader>("mock_safetensors", true);

    const std::vector<std::string>& tensor_names = loader->GetTensorNameList();
    weight_name_list = tensor_names;
    custom_name_list = tensor_names;
  }

  void TearDown() override { loguru::g_stderr_verbosity = origin_stderr_verbosity; }

  std::vector<float16> GetFP16DeviceData(void* data_ptr, const std::vector<size_t>& shape) {
    size_t num_elements = 1;
    for (auto& dim : shape) {
      num_elements *= dim;
    }
    std::vector<float16> cpu_data(num_elements);
    Memcpy(cpu_data.data(), data_ptr, num_elements * sizeof(float16), MEMCPY_DEVICE_TO_HOST);
    return cpu_data;
  }

  void TestSingleDevice() {
    // Create model config
    model_config.tensor_para_size = 1;
    model_config.expert_para_size = 1;
    model_config.expert_world_size = 1;
    model_config.moe_tensor_para_size = 1;
    std::shared_ptr<CommonMoeWeight<float16>> weight =
        std::make_shared<CommonMoeWeight<float16>>(model_config, 0, context_);

    // load weights.
    weight->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
    weight->ProcessWeights();

    // All the experts need to be loaded in single device mode.
    size_t num_experts = model_config.moe_config.num_experts;
    size_t moe_inter_size = model_config.moe_config.moe_inter_size;
    size_t hidden_units = model_config.hidden_units;

    for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
      // Check up_gate_proj weight
      std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx);
      Tensor& up_gate_tensor = weight->weights_map_[up_gate_name];
      EXPECT_EQ(up_gate_tensor.shape.size(), 3);
      EXPECT_EQ(up_gate_tensor.shape[0], num_experts);
      EXPECT_EQ(up_gate_tensor.shape[1], moe_inter_size * 2);
      EXPECT_EQ(up_gate_tensor.shape[2], hidden_units);
      // Shape [2, 12, 5]
      // array([[[0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14],
      //         [0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29],
      //         [0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14],
      //         [0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29]],
      //        [[0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44],
      //         [0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59],
      //         [0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44],
      //         [0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59]]])
      auto up_gate_data = GetFP16DeviceData(up_gate_tensor.GetPtr<void>(), up_gate_tensor.shape);
      size_t up_gate_element_size = up_gate_data.size();
      for (size_t i = 0; i < num_experts; ++i) {
        for (size_t j = 0; j < moe_inter_size; ++j) {
          for (size_t k = 0; k < hidden_units; ++k) {
            size_t idx = i * moe_inter_size * 2 * hidden_units + j * hidden_units + k;
            size_t value = i * moe_inter_size * hidden_units + j * hidden_units + k;
            EXPECT_NEAR(static_cast<float>(up_gate_data[idx]), value * 0.01f, 1e-3);
            EXPECT_NEAR(static_cast<float>(up_gate_data[idx + moe_inter_size * hidden_units]), value * 0.01f, 1e-3);
          }
        }
      }

      // Check down_proj weight
      std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx);
      Tensor& down_tensor = weight->weights_map_[down_name];
      EXPECT_EQ(down_tensor.shape.size(), 3);
      EXPECT_EQ(down_tensor.shape[0], num_experts);
      EXPECT_EQ(down_tensor.shape[1], hidden_units);
      EXPECT_EQ(down_tensor.shape[2], moe_inter_size);
      // Shape [2, 5, 6]
      // array([0.01, 0.02, ... 0.59])
      auto down_data = GetFP16DeviceData(down_tensor.GetPtr<void>(), down_tensor.shape);
      size_t down_element_size = down_data.size();
      for (size_t i = 0; i < down_element_size; ++i) {
        EXPECT_NEAR(static_cast<float>(down_data[i]), i * 0.01f, 1e-3);
      }
    }
  }

  void TestTensorParallel() {
    // Create model_config
    model_config.tensor_para_size = 2;
    model_config.expert_para_size = 1;
    model_config.expert_world_size = 1;
    model_config.moe_tensor_para_size = 2;
    std::shared_ptr<CommonMoeWeight<float16>> weight_rank_0 =
        std::make_shared<CommonMoeWeight<float16>>(model_config, 0, context_);
    std::shared_ptr<CommonMoeWeight<float16>> weight_rank_1 =
        std::make_shared<CommonMoeWeight<float16>>(model_config, 1, context_);

    size_t num_experts = model_config.moe_config.num_experts;
    size_t moe_inter_size = model_config.moe_config.moe_inter_size;
    size_t hidden_units = model_config.hidden_units;
    size_t elements_per_expert = moe_inter_size * hidden_units / model_config.tensor_para_size;

    // Load weight into rank 0
    SetDevice(0);
    weight_rank_0->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
    weight_rank_0->ProcessWeights();

    // Check rank 0 weight
    for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
      // Check up_gate_proj weight
      std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx);
      Tensor& up_gate_tensor_rank_0 = weight_rank_0->weights_map_[up_gate_name];
      EXPECT_EQ(up_gate_tensor_rank_0.shape.size(), 3);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[0], num_experts);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[1], moe_inter_size * 2 / model_config.tensor_para_size);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[2], hidden_units);
      // Shape [2, 6, 5]
      // array([[[0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14],
      //         [0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14]],
      //        [[0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44],
      //         [0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44]]])
      auto up_gate_data = GetFP16DeviceData(up_gate_tensor_rank_0.GetPtr<void>(), up_gate_tensor_rank_0.shape);
      size_t up_gate_element_size = up_gate_data.size();
      for (size_t i = 0; i < up_gate_element_size; ++i) {
        size_t expert_id = i / (2 * elements_per_expert);
        size_t inter_id = ((i - (expert_id * 2 * elements_per_expert)) / hidden_units) %
                          (moe_inter_size / model_config.tensor_para_size);
        size_t hidden_id = i % hidden_units;
        size_t value = expert_id * moe_inter_size * 2 * hidden_units / model_config.tensor_para_size +
                       inter_id * hidden_units + hidden_id;
        EXPECT_NEAR(static_cast<float>(up_gate_data[i]), value * 0.01f, 1e-3);
      }

      // Check down_proj weight
      std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx);
      Tensor& down_tensor_rank_0 = weight_rank_0->weights_map_[down_name];
      EXPECT_EQ(down_tensor_rank_0.shape.size(), 3);
      EXPECT_EQ(down_tensor_rank_0.shape[0], num_experts);
      EXPECT_EQ(down_tensor_rank_0.shape[1], hidden_units);
      EXPECT_EQ(down_tensor_rank_0.shape[2], moe_inter_size / model_config.tensor_para_size);
      // Shape [2, 5, 3]
      // array([[[0.  , 0.01, 0.02],
      //         [0.06, 0.07, 0.08],
      //         [0.12, 0.13, 0.14],
      //         [0.18, 0.19, 0.2 ],
      //         [0.24, 0.25, 0.26]],
      //        [[0.3 , 0.31, 0.32],
      //         [0.36, 0.37, 0.38],
      //         [0.42, 0.43, 0.44],
      //         [0.48, 0.49, 0.5 ],
      //         [0.54, 0.55, 0.56]]])
      auto down_data = GetFP16DeviceData(down_tensor_rank_0.GetPtr<void>(), down_tensor_rank_0.shape);
      size_t down_element_size = down_data.size();
      for (size_t i = 0; i < down_element_size; ++i) {
        size_t value = (i / (moe_inter_size / model_config.tensor_para_size)) *
                           (moe_inter_size / model_config.tensor_para_size * 2) +
                       (i % (moe_inter_size / model_config.tensor_para_size));
        EXPECT_NEAR(static_cast<float>(down_data[i]), value * 0.01f, 1e-3);
      }
    }

    // Check rank 0 expert_map
    // Shape [256]
    // array([0, 1, ..., 254, 255])
    Tensor& expert_map_tensor_rank_0 = weight_rank_0->weights_map_["expert_map"];
    EXPECT_EQ(expert_map_tensor_rank_0.shape.size(), 1);
    EXPECT_EQ(expert_map_tensor_rank_0.shape[0], num_experts);
    std::vector<int> expert_map_rank_0(num_experts);
    Memcpy(expert_map_rank_0.data(), expert_map_tensor_rank_0.GetPtr<void>(), num_experts * sizeof(int),
           MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < num_experts; ++i) {
      EXPECT_EQ(expert_map_rank_0[i], i);
    }

    // Load weight into rank 1
    SetDevice(1);
    weight_rank_1->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
    weight_rank_1->ProcessWeights();

    // Check rank 1 weight
    for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
      // Check up_gate_proj weight
      std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx);
      Tensor& up_gate_tensor_rank_1 = weight_rank_1->weights_map_[up_gate_name];
      EXPECT_EQ(up_gate_tensor_rank_1.shape.size(), 3);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[0], num_experts);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[1], moe_inter_size * 2 / model_config.tensor_para_size);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[2], hidden_units);
      // Shape [2, 6, 5]
      // array([[[0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29],
      //         [0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29]],
      //        [[0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59],
      //         [0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59]]])
      auto up_gate_data = GetFP16DeviceData(up_gate_tensor_rank_1.GetPtr<void>(), up_gate_tensor_rank_1.shape);
      size_t up_gate_element_size = up_gate_data.size();
      for (size_t i = 0; i < up_gate_element_size; ++i) {
        size_t expert_id = i / (2 * elements_per_expert);
        size_t inter_id = ((i - (expert_id * 2 * elements_per_expert)) / hidden_units) %
                          (moe_inter_size / model_config.tensor_para_size);
        size_t hidden_id = i % hidden_units;
        size_t value = expert_id * moe_inter_size * 2 * hidden_units / model_config.tensor_para_size +
                       inter_id * hidden_units + hidden_id +
                       moe_inter_size * hidden_units / model_config.tensor_para_size;
        EXPECT_NEAR(static_cast<float>(up_gate_data[i]), value * 0.01f, 1e-3);
      }

      // Check down_gate_proj weight
      std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx);
      Tensor& down_tensor_rank_1 = weight_rank_1->weights_map_[down_name];
      EXPECT_EQ(down_tensor_rank_1.shape.size(), 3);
      EXPECT_EQ(down_tensor_rank_1.shape[0], num_experts);
      EXPECT_EQ(down_tensor_rank_1.shape[1], hidden_units);
      EXPECT_EQ(down_tensor_rank_1.shape[2], moe_inter_size / model_config.tensor_para_size);
      // Shape [2, 5, 3]
      // array([[[0.03, 0.04, 0.05],
      //         [0.09, 0.1 , 0.11],
      //         [0.15, 0.16, 0.17],
      //         [0.21, 0.22, 0.23],
      //         [0.27, 0.28, 0.29]],
      //        [[0.33, 0.34, 0.35],
      //         [0.39, 0.4 , 0.41],
      //         [0.45, 0.46, 0.47],
      //         [0.51, 0.52, 0.53],
      //         [0.57, 0.58, 0.59]]]
      auto down_data = GetFP16DeviceData(down_tensor_rank_1.GetPtr<void>(), down_tensor_rank_1.shape);
      size_t down_element_size = down_data.size();
      for (size_t i = 0; i < down_element_size; ++i) {
        size_t value = (i / (moe_inter_size / model_config.tensor_para_size)) *
                           (moe_inter_size / model_config.tensor_para_size * 2) +
                       (i % (moe_inter_size / model_config.tensor_para_size)) +
                       moe_inter_size / model_config.tensor_para_size;
        EXPECT_NEAR(static_cast<float>(down_data[i]), value * 0.01f, 1e-3);
      }
    }

    // Check rank 1 expert_map
    // Shape [256]
    // array([0, 1, ..., 254, 255])
    Tensor& expert_map_tensor_rank_1 = weight_rank_1->weights_map_["expert_map"];
    EXPECT_EQ(expert_map_tensor_rank_1.shape.size(), 1);
    EXPECT_EQ(expert_map_tensor_rank_1.shape[0], model_config.moe_config.num_experts);
    std::vector<int> expert_map_rank_1(model_config.moe_config.num_experts);
    Memcpy(expert_map_rank_1.data(), expert_map_tensor_rank_1.GetPtr<void>(),
           model_config.moe_config.num_experts * sizeof(int), MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < model_config.moe_config.num_experts; ++i) {
      EXPECT_EQ(expert_map_rank_1[i], i);
    }

    SetDevice(0);
  }

  void TestExpertParallel() {
    // Create model_config
    model_config.tensor_para_size = 2;
    model_config.expert_para_size = 2;
    model_config.expert_world_size = 1;
    model_config.moe_tensor_para_size = 1;
    std::shared_ptr<CommonMoeWeight<float16>> weight_rank_0 =
        std::make_shared<CommonMoeWeight<float16>>(model_config, 0, context_);
    std::shared_ptr<CommonMoeWeight<float16>> weight_rank_1 =
        std::make_shared<CommonMoeWeight<float16>>(model_config, 1, context_);

    size_t num_experts = model_config.moe_config.num_experts;
    size_t moe_inter_size = model_config.moe_config.moe_inter_size;
    size_t hidden_units = model_config.hidden_units;

    // Load weight into rank 0
    SetDevice(0);
    weight_rank_0->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
    weight_rank_0->ProcessWeights();

    // Check rank 0 weight
    for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
      // Check up_gate_proj weight
      std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx);
      Tensor& up_gate_tensor_rank_0 = weight_rank_0->weights_map_[up_gate_name];
      EXPECT_EQ(up_gate_tensor_rank_0.shape.size(), 3);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[0], num_experts / model_config.expert_para_size);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[1], moe_inter_size * 2);
      EXPECT_EQ(up_gate_tensor_rank_0.shape[2], hidden_units);
      // Shape [1, 12, 5]
      // array([[[0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14],
      //         [0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29],
      //         [0.  , 0.01, 0.02, 0.03, 0.04],
      //         [0.05, 0.06, 0.07, 0.08, 0.09],
      //         [0.1 , 0.11, 0.12, 0.13, 0.14],
      //         [0.15, 0.16, 0.17, 0.18, 0.19],
      //         [0.2 , 0.21, 0.22, 0.23, 0.24],
      //         [0.25, 0.26, 0.27, 0.28, 0.29]]])
      auto up_gate_data = GetFP16DeviceData(up_gate_tensor_rank_0.GetPtr<void>(), up_gate_tensor_rank_0.shape);
      size_t up_gate_element_size = up_gate_data.size();
      for (size_t j = 0; j < moe_inter_size; ++j) {
        for (size_t k = 0; k < hidden_units; ++k) {
          size_t idx = j * hidden_units + k;
          size_t value = j * hidden_units + k;
          EXPECT_NEAR(static_cast<float>(up_gate_data[idx]), value * 0.01f, 1e-3);
          EXPECT_NEAR(static_cast<float>(up_gate_data[idx + moe_inter_size * hidden_units]), value * 0.01f, 1e-3);
        }
      }

      // Check down_proj Weight
      std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx);
      Tensor& down_tensor_rank_0 = weight_rank_0->weights_map_[down_name];
      EXPECT_EQ(down_tensor_rank_0.shape.size(), 3);
      EXPECT_EQ(down_tensor_rank_0.shape[0], num_experts / model_config.expert_para_size);
      EXPECT_EQ(down_tensor_rank_0.shape[1], hidden_units);
      EXPECT_EQ(down_tensor_rank_0.shape[2], moe_inter_size);
      // Shape [1, 5, 6]
      // array([0.0, 0.01, ... 0.29])
      auto down_data = GetFP16DeviceData(down_tensor_rank_0.GetPtr<void>(), down_tensor_rank_0.shape);
      size_t down_element_size = down_data.size();
      for (size_t i = 0; i < down_element_size; ++i) {
        EXPECT_NEAR(static_cast<float>(down_data[i]), i * 0.01f, 1e-3);
      }
    }

    // Check rank 0 expert_map
    // Shape [256]
    // array([0, 2])
    Tensor& expert_map_tensor_rank_0 = weight_rank_0->weights_map_["expert_map"];
    EXPECT_EQ(expert_map_tensor_rank_0.shape.size(), 1);
    EXPECT_EQ(expert_map_tensor_rank_0.shape[0], num_experts);
    std::vector<int> expert_map_rank_0(num_experts);
    Memcpy(expert_map_rank_0.data(), expert_map_tensor_rank_0.GetPtr<void>(), num_experts * sizeof(int),
           MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < num_experts; ++i) {
      if (i < num_experts / model_config.expert_para_size) {
        EXPECT_EQ(expert_map_rank_0[i], i);
      } else {
        EXPECT_EQ(expert_map_rank_0[i], num_experts / model_config.expert_para_size + 1);
      }
    }

    // Load weight into rank 1
    SetDevice(1);
    weight_rank_1->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
    weight_rank_1->ProcessWeights();

    // Check rank 1 Weight
    for (int layer_idx = 0; layer_idx < 2; ++layer_idx) {
      // Check up_gate_proj Weight
      std::string up_gate_name = fmt::format("model.layers.{}.mlp.experts.up_gate_proj.weight", layer_idx);
      Tensor& up_gate_tensor_rank_1 = weight_rank_1->weights_map_[up_gate_name];
      EXPECT_EQ(up_gate_tensor_rank_1.shape.size(), 3);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[0], num_experts / model_config.expert_para_size);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[1], moe_inter_size * 2);
      EXPECT_EQ(up_gate_tensor_rank_1.shape[2], hidden_units);
      // Shape [1, 12, 5]
      // array([[[0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44],
      //         [0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59],
      //         [0.3 , 0.31, 0.32, 0.33, 0.34],
      //         [0.35, 0.36, 0.37, 0.38, 0.39],
      //         [0.4 , 0.41, 0.42, 0.43, 0.44],
      //         [0.45, 0.46, 0.47, 0.48, 0.49],
      //         [0.5 , 0.51, 0.52, 0.53, 0.54],
      //         [0.55, 0.56, 0.57, 0.58, 0.59]]])
      auto up_gate_data = GetFP16DeviceData(up_gate_tensor_rank_1.GetPtr<void>(), up_gate_tensor_rank_1.shape);
      size_t up_gate_element_size = up_gate_data.size();
      for (size_t j = 0; j < moe_inter_size; ++j) {
        for (size_t k = 0; k < hidden_units; ++k) {
          size_t idx = j * hidden_units + k;
          size_t value = moe_inter_size * hidden_units + j * hidden_units + k;
          EXPECT_NEAR(static_cast<float>(up_gate_data[idx]), value * 0.01f, 1e-3);
          EXPECT_NEAR(static_cast<float>(up_gate_data[idx + moe_inter_size * hidden_units]), value * 0.01f, 1e-3);
        }
      }

      // Check down_proj Weight
      std::string down_name = fmt::format("model.layers.{}.mlp.experts.down_proj.weight", layer_idx);
      Tensor& down_tensor_rank_1 = weight_rank_1->weights_map_[down_name];
      EXPECT_EQ(down_tensor_rank_1.shape.size(), 3);
      EXPECT_EQ(down_tensor_rank_1.shape[0], num_experts / model_config.expert_para_size);
      EXPECT_EQ(down_tensor_rank_1.shape[1], hidden_units);
      EXPECT_EQ(down_tensor_rank_1.shape[2], moe_inter_size);
      // Shape [1, 5, 6]
      // array([0.30, 0.31, ... 0.59])
      auto down_data = GetFP16DeviceData(down_tensor_rank_1.GetPtr<void>(), down_tensor_rank_1.shape);
      size_t down_element_size = down_data.size();
      for (size_t i = 0; i < down_element_size; ++i) {
        EXPECT_NEAR(static_cast<float>(down_data[i]), (i + moe_inter_size * hidden_units) * 0.01f, 1e-3);
      }
    }

    // Check rank 1 expert_map
    // Shape [256]
    // array([2, 0])
    Tensor& expert_map_tensor_rank_1 = weight_rank_1->weights_map_["expert_map"];
    EXPECT_EQ(expert_map_tensor_rank_1.shape.size(), 1);
    EXPECT_EQ(expert_map_tensor_rank_1.shape[0], num_experts);
    std::vector<int> expert_map_rank_1(num_experts);
    Memcpy(expert_map_rank_1.data(), expert_map_tensor_rank_1.GetPtr<void>(), num_experts * sizeof(int),
           MEMCPY_DEVICE_TO_HOST);
    for (size_t i = 0; i < num_experts; ++i) {
      if (i < num_experts / model_config.expert_para_size) {
        EXPECT_EQ(expert_map_rank_1[i], num_experts / model_config.expert_para_size + 1);
      } else {
        EXPECT_EQ(expert_map_rank_1[i], i - num_experts / model_config.expert_para_size);
      }
    }

    SetDevice(0);
  }

  int origin_stderr_verbosity;
  ModelConfig model_config;
  std::shared_ptr<Context> context_{nullptr};
  std::vector<std::string> weight_name_list;
  std::vector<std::string> custom_name_list;
  std::shared_ptr<BaseFileTensorLoader> loader;
};

// Test loading MoE weights with a single device (No Tensor-Parallel nor Expert-Parallel)
TEST_F(CommonMoeWeightTest, SingleDeviceTest) {
#ifdef ENABLE_CUDA
  TestSingleDevice();
#endif
}

// Test loading MoE weights with 2 devices in Tensor-Parallel mode
TEST_F(CommonMoeWeightTest, TensorParallelTest) {
#ifdef ENABLE_CUDA
  TestTensorParallel();
#endif
}

// Test loading MoE weights with 2 devices in Expert-Parallel mode
TEST_F(CommonMoeWeightTest, ExpertParallelTest) {
#ifdef ENABLE_CUDA
  TestExpertParallel();
#endif
}
