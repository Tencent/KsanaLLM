/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ksana_llm/models/qwen/qwen_weight.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader_test_helper.h"
#include "ksana_llm/utils/singleton.h"

using namespace ksana_llm;

class QwenGateUpProjMockLoader : public MockSafeTensorsLoader {
 public:
  explicit QwenGateUpProjMockLoader(const std::string& file_name, const bool load_bias)
      : MockSafeTensorsLoader(file_name, load_bias) {
    InitMockData();
  }

 private:
  void InitMockData() override {
    const int num_layers = 2;
    const int hidden_size = 512;
    const int inter_size = 2560;
    const int vocab_size = 1000;

    // Create Mock Data for Qwen model
    CreateMockTensor("model.embed_tokens.weight", {vocab_size, hidden_size}, TYPE_FP16, 0);
    CreateMockTensor("model.norm.weight", {hidden_size}, TYPE_FP16, 0);

    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      std::string layer_prefix = "model.layers." + std::to_string(layer_idx);

      // Attention weights
      CreateMockTensor(layer_prefix + ".self_attn.query_key_value.weight", {hidden_size, hidden_size}, TYPE_FP16,
                       layer_idx);
      CreateMockTensor(layer_prefix + ".self_attn.o_proj.weight", {hidden_size, hidden_size}, TYPE_FP16, layer_idx);

      // Layer norm weights
      CreateMockTensor(layer_prefix + ".input_layernorm.weight", {hidden_size}, TYPE_FP16, layer_idx);
      CreateMockTensor(layer_prefix + ".post_attention_layernorm.weight", {hidden_size}, TYPE_FP16, layer_idx);

      // MLP weights - Qwen specific gate_up_proj
      // Data layout: [gate_proj_rank0, gate_proj_rank1, ..., up_proj_rank0, up_proj_rank1, ...]
      CreateMockTensor(layer_prefix + ".mlp.gate_up_proj.weight", {inter_size * 2, hidden_size}, TYPE_FP16, layer_idx);
      CreateMockTensor(layer_prefix + ".mlp.down_proj.weight", {hidden_size, inter_size}, TYPE_FP16, layer_idx);
    }

    CreateMockTensor("lm_head.weight", {vocab_size, hidden_size}, TYPE_FP16, 0);
  }

  void CreateMockTensor(const std::string& tensor_name, const std::vector<size_t>& shape, DataType data_type,
                        size_t layer_idx) override {
    tensor_name_list_.push_back(tensor_name);
    tensor_shape_map_[tensor_name] = shape;
    tensor_data_type_map_[tensor_name] = data_type;

    // Calculate element count
    size_t element_count = 1;
    for (const auto& dim : shape) {
      element_count *= dim;
    }
    size_t tensor_size = element_count * GetTypeSize(data_type);
    tensor_size_map_[tensor_name] = tensor_size;

    // Allocate and fill in mock values
    void* tensor_data = malloc(tensor_size);
    if (data_type == TYPE_FP16) {
      float16* data_ptr = static_cast<float16*>(tensor_data);

      // Init gate_up_proj tensor according to Qwen's Data layout
      if (tensor_name.find("gate_up_proj.weight") != std::string::npos) {
        // Data layout: [gate_proj_rank0, gate_proj_rank1, ..., up_proj_rank0, up_proj_rank1, ...]
        // For single rank test, it's: [gate_proj, up_proj]
        size_t half_size = element_count / 2;

        // First half: gate_proj data
        for (size_t i = 0; i < half_size; ++i) {
          float value = (1000.0f + layer_idx * 10000.0f + i) * 0.01f;
          data_ptr[i] = static_cast<float16>(value);
        }

        // Second half: up_proj data
        for (size_t i = half_size; i < element_count; ++i) {
          float value = (2000.0f + layer_idx * 10000.0f + (i - half_size)) * 0.01f;
          data_ptr[i] = static_cast<float16>(value);
        }
      } else {
        // Regular tensor initialization
        for (size_t i = 0; i < element_count; ++i) {
          float value = (i + layer_idx * 100) * 0.01f;
          data_ptr[i] = static_cast<float16>(value);
        }
      }
    } else if (data_type == TYPE_FP32) {
      float* data_ptr = static_cast<float*>(tensor_data);
      for (size_t i = 0; i < element_count; ++i) {
        float value = (i + layer_idx * 100) * 0.01f;
        data_ptr[i] = value;
      }
    }

    tensor_ptr_map_[tensor_name] = tensor_data;
  }
};

class QwenGateUpProjWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    context_ = std::make_shared<Context>(1, 1, 1);

    // Initialize ModelConfig for Qwen
    model_config_.type = "qwen";
    model_config_.is_moe = false;
    model_config_.num_layer = 2;
    model_config_.hidden_units = 512;
    model_config_.inter_size = 2560;
    model_config_.weight_data_type = TYPE_FP16;

    // Initialize RuntimeConfig
    runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    runtime_config_.inter_data_type = TYPE_FP16;

    // Initialize PipelineConfig
    PipelineConfig pipeline_config;
    pipeline_config.lower_layer_idx = 0;
    pipeline_config.upper_layer_idx = 2;
    Singleton<Environment>::GetInstance()->SetPipelineConfig(pipeline_config);

    loader_ = std::make_shared<QwenGateUpProjMockLoader>("mock_qwen_gate_up.safetensors", false);

    const std::vector<std::string>& tensor_names = loader_->GetTensorNameList();
    weight_name_list_ = tensor_names;
    custom_name_list_ = tensor_names;
  }

  void TearDown() override {}

  template <typename T>
  std::vector<T> CopyDataFromDevice(void* data_ptr, const std::vector<size_t>& shape) {
    size_t num_elements = 1;
    for (const auto& dim : shape) {
      num_elements *= dim;
    }
    std::vector<T> cpu_data(num_elements);
    Memcpy(cpu_data.data(), data_ptr, num_elements * sizeof(T), MEMCPY_DEVICE_TO_HOST);
    return cpu_data;
  }
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_{nullptr};
  std::vector<std::string> weight_name_list_;
  std::vector<std::string> custom_name_list_;
  std::shared_ptr<BaseFileTensorLoader> loader_;
};

// Test qwen gate_up_proj weight loading function
TEST_F(QwenGateUpProjWeightTest, QwenGateUpProjWeightLoadTest) {
#ifdef ENABLE_CUDA
  KLLM_LOG_INFO << "Testing Qwen gate_up_proj weight loading function";
  // Create QwenWeight instance
  std::shared_ptr<QwenWeight<float16>> qwen_weight =
      std::make_shared<QwenWeight<float16>>(model_config_, runtime_config_, 0, context_);

  // Load weights
  Status status = qwen_weight->LoadWeightsFromFile(loader_, weight_name_list_, custom_name_list_);
  EXPECT_TRUE(status.OK()) << "LoadWeightsFromFile failed: " << status.GetMessage();

  // Check gate_up_proj weights for each layer
  for (int layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
    std::string gate_up_name = "model.layers." + std::to_string(layer_idx) + ".mlp.gate_up_proj.weight";

    Tensor gate_up_tensor = qwen_weight->GetModelWeights(gate_up_name);

    // Check if tensor is valid
    EXPECT_NE(gate_up_tensor.GetPtr<void>(), nullptr) << "gate_up_proj.weight should be loaded for layer " << layer_idx;

    // Check shape
    EXPECT_EQ(gate_up_tensor.shape.size(), 2) << "gate_up_proj should have 2 dimensions";
    EXPECT_EQ(gate_up_tensor.shape[0], model_config_.inter_size * 2)
        << "gate_up_proj first dimension should be inter_size * 2";
    EXPECT_EQ(gate_up_tensor.shape[1], model_config_.hidden_units)
        << "gate_up_proj second dimension should be hidden_units";

    // Copy data from device to host
    auto gate_up_data = CopyDataFromDevice<float16>(gate_up_tensor.GetPtr<void>(), gate_up_tensor.shape);

    size_t half_size = gate_up_tensor.GetElementNumber() / 2;

    // Check first half (gate_proj data)
    KLLM_LOG_DEBUG << "Verifying gate_proj data for layer " << layer_idx;
    for (size_t i = 0; i < 10; ++i) {  // Check first 10 elements
      float expected = static_cast<float>(static_cast<float16>((1000.0f + layer_idx * 10000.0f + i) * 0.01f));
      float actual = static_cast<float>(gate_up_data[i]);
      EXPECT_EQ(actual, expected) << "gate_proj data mismatch at index " << i << " for layer " << layer_idx;
    }

    // Check second half (up_proj data)
    KLLM_LOG_DEBUG << "Verifying up_proj data for layer " << layer_idx;
    for (size_t i = 0; i < 10; ++i) {  // Check first 10 elements of second half
      float expected = static_cast<float>(static_cast<float16>((2000.0f + layer_idx * 10000.0f + i) * 0.01f));
      float actual = static_cast<float>(gate_up_data[half_size + i]);
      EXPECT_EQ(actual, expected) << "up_proj data mismatch at index " << i << " for layer " << layer_idx;
    }
  }
  KLLM_LOG_INFO << "Qwen gate_up_proj weight loading test completed successfully";
#endif
}