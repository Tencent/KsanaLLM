/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/deepseek_v3/deepseek_v3_weight.h"
#include "ksana_llm/utils/absorb_weights_type.h"
#include "ksana_llm/utils/safetensors_file_tensor_loader_test_helper.h"
#include "ksana_llm/utils/singleton.h"
#include "tests/test.h"

using namespace ksana_llm;

// Create a GPTQMlaMockSafeTensorsLoader class to simulate the behavior of SafeTensorsLoader
class GPTQMlaMockSafeTensorsLoader : public MockSafeTensorsLoader {
 public:
  explicit GPTQMlaMockSafeTensorsLoader(const std::string& file_name, const bool load_bias)
      : MockSafeTensorsLoader(file_name, load_bias) {
    InitMockData();
  }

 private:
  void InitMockData() override {
    const int num_layers = 1;
    const size_t kv_lora_rank = 256;
    const size_t qk_nope_head_dim = 128;
    const size_t v_head_dim = 128;
    const size_t pack_factor = 8;
    const size_t group_size = 128;
    const size_t head_num_per_tp = 2;

    // Create Mock Data
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
      // kv_b_proj
      CreateMockTensor(fmt::format("model.layers.{}.self_attn.kv_b_proj.qweight", layer_idx),
                       {kv_lora_rank / pack_factor, head_num_per_tp * (qk_nope_head_dim + v_head_dim)}, TYPE_INT32, 0);
      CreateMockTensor(fmt::format("model.layers.{}.self_attn.kv_b_proj.scales", layer_idx),
                       {kv_lora_rank / group_size, head_num_per_tp * (qk_nope_head_dim + v_head_dim)}, TYPE_FP16, 0);
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
        float value = 1.0f;
        data_ptr[i] = static_cast<float16>(value);
      }
    } else if (data_type == TYPE_INT32) {
      float* data_ptr = static_cast<float*>(tensor_data);
      for (size_t i = 0; i < element_count; ++i) {
        int32_t value = 0;
        data_ptr[i] = value;
      }
    }

    tensor_ptr_map_[tensor_name] = tensor_data;
  }
};

class DeepseekV3AbosrbWeightTest : public testing::Test {
 protected:
  void SetUp() override {
    model_config_.moe_config.moe_inter_size = 128;
    runtime_config_.parallel_basic_config.tensor_parallel_size = 1;
    runtime_config_.parallel_basic_config.attn_data_parallel_size = 1;
    runtime_config_.parallel_basic_config.expert_parallel_size = 1;
    runtime_config_.parallel_basic_config.expert_world_size = 1;
    runtime_config_.parallel_basic_config.moe_tensor_para_size = 1;
    model_config_.moe_config.num_experts = 4;
    model_config_.hidden_units = 128;
    model_config_.weight_data_type = TYPE_FP16;
    model_config_.num_layer = 1;
    model_config_.mla_config.q_lora_rank = 128;
    model_config_.mla_config.kv_lora_rank = 256;
    model_config_.mla_config.qk_rope_head_dim = 64;
    model_config_.mla_config.qk_nope_head_dim = 128;
    model_config_.mla_config.v_head_dim = 128;
    model_config_.head_num = 2;
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
  ModelConfig model_config_;
  RuntimeConfig runtime_config_;
  std::shared_ptr<Context> context_{nullptr};
  std::vector<std::string> weight_name_list;
  std::vector<std::string> custom_name_list;
  std::shared_ptr<BaseFileTensorLoader> loader;
};

TEST_F(DeepseekV3AbosrbWeightTest, GPTQAbsorbWeightV2Test) {
  SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbTypeBMM);
  model_config_.quant_config.bits = 4;  // fake bits
  model_config_.quant_config.group_size = 128;
  model_config_.is_quant = true;
  model_config_.quant_config.method = QUANT_GPTQ;
  model_config_.quant_config.backend = MACHETE_BACKEND;
  model_config_.use_mla = true;
  model_config_.is_moe = false;
  model_config_.has_shared_experts = false;
  model_config_.moe_config.first_k_dense_replace = 0;

  size_t head_num_per_tp = model_config_.head_num / (runtime_config_.parallel_basic_config.tensor_parallel_size /
                                                     runtime_config_.parallel_basic_config.attn_data_parallel_size);
  loader = std::make_shared<GPTQMlaMockSafeTensorsLoader>("mock_safetensors", true);
  const std::vector<std::string>& tensor_names = loader->GetTensorNameList();
  weight_name_list = tensor_names;
  custom_name_list = tensor_names;
  // init weights
  std::shared_ptr<DeepSeekV3Weight<float16>> weight =
      std::make_shared<DeepSeekV3Weight<float16>>(model_config_, runtime_config_, 0, context_);
  // load weights.
  weight->LoadWeightsFromFile(loader, weight_name_list, custom_name_list);
  weight->ProcessWeights();
  // test
  auto options = torch::TensorOptions().device(torch::kCUDA, rank).dtype(GetTorchTypeFromDataType(TYPE_FP16));
  for (int layer_idx = 0; layer_idx < model_config_.num_layer; ++layer_idx) {
    std::string w_uk_t_name = fmt::format("model.layers.{}.self_attn.w_uk_t.weight", layer_idx);
    std::string w_uv_name = fmt::format("model.layers.{}.self_attn.w_uv.weight", layer_idx);
    EXPECT_TRUE(weight->weights_map_.find(w_uk_t_name) != weight->weights_map_.end());
    EXPECT_TRUE(weight->weights_map_.find(w_uk_t_name) != weight->weights_map_.end());
    Tensor& w_uk_t_tensor = weight->weights_map_[w_uk_t_name];
    Tensor& w_uv_tensor = weight->weights_map_[w_uv_name];
    EXPECT_EQ(w_uk_t_tensor.shape[0], head_num_per_tp);
    EXPECT_EQ(w_uk_t_tensor.shape[1], model_config_.mla_config.qk_nope_head_dim);
    EXPECT_EQ(w_uk_t_tensor.shape[2], model_config_.mla_config.kv_lora_rank);
    EXPECT_EQ(w_uv_tensor.shape[0], head_num_per_tp);
    EXPECT_EQ(w_uv_tensor.shape[1], model_config_.mla_config.kv_lora_rank);
    EXPECT_EQ(w_uv_tensor.shape[2], model_config_.mla_config.v_head_dim);
    EXPECT_EQ(w_uk_t_tensor.dtype, TYPE_FP16);
    EXPECT_EQ(w_uv_tensor.dtype, TYPE_FP16);
    // test value
    torch::Tensor w_uk_t_torch = torch::from_blob(w_uk_t_tensor.GetPtr<void>(),
                                                  {static_cast<int64_t>(w_uk_t_tensor.GetElementNumber())}, options);
    torch::Tensor w_uk_t_cpu_torch = w_uk_t_torch.to(torch::kCPU);
    EXPECT_TRUE(torch::all(w_uk_t_cpu_torch == -8).item<bool>());
    torch::Tensor w_uv_torch =
        torch::from_blob(w_uv_tensor.GetPtr<void>(), {static_cast<int64_t>(w_uv_tensor.GetElementNumber())}, options);
    torch::Tensor w_uv_cpu_torch = w_uv_torch.to(torch::kCPU);
    EXPECT_TRUE(torch::all(w_uv_cpu_torch == -8).item<bool>());
  }
  SetAbsorbWeightsType(AbsorbWeightsType::kAbsorbDisabled);
}
