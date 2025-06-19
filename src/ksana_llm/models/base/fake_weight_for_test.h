/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model.h"
#include "ksana_llm/utils/device_types.h"

#ifdef ENABLE_CUDA
#  include "csrc/utils/nvidia/cuda_utils.h"
#endif

#include <iostream>
#include <random>
#include <type_traits>

namespace ksana_llm {

inline void CopyVectorToHostTensor(Tensor host_tensor, DataType dtype, std::vector<float>& values) {
  KLLM_CHECK(host_tensor.GetElementNumber() == values.size());
  for (size_t idx = 0; idx < host_tensor.GetElementNumber(); idx++) {
    float val = values[idx];
    switch (dtype) {
      case DataType::TYPE_FP32: {
        float* arr = host_tensor.GetPtr<float>();
        arr[idx] = val;
        break;
      }
      case DataType::TYPE_FP16: {
        float16* arr = host_tensor.GetPtr<float16>();
#ifdef ENABLE_ACL
        arr[idx] = aclFloatToFloat16(val);
#else
        arr[idx] = __float2half(val);
#endif
        break;
      }
#ifdef ENABLE_BFLOAT16
      case DataType::TYPE_BF16: {
        bfloat16* arr = host_tensor.GetPtr<bfloat16>();
        arr[idx] = (bfloat16)val;
        break;
      }
#endif
      default:
        assert(false);
    }
  }
}

// Custom random number generator that doesn't use std library's random functions
class CustomRandomGenerator {
 public:
  // Constructor with seed
  explicit CustomRandomGenerator(uint32_t seed = 42) : state_(seed) {}

  // Generate a uniform random number in [0, 1)
  float Uniform() {
    // Linear Congruential Generator parameters
    // Using parameters from Numerical Recipes
    const uint64_t a = 1664525;
    const uint64_t c = 1013904223;
    const uint64_t m = 0xFFFFFFFF;  // 2^32

    // Update state
    state_ = (a * state_ + c) & m;

    // Convert to float in [0, 1)
    return static_cast<float>(state_) / static_cast<float>(m + 1);
  }

  // Generate a normal distributed random number using Box-Muller transform
  float Normal(float mean = 0.0f, float stddev = 0.02f) {
    // Box-Muller transform requires two uniform random numbers
    float u1, u2;

    // Ensure u1 is not exactly 0
    do {
      u1 = Uniform();
    } while (u1 <= 0.0f);

    u2 = Uniform();

    // Box-Muller transform (using 3.14159265358979323846 as π)
    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265358979323846f * u2);

    // Scale and shift to get desired mean and standard deviation
    return mean + z0 * stddev;
  }

  // Fill a vector with normal distributed random numbers
  void FillNormal(std::vector<float>& values, size_t size, float mean = 0.0f, float stddev = 0.02f) {
    values.resize(size);
    for (size_t i = 0; i < size; ++i) {
      values[i] = Normal(mean, stddev);
    }
  }

 private:
  uint32_t state_;  // Random number generator state
};

class FakeWeightValueInitializer {
 public:
  FakeWeightValueInitializer() {}
  ~FakeWeightValueInitializer() = default;
  virtual void InitValues(const std::string& weight_name, int rank, Tensor& weight) {
    // Set to none zero values
    Memset(weight.GetPtr<void>(), 1, weight.GetTotalBytes());
  }
};

class DefaultWeightValueInitializer : public FakeWeightValueInitializer {
 public:
  DefaultWeightValueInitializer() : rng(42) {}
  ~DefaultWeightValueInitializer() = default;
  void InitValues(const std::string& weight_name, int rank, Tensor& weight) override {
    SetDevice(rank);
    std::vector<float> fp32_values;
    InitVector(weight_name, weight.GetElementNumber(), fp32_values);
    Tensor host_weight(MemoryLocation::LOCATION_HOST, weight.dtype, weight.shape, rank);
    CopyVectorToHostTensor(host_weight, host_weight.dtype, fp32_values);
    Memcpy(weight.GetPtr<void>(), host_weight.GetPtr<void>(), weight.GetTotalBytes(), MEMCPY_HOST_TO_DEVICE);
  }

 private:
  void InitVector(const std::string& weight_name, size_t len, std::vector<float>& values) {
    values.resize(len);

    // When weight_name contains "layernorm.weight", all values should be 1
    if (weight_name.find("layernorm.weight") != std::string::npos) {
      std::fill(values.begin(), values.end(), 1.0f);
      return;
    }
    // When weight_name contains ".bias", all values should be 0

    if (weight_name.find(".bias") != std::string::npos) {
      rng.FillNormal(values, len, 0.0f, 0.002f);
      return;
    }
    // For all other cases, values should follow a normal distribution with standard deviation 0.02
    rng.FillNormal(values, len, 0.0f, 0.02f);
  }

 private:
  CustomRandomGenerator rng;
};

class FakeWeight : public BaseWeight {
 public:
  explicit FakeWeight(int rank, FakeWeightValueInitializer* value_initializer = nullptr)
      : rank_(rank), value_initializer_(value_initializer) {}
  ~FakeWeight() {}

  Tensor GetModelWeights(const std::string& weight_name) override {
    if (!weights_map_.count(weight_name)) {
      KLLM_LOG_WARNING << fmt::format("weight name {} not in weights map", weight_name);
      return Tensor();
    }
    return weights_map_[weight_name];
  }

  Status LoadWeightsFromFile(std::shared_ptr<BaseFileTensorLoader>& weights_loader,
                             std::vector<std::string>& weight_name_list, std::vector<std::string>& custom_name_list) {
    return Status();
  }

  void ProcessWeights() {}

  void SetEmbeddingsConfig() {}

 public:
  bool CreateWeight(const std::string& weight_name, const std::vector<size_t> shape, const DataType dtype,
                    const MemoryLocation location) {
    KLLM_CHECK_WITH_INFO(!weights_map_.count(weight_name),
                         fmt::format("weight name {} is in weights map already", weight_name));
    Tensor weight(location, dtype, shape, rank_);
    if (value_initializer_) {
      value_initializer_->InitValues(weight_name, rank_, weight);
    } else {
      non_zero_intializer_.InitValues(weight_name, rank_, weight);
    }
    weights_map_[weight_name] = weight;
    return true;
  }

  void MergeWeight(const std::shared_ptr<FakeWeight>& other) {
    for (const auto& [name, tensor] : other->weights_map_) {
      KLLM_CHECK_WITH_INFO(!weights_map_.count(name), fmt::format("weight name {} is in weights map already", name));
      weights_map_[name] = tensor;
    }
  }

 public:
  int rank_;

  std::map<std::string, Tensor> weights_map_;

  FakeWeightValueInitializer* value_initializer_;
  FakeWeightValueInitializer non_zero_intializer_;
};

class FakeBaseLayersWeight : public FakeWeight {
 public:
  FakeBaseLayersWeight(const ModelConfig& model_config, int rank,
                       FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    DataType dtype = model_config.weight_data_type;
    MemoryLocation location = MemoryLocation::LOCATION_DEVICE;
    KLLM_CHECK_WITH_INFO(model_config.tensor_para_size == 1,
                         fmt::format("only tensor_para_size == 1 supported, given {}", model_config.tensor_para_size));
    // embedding
    CreateWeight("model.embed_tokens.weight", {model_config.vocab_size, model_config.hidden_units}, dtype, location);
    // lm_head_proj_
    CreateWeight("lm_head.weight", {model_config.hidden_units, model_config.vocab_size}, dtype, location);
    // norm
    CreateWeight("model.norm.weight", {model_config.hidden_units}, dtype, location);

    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
      std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
      // input_layernorm
      CreateWeight(layer_prefix + ".input_layernorm.weight", {model_config.hidden_units}, dtype, location);
      // post_attention_layernorm
      CreateWeight(layer_prefix + ".post_attention_layernorm.weight", {model_config.hidden_units}, dtype, location);
    }
  }
};

class FakeMultiHeadAttentionWeight : public FakeWeight {
 public:
  FakeMultiHeadAttentionWeight(const ModelConfig& model_config, int rank, bool add_qkv_bias,
        bool use_qk_norm, FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer), add_qkv_bias_(add_qkv_bias), use_qk_norm_(use_qk_norm) {
    DataType dtype = model_config.weight_data_type;
    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
      InitOneLayer(layer_idx, dtype, (size_t)model_config.hidden_units, (size_t)model_config.head_num);
    }
  }

 private:
  void InitOneLayer(int layer_idx, DataType dtype, size_t hidden_size, size_t total_head_num) {
    size_t head_dim = hidden_size / total_head_num;

    std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
    MemoryLocation location = MemoryLocation::LOCATION_DEVICE;
    // attn_qkv_proj_
    CreateWeight(layer_prefix + ".self_attn.query_key_value.weight", {hidden_size, head_dim * total_head_num}, dtype,
                 location);
    if (add_qkv_bias_) {
      CreateWeight(layer_prefix + ".self_attn.query_key_value.bias", {1, head_dim * total_head_num}, dtype, location);
    }

    if (use_qk_norm_) {
      CreateWeight(layer_prefix + ".self_attn.query_layernorm.weight", {head_dim}, dtype, location);
      CreateWeight(layer_prefix + ".self_attn.key_layernorm.weight", {head_dim}, dtype, location);
    }

    // attn_o_proj_
    CreateWeight(layer_prefix + ".self_attn.o_proj.weight", {head_dim * total_head_num, hidden_size}, dtype, location);
  }

 private:
  bool add_qkv_bias_;
  bool use_qk_norm_;
};

class FakeTwoLayeredFFNWeight : public FakeWeight {
 public:
  FakeTwoLayeredFFNWeight(const ModelConfig& model_config, int rank,
                          FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    DataType dtype = model_config.weight_data_type;
    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
      InitOneLayer(layer_idx, dtype, (size_t)model_config.hidden_units, (size_t)model_config.inter_size);
    }
  }

 private:
  void InitOneLayer(int layer_idx, DataType dtype, size_t hidden_size, size_t intermedia_size) {
    std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
    MemoryLocation location = MemoryLocation::LOCATION_DEVICE;
    // mlp_gate_proj_
    CreateWeight(layer_prefix + ".mlp.gate_proj.weight", {hidden_size, intermedia_size}, dtype, location);
    // mlp_up_proj_
    CreateWeight(layer_prefix + ".mlp.up_proj.weight", {hidden_size, intermedia_size}, dtype, location);
    // mlp_down_proj_
    CreateWeight(layer_prefix + ".mlp.down_proj.weight", {intermedia_size, hidden_size}, dtype, location);
  }
};

class FakeMoeLayersWeight : public FakeWeight {
 public:
  FakeMoeLayersWeight(const ModelConfig& model_config, int rank, bool use_shared_moe,
                      FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer), use_shared_moe_(use_shared_moe) {
    DataType dtype = model_config.weight_data_type;
    const std::vector<size_t>& moe_layers = model_config.moe_config.moe_layers;
    size_t shared_expert_inter_size = (size_t)model_config.moe_config.shared_expert_inter_size;
    size_t moe_inter_size = (size_t)model_config.moe_config.moe_inter_size;
    size_t inter_size = (size_t)model_config.inter_size;
    size_t hidden_units = (size_t)model_config.hidden_units;
    size_t num_experts = (size_t)model_config.moe_config.num_experts;
    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
      InitOneLayer(layer_idx, dtype, hidden_units, shared_expert_inter_size, num_experts, moe_inter_size, inter_size,
                   moe_layers);
    }
  }

 private:
  void InitOneLayer(int layer_idx, DataType dtype, size_t hidden_size, size_t shared_expert_inter_size,
                    size_t num_experts, size_t moe_inter_size, size_t inter_size,
                    const std::vector<size_t>& moe_layers) {
    std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
    MemoryLocation location = MemoryLocation::LOCATION_DEVICE;
    bool is_moe_layer =
        (moe_layers.empty() || std::find(moe_layers.begin(), moe_layers.end(), layer_idx) != moe_layers.end());
    if (is_moe_layer) {
      // expert_gates_
      CreateWeight(layer_prefix + ".mlp.gate.weight", {hidden_size, num_experts}, dtype, location);
      // moes_
      CreateWeight(layer_prefix + ".mlp.experts.up_gate_proj.weight", {num_experts, moe_inter_size * 2, hidden_size},
                   dtype, location);
      CreateWeight(layer_prefix + ".mlp.experts.down_proj.weight", {num_experts, hidden_size, moe_inter_size}, dtype,
                   location);

      if (use_shared_moe_) {
        // shared_mlps_
        CreateWeight(layer_prefix + ".mlp.shared_expert.gate_proj.weight", {hidden_size, shared_expert_inter_size},
                     dtype, location);
        CreateWeight(layer_prefix + ".mlp.shared_expert.up_proj.weight", {hidden_size, shared_expert_inter_size}, dtype,
                     location);
        CreateWeight(layer_prefix + ".mlp.shared_expert.down_proj.weight", {shared_expert_inter_size, hidden_size},
                     dtype, location);
        // shared_expert_gates_
        CreateWeight(layer_prefix + ".mlp.shared_expert_gate.weight", {1, hidden_size}, dtype, location);
      }
    } else {
      // mlp_gate_proj_
      CreateWeight(layer_prefix + ".mlp.gate_proj.weight", {hidden_size, inter_size}, dtype, location);
      // mlp_up_proj_
      CreateWeight(layer_prefix + ".mlp.up_proj.weight", {hidden_size, inter_size}, dtype, location);
      // mlp_down_proj_
      CreateWeight(layer_prefix + ".mlp.down_proj.weight", {inter_size, hidden_size}, dtype, location);
    }
  }

 private:
  bool use_shared_moe_;
};

class FakeGptBaseBiasLayersWeight : public FakeWeight {
 public:
  FakeGptBaseBiasLayersWeight(const ModelConfig& model_config, int rank,
                              FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    DataType dtype = model_config.weight_data_type;
    MemoryLocation location = MemoryLocation::LOCATION_DEVICE;
    KLLM_CHECK_WITH_INFO(model_config.tensor_para_size == 1,
                         fmt::format("only tensor_para_size == 1 supported, given {}", model_config.tensor_para_size));
    CreateWeight("model.embed_positions.weight", {model_config.max_token_num, model_config.hidden_units}, dtype,
                 location);
    for (uint32_t layer_idx = 0; layer_idx < model_config.num_layer; layer_idx++) {
      std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
      // input_layernorm bias
      CreateWeight(layer_prefix + ".input_layernorm.bias", {model_config.hidden_units}, dtype, location);
      // post_attention_layernorm bias
      CreateWeight(layer_prefix + ".post_attention_layernorm.bias", {model_config.hidden_units}, dtype, location);
      // attn proj bias
      CreateWeight(layer_prefix + ".self_attn.o_proj.bias", {model_config.hidden_units}, dtype, location);
      // mlp gate bias add
      CreateWeight(layer_prefix + ".mlp.gate_proj.bias", {4 * model_config.hidden_units}, dtype, location);
      // mlp down bias add
      CreateWeight(layer_prefix + ".mlp.down_proj.bias", {model_config.hidden_units}, dtype, location);
      // query_key_value bias add
      size_t head_dim = (size_t)model_config.hidden_units / (size_t)model_config.head_num;
      size_t total_head_num = (size_t)model_config.head_num;
      CreateWeight(layer_prefix + ".self_attn.query_key_value.bias", {head_dim * total_head_num}, dtype, location);
    }
  }
};

class FakeSimpleWeight : public FakeWeight {
 public:
  FakeSimpleWeight(const ModelConfig& model_config, int rank, bool add_qkv_bias, bool use_shared_moe,
                   bool use_qk_norm, FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    auto base_weight = std::make_shared<FakeBaseLayersWeight>(model_config, rank, value_initializer);
    auto mha_weight =
        std::make_shared<FakeMultiHeadAttentionWeight>(model_config, rank, add_qkv_bias, use_qk_norm,
                                                       value_initializer);
    auto mlp_weight = std::make_shared<FakeTwoLayeredFFNWeight>(model_config, rank, value_initializer);

    MergeWeight(base_weight);
    MergeWeight(mha_weight);
    MergeWeight(mlp_weight);
  }
};

class FakeGptSimpleWeight : public FakeWeight {
 public:
  FakeGptSimpleWeight(const ModelConfig& model_config, int rank, bool add_qkv_bias, bool use_shared_moe,
                      FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    auto base_weight = std::make_shared<FakeBaseLayersWeight>(model_config, rank, value_initializer);
    auto bias_weight = std::make_shared<FakeGptBaseBiasLayersWeight>(model_config, rank, value_initializer);
    auto mha_weight =
        std::make_shared<FakeMultiHeadAttentionWeight>(model_config, rank, add_qkv_bias, value_initializer);
    auto mlp_weight = std::make_shared<FakeTwoLayeredFFNWeight>(model_config, rank, value_initializer);
    MergeWeight(base_weight);
    MergeWeight(mha_weight);
    MergeWeight(mlp_weight);
    MergeWeight(bias_weight);
  }
};

class FakeMoeWeight : public FakeWeight {
 public:
  FakeMoeWeight(const ModelConfig& model_config, int rank, bool add_qkv_bias, bool use_shared_moe,
                bool use_qk_norm, FakeWeightValueInitializer* value_initializer = nullptr)
      : FakeWeight(rank, value_initializer) {
    auto base_weight = std::make_shared<FakeBaseLayersWeight>(model_config, rank, value_initializer);
    auto mha_weight =
        std::make_shared<FakeMultiHeadAttentionWeight>(model_config, rank, add_qkv_bias, use_qk_norm,
                                                       value_initializer);
    auto moe_weight = std::make_shared<FakeMoeLayersWeight>(model_config, rank, use_shared_moe, value_initializer);

    MergeWeight(base_weight);
    MergeWeight(mha_weight);
    MergeWeight(moe_weight);
  }
};

}  // namespace ksana_llm
