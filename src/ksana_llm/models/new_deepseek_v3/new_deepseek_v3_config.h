/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <regex>
#include <vector>

#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/environment.h"

namespace ksana_llm {
struct NewDeepSeekV3Config : public BaseModelConfig {
  virtual ~NewDeepSeekV3Config() {}

  bool ContainGptqWeights() const;

  QuantConfig GetGptqQuantConfig();

  bool IsWeightMatchGptq(const std::string& weight_name);

  bool IsGptqContain(const std::string& weight_name);

  std::string type;

  // Type of weight
  DataType weight_data_type;

  std::string tokenizer_path;

  // The max number of (input + output tokens)
  size_t max_token_num;

  size_t max_scheduler_token_num;

  int tensor_para_size;
  int attn_data_para_size;

  // The expert parallel size
  int expert_para_size;
  int moe_tensor_para_size;

  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t num_nextn_predict_layers = 0;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  std::vector<uint32_t> end_ids;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;
  std::vector<float> k_scales;
  std::vector<float> v_scales;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Determines if the model is a visual llm model.
  bool is_visual = false;

  // Determines if the model is a quant model.
  bool is_quant;
  QuantConfig quant_config;
  std::vector<QuantConfig> sub_quant_configs;

  // Config for moe
  bool is_moe = false;
  bool has_shared_experts = false;
  MoeConfig moe_config;
  ExpertParallelConfig expert_parallel_config;

  // Config for mla
  bool use_mla = false;
  MlaConfig mla_config;
  // Config for dsa
  bool use_dsa = false;
  DsaConfig dsa_config;

  // others attributes
  std::unordered_map<std::string, std::string> model_attributes;

  ModelFileFormat model_file_format;

  bool load_bias = true;     // Check if load all weights bias.
  int cla_share_factor = 0;  // Determines the number of layers that share k and v.
  bool use_qk_norm = false;  // Check if normlize the attention out q and k.

  std::vector<std::pair<std::regex, std::string>> w4a8_patterns_ = {
      std::make_pair(std::regex(R"(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.weight_scale_inv)"),
                     std::string("model.layers.$1.mlp.experts.$2.$3.scales")),
      std::make_pair(std::regex(R"(model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.([^.]+)\.weight)"),
                     std::string("model.layers.$1.mlp.experts.$2.$3.qweight")),
  };
};
}  // namespace ksana_llm
