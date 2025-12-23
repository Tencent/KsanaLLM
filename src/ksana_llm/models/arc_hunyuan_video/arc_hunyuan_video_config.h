/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/base/base_model_config.h"
#include "ksana_llm/models/common/common_config.h"

namespace ksana_llm {

inline void PrepareArcHunyuanVideoAttributes(const nlohmann::json &config_json, ModelConfig &model_config) {
  PrepareCommonModelAttributes(config_json, model_config);
  model_config.enable_add_qkv_bias = config_json.value("attention_bias", false);
  model_config.use_qk_norm = config_json.value("use_qk_norm", true);
  model_config.mlp_bias = config_json.value("mlp_bias", false);
  model_config.size_per_head = config_json.value("attention_head_dim", 128);
  model_config.position_embedding_xdrope = config_json.value("position_embedding_xdrope", true);
}

// Model config for arc_hunyuan_video-dense architecture.
struct ArcHunyuanVideoConfig : public BaseModelConfig {
  virtual ~ArcHunyuanVideoConfig() {}

  // Type of weight
  DataType weight_data_type;

  std::string tokenizer_path;

  // The max number of (input + output tokens)
  size_t max_token_num;

  size_t max_step_token_num;

  size_t tensor_para_size;

  size_t head_groups;
  size_t head_num;
  uint32_t size_per_head;
  uint32_t inter_size;
  uint32_t hidden_units;
  uint32_t num_layer;
  uint32_t rotary_embedding;
  float rope_theta;
  float layernorm_eps;
  uint32_t vocab_size;
  uint32_t start_id;
  uint32_t end_id;
  uint32_t pad_id;
  size_t num_key_value_heads;
  int max_batch_size;
  int max_position_embeddings;
  size_t block_token_num;

  RoPEScalingFactor rope_scaling_factor_config;

  bool tie_word_embeddings;
  bool exist_tie_embeddings_param = true;

  // The activation function used.
  std::string activation_function{"swiglu"};

  // Text config specific parameters
  bool attention_bias = false;
  float attention_dropout = 0.0f;
  uint32_t eod_token_id = 0;
  uint32_t im_end_id = 0;
  uint32_t im_newline_id = 0;
  uint32_t im_start_id = 0;
  uint32_t image_token_id = 0;
  float initializer_range = 0.02f;
  bool is_causal = true;
  bool mlp_bias = false;
  std::string norm_type = "hf_rms";
  uint32_t num_media_embeds = 0;
  bool position_embedding_xdrope = false;
  bool use_qk_norm = true;
  bool use_rotary_pos_emb = true;

  bool is_visual = true;
};

}  // namespace ksana_llm
