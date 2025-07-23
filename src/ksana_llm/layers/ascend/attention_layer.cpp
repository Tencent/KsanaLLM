/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/attention_layer.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

template <typename T>
Status AttentionLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                               std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  int parameter_index = 0;
  mm_quant_mode_ = std::any_cast<const QuantMode>(parameters[parameter_index++]);
  layernorm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);
  use_qk_norm_ = std::any_cast<const bool>(parameters[parameter_index++]);
  layer_index_ = std::any_cast<const int>(parameters[parameter_index++]);
  layer_num_ = std::any_cast<const int>(parameters[parameter_index++]);
  max_position_embeddings_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  num_kv_heads_ = std::any_cast<const int>(parameters[parameter_index++]);
  head_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  stride_size_ = std::any_cast<const int>(parameters[parameter_index++]);
  tensor_para_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);
  kv_cache_dtype_ = std::any_cast<DataType>(parameters[parameter_index++]);
  k_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  v_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  uint32_t rotary_dim = std::any_cast<const int>(parameters[parameter_index++]);
  base_ = std::any_cast<const float>(parameters[parameter_index++]);
  // TODO(karlluo): NPU support MLA
  qk_rope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  qk_nope_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  q_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  kv_lora_rank_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);
  v_head_dim_ = std::any_cast<const uint32_t>(parameters[parameter_index++]);

  bool is_neox = std::any_cast<const bool>(parameters[parameter_index++]);
  PositionEncoding position_encoding = std::any_cast<const PositionEncoding>(parameters[parameter_index++]);
  void* cos_sin_cache_ptr = std::any_cast<void*>(parameters[parameter_index++]);
  RoPEScalingFactor rope_scaling_factor_config = std::any_cast<const RoPEScalingFactor>(parameters[parameter_index++]);
  max_batch_size_ = std::any_cast<const size_t>(parameters[parameter_index++]);

  attn_temperature_tuning_ = std::any_cast<const size_t>(parameters[parameter_index++]) > 0;
  attn_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  floor_scale_ = std::any_cast<const size_t>(parameters[parameter_index++]);

  is_multi_token_forward_ = std::any_cast<const bool>(parameters[parameter_index++]);

  // TODO(zhongzhicao): The cast should be removed after implementing ROPE.
  // Cast the unused variables to void to suppress the -Wunused-value warnings.
  (void)rotary_dim;
  (void)is_neox;
  (void)position_encoding;
  (void)cos_sin_cache_ptr;

  block_size_ = runtime_config.attn_backend_config.block_size;
  block_token_num_ = runtime_config.attn_backend_config.block_token_num;
  return Status();
}

template class AttentionLayer<float>;
template class AttentionLayer<float16>;
template class AttentionLayer<bfloat16>;

}  // namespace ksana_llm
