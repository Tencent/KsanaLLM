/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen3_moe/qwen3_moe_model.h"

namespace ksana_llm {

template <typename T>
Qwen3MoeDecoderLayer<T>::Qwen3MoeDecoderLayer(int layer_idx, TensorBuffer* moe_buffer,
                                              LayerCreationContext<T>& creation_context,
                                              ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), moe_buffer_(moe_buffer) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  // Common blocks
  adds_ = std::make_shared<Add<T>>(creation_context);
  tp_comm_ = std::make_shared<TpCommunicator<T>>();

  input_layernorms_ = std::make_shared<Layernorm<T>>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".post_attention_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);

  bool is_neox = true;
  bool add_qkv_bias = false;
  bool use_qk_norm = false;
  KLLM_LOG_DEBUG << "use_qk_norm: " << use_qk_norm;
  KLLM_LOG_DEBUG << "add_qkv_bias: " << add_qkv_bias;
  mha_ = std::make_shared<MultiHeadAttention<T>>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                                 model_creation_config);
  // // MoE related blocks
  expert_gates_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  moes_ = std::make_shared<MoE<T>>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                   layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                   MoeScaleNormMode::RE_NORM);
}

template <typename T>
Status Qwen3MoeDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                        ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);
  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);

  input_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // MultiHeadAttention
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  // Pre layernorm uses layernorm input for residual connection.
  post_attention_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0, moe_buffer_tensors));

  // MOE layer
  STATUS_CHECK_RETURN(moes_->Forward(hidden_buffer_tensors_0[0], moe_buffer_tensors[0], reduce_buffer_tensors));

  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  return Status();
}

template <typename T>
Status Qwen3Moe<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Qwen3Moe<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                 ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  model_creation_config.attn_config.model_config.moe_config.norm_topk_prob = true;
  DataType weight_type = model_config.weight_data_type;
  size_t max_token_num = model_creation_config.runtime_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_config.hidden_units;
  KLLM_LOG_DEBUG << "moe_buffer_size: " << moe_buffer_size;
  KLLM_LOG_DEBUG << "max_token_num: " << max_token_num;
  KLLM_LOG_DEBUG << "weight_type: " << model_config.weight_data_type;
  moe_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] =
        std::make_shared<Qwen3MoeDecoderLayer<T>>(layer_idx, moe_buffer_, creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status Qwen3Moe<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Qwen3Moe<float>;
template class Qwen3Moe<float16>;
template class Qwen3Moe<bfloat16>;

/* **************************************
 * Qwen3MoeModel
 */
template <typename T>
Qwen3MoeModel<T>::Qwen3MoeModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                                std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  qwen3moe_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Qwen3MoeModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                      ModelCreationConfig& model_creation_config) {
  model_creation_config.attn_config.model_config.enable_qk_pre_norm_before_rotary_pos = true;
  return qwen3moe_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status Qwen3MoeModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(qwen3moe_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class Qwen3MoeModel<float>;
template class Qwen3MoeModel<float16>;
template class Qwen3MoeModel<bfloat16>;

}  // namespace ksana_llm
