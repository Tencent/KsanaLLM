/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/mixtral/mixtral_model.h"

namespace ksana_llm {

template <typename T>
MixtralDecoderLayer<T>::MixtralDecoderLayer(int layer_idx, LayerCreationContext<T>& creation_context,
                                            ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
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
  mha_ = std::make_shared<MultiHeadAttention<T>>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                                 model_creation_config);

  // MoE related blocks
  expert_gates_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  moes_ = std::make_shared<MoE<T>>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                   layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                   MoeScaleNormMode::RE_NORM);
}

template <typename T>
Status MixtralDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                       ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);
  auto& gated_buffer_ = reduce_buffer_tensors;

  // Pre attn layernorm
  // Pre layernorm uses layernorm input for residual connection.
  input_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // MultiHeadAttention
  STATUS_CHECK_RETURN(
      mha_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Attn residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  // Pre mlp layernorm
  // Pre layernorm uses layernorm input for residual connection.
  post_attention_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // Expert gating MatMul
  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0, gated_buffer_));

  // MOE layer
  moes_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], hidden_buffer_tensors_0);

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  return Status();
}

template <typename T>
Status Mixtral<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Mixtral<T>::CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) {
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] =
        std::make_shared<MixtralDecoderLayer<T>>(layer_idx, creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status Mixtral<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Mixtral<float>;
template class Mixtral<float16>;
#ifdef ENABLE_BFLOAT16
template class Mixtral<bfloat16>;
#endif

/* **************************************
 * MixtralModel
 */
template <typename T>
MixtralModel<T>::MixtralModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                              std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  mixtral_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status MixtralModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                     ModelCreationConfig& model_creation_config) {
  return mixtral_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status MixtralModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(mixtral_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class MixtralModel<float>;
template class MixtralModel<float16>;
#ifdef ENABLE_BFLOAT16
template class MixtralModel<bfloat16>;
#endif

}  // namespace ksana_llm
