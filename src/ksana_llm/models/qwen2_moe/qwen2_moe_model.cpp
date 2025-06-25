/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen2_moe/qwen2_moe_model.h"

namespace ksana_llm {

template <typename T>
Qwen2MoeDecoderLayer<T>::Qwen2MoeDecoderLayer(int layer_idx, TensorBuffer* moe_buffer,
                                              TensorBuffer* share_gating_buffer,
                                              LayerCreationContext<T>& creation_context,
                                              ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), moe_buffer_(moe_buffer), share_gating_buffer_(share_gating_buffer) {
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
  bool add_qkv_bias = true;
  bool use_qk_norm = false;
  mha_ = std::make_shared<MultiHeadAttention<T>>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                                 model_creation_config);

  // MoE related blocks
  expert_gates_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  moes_ = std::make_shared<MoE<T>>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                   layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                   MoeScaleNormMode::NO_NORM);
  shared_mlps_ = std::make_shared<TwoLayeredFFN<T>>(layer_idx, creation_context, model_creation_config,
                                                    ".mlp.shared_expert.{}.weight");
  shared_expert_gates_ =
      std::make_shared<Linear<T>>(layer_prefix + ".mlp.shared_expert_gate.weight", creation_context,
                                  model_creation_config.attn_config.model_config.quant_config.backend);

#ifdef ENABLE_CUDA
  muls_ = std::make_shared<Mul<T>>(creation_context);
  sigmoids_ = std::make_shared<Sigmoid<T>>(creation_context);
#endif
}

template <typename T>
Status Qwen2MoeDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                        ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);

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

  // Common mlp
  STATUS_CHECK_RETURN(
      ForwardMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  return Status();
}

template <typename T>
Status Qwen2MoeDecoderLayer<T>::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                           std::vector<Tensor>& reduce_buffer_tensors,
                                           const bool is_multi_token_forward,
                                           ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
  CREATE_BUFFER_SCOPE(share_gating_buffer_tensors, share_gating_buffer_);
  auto& gated_buffer_ = reduce_buffer_tensors;

  // Expert gating MatMul
  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0, gated_buffer_));

  // MOE layer
  moes_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], moe_buffer_tensors);

  // Expert share gating MatMul
  STATUS_CHECK_RETURN(shared_expert_gates_->Forward(hidden_buffer_tensors_0, share_gating_buffer_tensors));

  shared_mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);

#ifdef ENABLE_CUDA
  // Expert share gating sigmoid
  STATUS_CHECK_RETURN(sigmoids_->Forward(share_gating_buffer_tensors, share_gating_buffer_tensors));

  // hidden_buffer_1_ is used in shared_mlps_
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);

  // Expert share gating mul
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(
        muls_->Forward(share_gating_buffer_tensors[0], reduce_buffer_tensors[0], hidden_buffer_tensors_1));
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], moe_buffer_tensors[0], reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(
        muls_->Forward(share_gating_buffer_tensors[0], hidden_buffer_tensors_0[0], hidden_buffer_tensors_1));
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], moe_buffer_tensors[0], hidden_buffer_tensors_0));
  }
#endif
  return Status();
}

template <typename T>
Status Qwen2Moe<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Qwen2Moe<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                 ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;

  size_t max_token_num = model_config.max_step_token_num;
  share_gating_buffer_ =
      creation_context.buffer_mgr_->CreateBufferTensor("share_gating_buffer_", {max_token_num}, weight_type);

  size_t moe_buffer_size = max_token_num * model_config.hidden_units;
  moe_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<Qwen2MoeDecoderLayer<T>>(layer_idx, moe_buffer_, share_gating_buffer_,
                                                                           creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status Qwen2Moe<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Qwen2Moe<float>;
template class Qwen2Moe<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2Moe<bfloat16>;
#endif

/* **************************************
 * Qwen2MoeModel
 */
template <typename T>
Qwen2MoeModel<T>::Qwen2MoeModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                                std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  qwen2moe_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Qwen2MoeModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                      ModelCreationConfig& model_creation_config) {
  return qwen2moe_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status Qwen2MoeModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(qwen2moe_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class Qwen2MoeModel<float>;
template class Qwen2MoeModel<float16>;
#ifdef ENABLE_BFLOAT16
template class Qwen2MoeModel<bfloat16>;
#endif

}  // namespace ksana_llm
