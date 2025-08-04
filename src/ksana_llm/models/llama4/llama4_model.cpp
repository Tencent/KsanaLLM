/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama4/llama4_model.h"

namespace ksana_llm {

template <typename T>
Llama4DecoderLayer<T>::Llama4DecoderLayer(int layer_idx, TensorBuffer* moe_buffer, bool is_moe_layer,
                                          LayerCreationContext& creation_context,
                                          ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), moe_buffer_(moe_buffer), is_moe_layer_(is_moe_layer) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  // Common blocks
  adds_ = std::make_shared<Add<T>>(creation_context);
  tp_comm_ = std::make_shared<TpCommunicator<T>>();

  input_layernorms_ = std::make_shared<Layernorm<T>>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".post_attention_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);

  bool is_neox = false;
  bool add_qkv_bias = false;
  bool use_qk_norm = model_creation_config.attn_config.model_config.use_qk_norm;
  mha_ = std::make_shared<MultiHeadAttention<T>>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                                 model_creation_config);
  if (is_moe_layer) {
    // MoE related blocks
    expert_gates_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate.weight", creation_context,
                                                model_creation_config.attn_config.model_config.quant_config.backend);
    moes_ = std::make_shared<MoE<T>>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                     layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                     MoeScaleNormMode::NO_NORM);
    std::vector<std::string> shared_expert_weight_names = {".mlp.shared_expert.gate_proj.weight",
                                                           ".mlp.shared_expert.up_proj.weight",
                                                           ".mlp.shared_expert.down_proj.weight"};
    mlps_ = std::make_shared<TwoLayeredFFN<T>>(layer_idx, creation_context, model_creation_config,
                                               ".mlp.shared_expert.{}.weight");
  } else {
    mlps_ = std::make_shared<TwoLayeredFFN<T>>(layer_idx, creation_context, model_creation_config);
  }
}

template <typename T>
Status Llama4DecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                      ForwardingContext& forwarding_context) {
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

  if (is_moe_layer_) {
    // Common mlp
    STATUS_CHECK_RETURN(
        ForwardMlp(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  } else {
    // Common mlp
    STATUS_CHECK_RETURN(
        mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  }

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));

  return Status();
}

template <typename T>
Status Llama4DecoderLayer<T>::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                         std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                         ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
  auto& gated_buffer_ = reduce_buffer_tensors;

  // Expert gating MatMul
  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0, gated_buffer_));

  // MOE layer
  moes_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], moe_buffer_tensors);

  mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);

  // Add moe output and share_expert output
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(adds_->Forward(reduce_buffer_tensors[0], moe_buffer_tensors[0], reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], moe_buffer_tensors[0], hidden_buffer_tensors_0));
  }

  return Status();
}

template <typename T>
Status Llama4<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Llama4<T>::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;

  size_t max_token_num = creation_context.runtime_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_config.hidden_units;

  moe_buffer_ = creation_context.buffer_mgr_->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  PositionEncoding position_encoding = model_creation_config.attn_config.position_encoding;
  std::vector<size_t>& moe_layers = model_config.moe_config.moe_layers;
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    bool no_rope = (model_config.no_rope_layers[layer_idx] == 0);
    model_creation_config.attn_config.position_encoding = no_rope ? PositionEncoding::NO_ROPE : position_encoding;
    bool is_moe_layer =
        (moe_layers.empty() || std::find(moe_layers.begin(), moe_layers.end(), layer_idx) != moe_layers.end());
    decoder_layers_[layer_idx] = std::make_shared<Llama4DecoderLayer<T>>(layer_idx, moe_buffer_, is_moe_layer,
                                                                         creation_context, model_creation_config);
    // revert config
    model_creation_config.attn_config.position_encoding = position_encoding;
  }
  return Status();
}

template <typename T>
Status Llama4<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Llama4<float>;
template class Llama4<float16>;
template class Llama4<bfloat16>;

/* **************************************
 * Llama4Model
 */
template <typename T>
Llama4Model<T>::Llama4Model(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                            std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  Llama4_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Llama4Model<T>::CreateLayers(LayerCreationContext& creation_context,
                                    ModelCreationConfig& model_creation_config) {
  return Llama4_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status Llama4Model<T>::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(Llama4_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class Llama4Model<float>;
template class Llama4Model<float16>;
template class Llama4Model<bfloat16>;

}  // namespace ksana_llm
