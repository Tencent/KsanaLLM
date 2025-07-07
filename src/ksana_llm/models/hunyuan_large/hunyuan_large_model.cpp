/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/hunyuan_large/hunyuan_large_model.h"

namespace ksana_llm {

template <typename T>
HunyuanDecoderLayer<T>::HunyuanDecoderLayer(int layer_idx, TensorBuffer* moe_buffer, int cla_share_factor,
                                            ClaBuffers& cla_buffers, LayerCreationContext<T>& creation_context,
                                            ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx), moe_buffer_(moe_buffer) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  // Common blocks
  adds_ = std::make_shared<Add<T>>(creation_context);
  input_layernorms_ = std::make_shared<Layernorm<T>>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm<T>>(layer_prefix + ".post_attention_layernorm.weight",
                                     model_creation_config.layernorm_config.layernorm_eps, creation_context);
  tp_comm_ = std::make_shared<TpCommunicator<T>>();

  cla_ = std::make_shared<CrossLayerAttention<T>>(layer_idx, cla_share_factor, cla_buffers, creation_context,
                                                  model_creation_config);

  shared_mlps_ = std::make_shared<TwoLayeredFFN<T>>(layer_idx, creation_context, model_creation_config,
                                                    ".mlp.shared_expert.{}.weight");

  // MoE related blocks
  expert_gates_ = std::make_shared<Linear<T>>(layer_prefix + ".mlp.gate.weight", creation_context,
                                              model_creation_config.attn_config.model_config.quant_config.backend);
  moes_ = std::make_shared<MoE<T>>(layer_prefix + ".mlp.experts.up_gate_proj.weight",
                                   layer_prefix + ".mlp.experts.down_proj.weight", creation_context,
                                   MoeScaleNormMode::NO_NORM);
}

template <typename T>
Status HunyuanDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                                       ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_0, forwarding_context.GetForwardingBuffers()->hidden_buffer_0);
  CREATE_BUFFER_SCOPE(reduce_buffer_tensors, forwarding_context.GetForwardingBuffers()->shared_buffer);
  // Pre attn layernorm
  // Pre layernorm uses layernorm input for residual connection.
  input_layernorms_->Forward(residual_buffer, hidden_buffer_tensors_0);

  // Cla attention
  STATUS_CHECK_RETURN(
      cla_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);

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
Status HunyuanDecoderLayer<T>::ForwardMlp(std::vector<Tensor>& hidden_buffer_tensors_0,
                                          std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                          ForwardingContext<T>& forwarding_context) {
  CREATE_BUFFER_SCOPE(moe_buffer_tensors, moe_buffer_);
  auto& gated_buffer_ = reduce_buffer_tensors;
  // Expert gating MatMul
  STATUS_CHECK_RETURN(expert_gates_->Forward(hidden_buffer_tensors_0, gated_buffer_));

  // MOE layer
  moes_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], moe_buffer_tensors);

  // shared expert
  shared_mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context);

  // Add moe output and share_expert output
  if (forwarding_context.GetModelCommunicator()) {
    STATUS_CHECK_RETURN(adds_->Forward(reduce_buffer_tensors[0], moe_buffer_tensors[0], reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], moe_buffer_tensors[0], hidden_buffer_tensors_0));
  }

  return Status();
}

/************************************************************************
 * HunyuanLargeModel
 ************************************************************************/

template <typename T>
HunyuanLargeModel<T>::HunyuanLargeModel(const ModelConfig& model_config, const int rank,
                                        std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context), cla_share_factor_(model_config.cla_share_factor) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status HunyuanLargeModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                          ModelCreationConfig& model_creation_config) {
  auto& model_config = model_creation_config.attn_config.model_config;
  DataType weight_type = model_config.weight_data_type;

  CrossLayerAttention<T>::CreateBuffers(CommonModel<T>::GetBufferManager(), model_creation_config.attn_config,
                                        cla_buffers_);

  size_t max_token_num = model_config.max_step_token_num;
  size_t moe_buffer_size = max_token_num * model_config.hidden_units;

  moe_buffer_ = CommonModel<T>::GetBufferManager()->CreateBufferTensor("moe_buffer_", {moe_buffer_size}, weight_type);

  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<HunyuanDecoderLayer<T>>(
        layer_idx, moe_buffer_, cla_share_factor_, cla_buffers_, creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status HunyuanLargeModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;

  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class HunyuanLargeModel<float>;
template class HunyuanLargeModel<float16>;
#ifdef ENABLE_BFLOAT16
template class HunyuanLargeModel<bfloat16>;
#endif
}  // namespace ksana_llm
