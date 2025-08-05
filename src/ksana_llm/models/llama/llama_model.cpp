/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/llama/llama_model.h"

#include <memory>
#include <vector>
namespace ksana_llm {

template <typename T>
Status Llama<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Llama<T>::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  bool is_neox = true;
  bool add_qkv_bias = false;
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<SimpleDecoderLayer<T>>(layer_idx, is_neox, add_qkv_bias,
                                                                         creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status Llama<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Llama<float>;
template class Llama<float16>;
template class Llama<bfloat16>;

/* **************************************
 * LlamaModel
 */
template <typename T>
LlamaModel<T>::LlamaModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                          std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  llama_.GetModelRunConfig(model_run_config, model_config);
  CommonModel::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status LlamaModel<T>::CreateLayers(LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config) {
  return llama_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status LlamaModel<T>::LayerForward(ForwardingContext& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(llama_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class LlamaModel<float>;
template class LlamaModel<float16>;
template class LlamaModel<bfloat16>;

}  // namespace ksana_llm
