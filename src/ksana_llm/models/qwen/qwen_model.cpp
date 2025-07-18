/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/qwen/qwen_model.h"

#include <memory>
#include <vector>
namespace ksana_llm {

template <typename T>
Status Qwen<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Qwen<T>::CreateLayers(LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config) {
  bool is_neox = true;
  bool add_qkv_bias = model_creation_config.attn_config.model_config.enable_add_qkv_bias;
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<SimpleDecoderLayer<T>>(layer_idx, is_neox, add_qkv_bias,
                                                                         creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status Qwen<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Qwen<float>;
template class Qwen<float16>;
template class Qwen<bfloat16>;

/* **************************************
 * QwenModel
 */
template <typename T>
QwenModel<T>::QwenModel(const ModelConfig& model_config, const RuntimeConfig& runtime_config, const int rank,
                        std::shared_ptr<Context> context, std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, runtime_config, rank, context) {
  ModelRunConfig model_run_config;
  qwen_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status QwenModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                  ModelCreationConfig& model_creation_config) {
  return qwen_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status QwenModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(qwen_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class QwenModel<float>;
template class QwenModel<float16>;
template class QwenModel<bfloat16>;

}  // namespace ksana_llm
