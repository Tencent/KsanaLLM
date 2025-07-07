/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/baichuan/baichuan_model.h"

#include <memory>
#include <vector>
namespace ksana_llm {

template <typename T>
Status Baichuan<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  // The Baichuan1-7B and Baichuan2-7B models do not use the Alibi mode for loading,
  // and both of these models have a hidden_units value of 4096.
  model_run_config.position_encoding =
      (model_config.size_per_head * model_config.head_num != 4096) ? PositionEncoding::ALIBI : PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Baichuan<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                 ModelCreationConfig& model_creation_config) {
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
Status Baichuan<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.GetModelInput()->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.GetPipelineConfig().lower_layer_idx;
       layer_idx <= forwarding_context.GetPipelineConfig().upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Baichuan<float>;
template class Baichuan<float16>;
#ifdef ENABLE_BFLOAT16
template class Baichuan<bfloat16>;
#endif

/* **************************************
 * BaichuanModel
 */
template <typename T>
BaichuanModel<T>::BaichuanModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                                std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  baichuan_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status BaichuanModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                      ModelCreationConfig& model_creation_config) {
  return baichuan_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status BaichuanModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.GetContext()->IsChief());
  STATUS_CHECK_RETURN(baichuan_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class BaichuanModel<float>;
template class BaichuanModel<float16>;
#ifdef ENABLE_BFLOAT16
template class BaichuanModel<bfloat16>;
#endif

}  // namespace ksana_llm
