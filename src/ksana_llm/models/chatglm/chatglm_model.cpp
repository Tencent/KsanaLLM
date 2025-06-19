/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/chatglm/chatglm_model.h"

#include <memory>
#include <vector>
namespace ksana_llm {

template <typename T>
ChatglmModel<T>::ChatglmModel(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                              std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status ChatglmModel<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                     ModelCreationConfig& model_creation_config) {
  bool is_neox = false;
  bool add_qkv_bias = true;
  for (int layer_idx = creation_context.pipeline_config.lower_layer_idx;
       layer_idx <= creation_context.pipeline_config.upper_layer_idx; layer_idx++) {
    decoder_layers_[layer_idx] = std::make_shared<SimpleDecoderLayer<T>>(layer_idx, is_neox, add_qkv_bias,
                                                                         creation_context, model_creation_config);
  }
  return Status();
}

template <typename T>
Status ChatglmModel<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  const bool is_multi_token_forward = forwarding_context.model_input_->multi_token_request_num > 0;

  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.context_->IsChief());
  for (int layer_idx = forwarding_context.pipeline_config_.lower_layer_idx;
       layer_idx <= forwarding_context.pipeline_config_.upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);

  return Status();
}

template class ChatglmModel<float>;
template class ChatglmModel<float16>;
#ifdef ENABLE_BFLOAT16
template class ChatglmModel<bfloat16>;
#endif

}  // namespace ksana_llm
