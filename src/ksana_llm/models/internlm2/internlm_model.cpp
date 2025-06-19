/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/internlm2/internlm_model.h"

#include <memory>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {
template <typename T>
Status Internlm2<T>::GetModelRunConfig(ModelRunConfig& model_run_config, const ModelConfig& model_config) {
  model_run_config.position_encoding = PositionEncoding::ROPE;
  model_run_config.layernorm_position = LayerNormPosition::PRE_NORM;
  return Status();
}

template <typename T>
Status Internlm2<T>::CreateLayers(LayerCreationContext<T>& creation_context,
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
Status Internlm2<T>::Forward(std::vector<Tensor>& residual_buffer, ForwardingContext<T>& forwarding_context) {
  const bool is_multi_token_forward = forwarding_context.model_input_->multi_token_request_num > 0;
  for (int layer_idx = forwarding_context.pipeline_config_.lower_layer_idx;
       layer_idx <= forwarding_context.pipeline_config_.upper_layer_idx; ++layer_idx) {
    STATUS_CHECK_RETURN(
        decoder_layers_[layer_idx]->Forward(residual_buffer, is_multi_token_forward, forwarding_context));
  }
  return Status();
}

template class Internlm2<float>;
template class Internlm2<float16>;
#ifdef ENABLE_BFLOAT16
template class Internlm2<bfloat16>;
#endif

template <typename T>
Internlm2Model<T>::Internlm2Model(const ModelConfig& model_config, const int rank, std::shared_ptr<Context> context,
                                  std::shared_ptr<BaseWeight> base_weight)
    : CommonModel<T>(model_config, rank, context) {
  ModelRunConfig model_run_config;
  internlm2_.GetModelRunConfig(model_run_config, model_config);
  CommonModel<T>::InitRunConfig(model_run_config, base_weight);
}

template <typename T>
Status Internlm2Model<T>::CreateLayers(LayerCreationContext<T>& creation_context,
                                       ModelCreationConfig& model_creation_config) {
  return internlm2_.CreateLayers(creation_context, model_creation_config);
}

template <typename T>
Status Internlm2Model<T>::LayerForward(ForwardingContext<T>& forwarding_context, const RunMode run_mode) {
  std::vector<Tensor>& residual_buffer =
      GetHiddenUnitBuffer(forwarding_context, !forwarding_context.context_->IsChief());
  STATUS_CHECK_RETURN(internlm2_.Forward(residual_buffer, forwarding_context));
  SetHiddenUnitBuffer(residual_buffer, forwarding_context);
  return Status();
}

template class Internlm2Model<float>;
template class Internlm2Model<float16>;
#ifdef ENABLE_BFLOAT16
template class Internlm2Model<bfloat16>;
#endif

}  // namespace ksana_llm
