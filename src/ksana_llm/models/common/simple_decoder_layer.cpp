/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/common/simple_decoder_layer.h"

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
SimpleDecoderLayer<T>::SimpleDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias,
                                          LayerCreationContext& creation_context,
                                          ModelCreationConfig& model_creation_config)
    : layer_idx_(layer_idx) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  input_layernorms_ = std::make_shared<Layernorm>(
      layer_prefix + ".input_layernorm.weight", model_creation_config.layernorm_config.layernorm_eps, creation_context);
  post_attention_layernorms_ =
      std::make_shared<Layernorm>(layer_prefix + ".post_attention_layernorm.weight",
                                  model_creation_config.layernorm_config.layernorm_eps, creation_context);

  adds_ = std::make_shared<Add>(creation_context);

  bool use_qk_norm = model_creation_config.attn_config.use_qk_norm;
  mha_ = std::make_shared<MultiHeadAttention<T>>(layer_idx, is_neox, add_qkv_bias, use_qk_norm, creation_context,
                                                 model_creation_config);
  mlps_ = std::make_shared<TwoLayeredFFN>(layer_idx, creation_context, model_creation_config);
  tp_comm_ = std::make_shared<TpCommunicator<T>>();
}

template <typename T>
Status SimpleDecoderLayer<T>::Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
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

  // Common mlp
  STATUS_CHECK_RETURN(
      mlps_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));
  // AllReduce Sum
  tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0, is_multi_token_forward, forwarding_context);
  // Mlp residual add
  STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer));
  return Status();
}

template class SimpleDecoderLayer<float>;
template class SimpleDecoderLayer<float16>;
template class SimpleDecoderLayer<bfloat16>;

}  // namespace ksana_llm