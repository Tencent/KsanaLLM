/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/forward_request.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

/*
 * This decoder layer is defined according to Llama with config support on
 * is_neox, add_qkv_bias
 * layernorm_position = LayerNormPosition::PRE_NORM
 * */
template <typename T>
class SimpleDecoderLayer {
 public:
  SimpleDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias, LayerCreationContext<T>& creation_context,
                     ModelCreationConfig& model_creation_config);
  ~SimpleDecoderLayer() {}
  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext<T>& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<MultiHeadAttention<T>> mha_;
  std::shared_ptr<TwoLayeredFFN<T>> mlps_;
  std::shared_ptr<TpCommunicator<T>> tp_comm_;

  std::shared_ptr<Add<T>> adds_;

  std::shared_ptr<Layernorm<T>> input_layernorms_;
  std::shared_ptr<Layernorm<T>> post_attention_layernorms_;
};

}  // namespace ksana_llm
