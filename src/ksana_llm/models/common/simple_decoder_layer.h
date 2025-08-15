/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/runtime/forward_request.h"

#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/attention/multihead_attention.h"
#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/ffn/two_layered_ffn.h"
#include "ksana_llm/modules/basic/add_norm.h"
#include "ksana_llm/modules/basic/layernorm.h"
#include "ksana_llm/modules/basic/fuse_pre_attention_add_norm.h"
#include "ksana_llm/modules/basic/fuse_post_attention_add_norm.h"

namespace ksana_llm {

/*
 * This decoder layer is defined according to Llama with config support on
 * is_neox, add_qkv_bias
 * layernorm_position = LayerNormPosition::PRE_NORM
 * */

class SimpleDecoderLayer {
 public:
  SimpleDecoderLayer(int layer_idx, bool is_neox, bool add_qkv_bias, LayerCreationContext& creation_context,
                     ModelCreationConfig& model_creation_config);
  ~SimpleDecoderLayer() {}

  Status Forward(std::vector<Tensor>& residual_buffer, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<MultiHeadAttention> mha_;
  std::shared_ptr<TwoLayeredFFN> mlps_;
  std::shared_ptr<TpCommunicator> tp_comm_;

  std::shared_ptr<Add> adds_;
  std::shared_ptr<FusePreAttentionAddNorm> pre_attention_add_norm_;
  std::shared_ptr<FusePostAttentionAddNorm> post_attention_add_norm_;

  bool need_add_residual_before_attn_ = false;
  bool need_add_residual_after_mlp_ = true;
  bool use_fused_add_layernorm_ = false;
};

}  // namespace ksana_llm
