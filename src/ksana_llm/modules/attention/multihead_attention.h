/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/models/common/common_model.h"
#include "ksana_llm/models/common/common_weight.h"
#include "ksana_llm/runtime/forward_request.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/utils.h"

#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/linear.h"

#include "ksana_llm/modules/attention/common_attention.h"

namespace ksana_llm {

class MultiHeadAttention {
 public:
  MultiHeadAttention(int layer_idx, bool is_neox, bool add_qkv_bias, bool use_qk_norm,
                     LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  ~MultiHeadAttention() {}

  // Input tensors: hidden_buffer_tensors_0
  // Output tensors: hidden_buffer_tensors_0
  //                 or reduce_buffer_tensors if have forwarding_context.GetModelCommunicator()
  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  std::shared_ptr<CommonAttention> attentions_;

  std::shared_ptr<Add> adds_;

  bool add_qkv_bias_;
  Tensor qkv_bais_;
  std::shared_ptr<Linear> attn_qkv_projs_;
};

}  // namespace ksana_llm
