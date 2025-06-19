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

template <typename T>
class MultiHeadAttention {
 public:
  MultiHeadAttention(int layer_idx, bool is_neox, bool add_qkv_bias, bool use_qk_norm,
                     LayerCreationContext<T>& creation_context, ModelCreationConfig& model_creation_config);
  ~MultiHeadAttention() {}

  // Input tensors: hidden_buffer_tensors_0
  // Output tensors: hidden_buffer_tensors_0
  //                 or reduce_buffer_tensors if have forwarding_context.model_communicator_
  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext<T>& forwarding_context);

 private:
  std::shared_ptr<CommonAttention<T>> attentions_;

  std::shared_ptr<Add<T>> adds_;

  bool add_qkv_bias_;
  Tensor qkv_bais_;
  std::shared_ptr<Linear<T>> attn_qkv_projs_;
};

}  // namespace ksana_llm
