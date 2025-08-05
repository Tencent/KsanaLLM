/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/attention/multihead_attention.h"

#include <memory>
#include <vector>

#include "fmt/core.h"
#include "ksana_llm/runtime/infer_stage.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/request.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(int layer_idx, bool is_neox, bool add_qkv_bias, bool use_qk_norm,
                                          LayerCreationContext& creation_context,
                                          ModelCreationConfig& model_creation_config)
    : add_qkv_bias_(add_qkv_bias) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);

  attn_qkv_projs_ = std::make_shared<Linear>(layer_prefix + ".self_attn.query_key_value.weight", creation_context,
                                             model_creation_config.attn_config.model_config.quant_config.backend);
  if (add_qkv_bias_) {
    qkv_bais_ = creation_context.base_weight->GetModelWeights(layer_prefix + ".self_attn.query_key_value.bias");
  }

  adds_ = std::make_shared<Add>(creation_context);

  attentions_ =
      std::make_shared<CommonAttention<T>>(layer_idx, is_neox, use_qk_norm, creation_context, model_creation_config);
}

template <typename T>
Status MultiHeadAttention<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                      std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                      ForwardingContext& forwarding_context) {
  {
    CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
    // Attn proj MatMul
    STATUS_CHECK_RETURN(attn_qkv_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));

    if (add_qkv_bias_) {
      STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], qkv_bais_, hidden_buffer_tensors_1));
    }

    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }
  // Common attention
  STATUS_CHECK_RETURN(
      attentions_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors, is_multi_token_forward, forwarding_context));

  return Status();
}

template class MultiHeadAttention<float>;
template class MultiHeadAttention<float16>;
template class MultiHeadAttention<bfloat16>;

}  // namespace ksana_llm
