/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/flash_attention.h"
#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/modules/basic/paged_attention.h"
#include "ksana_llm/modules/basic/silu_mul.h"

#include "ksana_llm/models/base/forwarding_context.h"
#include "ksana_llm/models/common/common_model.h"

namespace ksana_llm {

class CommonAttention {
 public:
  CommonAttention(int layer_idx, bool is_neox, bool use_qk_norm, LayerCreationContext& creation_context,
                  ModelCreationConfig& model_creation_config);

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  int layer_idx_;
  std::shared_ptr<FlashAttention> flash_attentions_;
  std::shared_ptr<PagedAttention> paged_attentions_;

  Tensor query_layernorm_weight_;
  Tensor key_layernorm_weight_;

  std::shared_ptr<Linear> attn_o_projs_;
#ifdef ENABLE_VLLM_FLASH_ATTN_2
  std::shared_ptr<BaseLayer> set_torch_stream_layer_;
#endif
};  // namespace ksana_llm

}  // namespace ksana_llm
