/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class PagedAttention {
 public:
  // Disable a default constructor
  PagedAttention(bool is_neox, const LayerCreationContext<T>& creation_context,
                 const AttentionCreationConfig& attn_config);

  ~PagedAttention();

  // TODO(robertyuan): param after output tensor should be removed
  Status Forward(std::vector<Tensor>& input_tensors, std::shared_ptr<ModelInput>& model_input,
                 std::vector<Tensor>& output_tensors, std::vector<Tensor>& paged_buffer_tensors,
                 Tensor& kv_cache_buffer_tensor, const AttentionForwardContext& forward_context,
                 Tensor query_layernorm_weight, Tensor key_layernorm_weight);

 protected:
  std::shared_ptr<BaseLayer> paged_attention_layer_;
  bool is_cudagraph_enabled_;
  int rank_;
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     attn   |     attn    |     attn   |     attn    |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  int attn_dp_group_id_ = 0;
};
}  // namespace ksana_llm
