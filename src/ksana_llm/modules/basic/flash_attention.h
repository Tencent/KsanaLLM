/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

template <typename T>
class FlashAttention {
 public:
  // Disable a default constructor
  FlashAttention(bool is_neox, const LayerCreationContext<T>& creation_context,
                 const AttentionCreationConfig& attn_config);

  ~FlashAttention();

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                 std::vector<Tensor>& hidden_buffer_tensors_1, std::vector<Tensor>& shared_buffer_tensors,
                 const AttentionForwardContext& forward_context, Tensor query_layernorm_weight,
                 Tensor key_layernorm_weight);

 private:
  Status AddAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0, std::shared_ptr<ModelInput>& model_input,
                                 std::vector<Tensor>& hidden_buffer_tensors_1,
                                 std::vector<Tensor>& shared_buffer_tensors);

  Status RemoveAttentionPrefixCache(std::vector<Tensor>& hidden_buffer_tensors_0,
                                    std::shared_ptr<ModelInput>& model_input,
                                    std::vector<Tensor>& hidden_buffer_tensors_1,
                                    std::vector<Tensor>& shared_buffer_tensors);

 protected:
  std::shared_ptr<BaseLayer> flash_attention_layer_;
  bool reuse_prefix_caching_;
  std::shared_ptr<Context> context_;
  int rank_;
  // NOTE(karlluo): for example: machine has 4 GPUs, Attention Data Parallelism is 2, Tensor Parallelism is 2.
  // |----Attn DP Group id 0----|----Attn DP Group id 1----|
  // |     TP 0   |     TP1     |     TP0    |     TP1     |
  // |     attn   |     attn    |     attn   |     attn    |
  // |     GPU0   |     GPU1    |     GPU2   |     GPU3    |
  int attn_dp_group_id_ = 0;
  bool use_mrotary_ = false;
};
}  // namespace ksana_llm
