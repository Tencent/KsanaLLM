/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/models/communicator/tp_communicator.h"
#include "ksana_llm/modules/basic/fuse_post_attention_add_norm.h"
#include "ksana_llm/modules/basic/fuse_pre_attention_add_norm.h"

namespace ksana_llm {

enum class ReduceFuseType { kPreAttn, kPostAttn };

/**
 * Fused the next three steps into one kernel operation for dtype: fp16, bf16
 * Step 1: tp_comm_->AllReduce(reduce_buffer_tensors, hidden_buffer_tensors_0)
 * Step 2: adds->Forward(hidden_buffer_tensors_0[0], residual_buffer[0], residual_buffer)
 * Step 3 (optional): layernorm_layer_->Forward(residual_buffer, hidden_buffer_tensors_0) (post_attention_add_norm
 *         or pre_attention_add_norm)
 */
class FusedAllReduceNormAdd {
 public:
  FusedAllReduceNormAdd(const std::string& weight_name, float norm_eps, const LayerCreationContext& creation_context,
                        ReduceFuseType reduce_fuse_type);

  ~FusedAllReduceNormAdd();

  Status Forward(std::vector<Tensor>& reduce_buffer_tensors, std::vector<Tensor>& residual_buffer,
                 std::vector<Tensor>& output_tensors, const bool is_multi_token_forward,
                 ForwardingContext& forwarding_context, bool need_add_residual, bool need_apply_norm = true);

 private:
  std::shared_ptr<BaseLayer> all_reduce_residual_add_norm_layer_;
  std::shared_ptr<TpCommunicator> tp_comm_;
  std::shared_ptr<FusePostAttentionAddNorm> post_attention_add_norm_;
  std::shared_ptr<FusePreAttentionAddNorm> pre_attention_add_norm_;
  std::shared_ptr<Add> add_;
  ReduceFuseType reduce_fuse_type_ = ReduceFuseType::kPostAttn;
  Tensor layernorm_weight_;

  // Token number under 128 is tested to have good performance gain
  const size_t kAllReduceFusionTokenNumThreshold = 128;
};

}  // namespace ksana_llm
