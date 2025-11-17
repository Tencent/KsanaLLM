/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/modules/basic/all_reduce_fused_norm_add.h"
#include "ksana_llm/layers/all_reduce_residual_add_norm_layer.h"

namespace ksana_llm {

FusedAllReduceNormAdd::FusedAllReduceNormAdd(const std::string& weight_name, float norm_eps,
                                             const LayerCreationContext& creation_context,
                                             ReduceFuseType reduce_fuse_type) {
  all_reduce_residual_add_norm_layer_ = std::make_shared<AllReduceResidualAddNormLayer>();
  layernorm_weight_ = creation_context.base_weight->GetModelWeights(weight_name);
  all_reduce_residual_add_norm_layer_->Init({norm_eps, layernorm_weight_, tp_comm_}, creation_context.runtime_config,
    creation_context.context, creation_context.rank);
  tp_comm_ = std::make_shared<TpCommunicator>();
  tp_size_ = creation_context.context->GetTensorParallelSize();
  reduce_fuse_type_ = reduce_fuse_type;
  if (reduce_fuse_type == ReduceFuseType::kPostAttn) {
    post_attention_add_norm_ = std::make_shared<FusePostAttentionAddNorm>(weight_name, norm_eps, creation_context);
  } else {
    pre_attention_add_norm_ = std::make_shared<FusePreAttentionAddNorm>(weight_name, norm_eps, creation_context);
  }
}

FusedAllReduceNormAdd::~FusedAllReduceNormAdd() {}

Status FusedAllReduceNormAdd::Forward(std::vector<Tensor>& reduce_buffer_tensors,
                                           std::vector<Tensor>& residual_buffer, std::vector<Tensor>& output_tensors,
                                           const bool is_multi_token_forward, ForwardingContext& forwarding_context,
                                           bool need_add_residual_before_attn) {
  // trtllm allreduce rmsnorm residual add is only beneficial under small token nums with world size==2
  if (reduce_buffer_tensors[0].shape[0] < kAllReduceFusionTokenNumThreshold && tp_size_ == 2) {
    STATUS_CHECK_RETURN(all_reduce_residual_add_norm_layer_->Forward(
                       {reduce_buffer_tensors[0], residual_buffer[0]}, output_tensors));
  } else {
    if (reduce_fuse_type_ == ReduceFuseType::kPostAttn) {
      tp_comm_->AllReduce(reduce_buffer_tensors, output_tensors, is_multi_token_forward, forwarding_context);
      STATUS_CHECK_RETURN(post_attention_add_norm_->Forward(output_tensors, residual_buffer));
    } else {
      tp_comm_->AllReduce(reduce_buffer_tensors, output_tensors, is_multi_token_forward, forwarding_context);
      STATUS_CHECK_RETURN(pre_attention_add_norm_->Forward(output_tensors, residual_buffer,
        need_add_residual_before_attn));
    }
  }
  return Status();
}

}  // namespace ksana_llm