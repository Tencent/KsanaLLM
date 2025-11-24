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
  reduce_fuse_type_ = reduce_fuse_type;
  if (reduce_fuse_type == ReduceFuseType::kPostAttn) {
    post_attention_add_norm_ = std::make_shared<FusePostAttentionAddNorm>(weight_name, norm_eps, creation_context);
  } else {
    pre_attention_add_norm_ = std::make_shared<FusePreAttentionAddNorm>(weight_name, norm_eps, creation_context);
  }
  add_ = std::make_shared<Add>(creation_context);
}

FusedAllReduceNormAdd::~FusedAllReduceNormAdd() {}

Status FusedAllReduceNormAdd::Forward(std::vector<Tensor>& reduce_buffer_tensors, std::vector<Tensor>& residual_buffer,
                                      std::vector<Tensor>& output_tensors, const bool is_multi_token_forward,
                                      ForwardingContext& forwarding_context, bool need_add_residual,
                                      bool need_apply_norm) {
  // tokenweave allreduce fusion requires multicast support
  // trtllm allreduce fusion is beneficial only under small token nums with world size==2

  // Case 1: AllReduce + Residual
  if (!need_apply_norm) {
    if (forwarding_context.GetContext()->ext->IsMulticastSupported()) {
      return all_reduce_residual_add_norm_layer_->Forward({reduce_buffer_tensors[0]}, residual_buffer);
    } else {
      tp_comm_->AllReduce(reduce_buffer_tensors, output_tensors, is_multi_token_forward, forwarding_context);
      return add_->Forward(output_tensors[0], residual_buffer[0], residual_buffer);
    }
  }

  // Case 2: AllReduce + Residual + Norm
  bool use_custom = forwarding_context.GetContext()->ext->IsMulticastSupported();
#ifdef ENABLE_CUDA
  if (reduce_buffer_tensors[0].shape[0] < kAllReduceFusionTokenNumThreshold &&
      forwarding_context.GetContext()->ext->GetTrtAllReduceBuffers().front() != nullptr) {
    use_custom = true;
  }
#endif
  if (use_custom) {
    STATUS_CHECK_RETURN(
        all_reduce_residual_add_norm_layer_->Forward({reduce_buffer_tensors[0], residual_buffer[0]}, output_tensors));
  } else {
    tp_comm_->AllReduce(reduce_buffer_tensors, output_tensors, is_multi_token_forward, forwarding_context);
    if (reduce_fuse_type_ == ReduceFuseType::kPostAttn) {
      STATUS_CHECK_RETURN(post_attention_add_norm_->Forward(output_tensors, residual_buffer));
    } else {
      STATUS_CHECK_RETURN(pre_attention_add_norm_->Forward(output_tensors, residual_buffer, need_add_residual));
    }
  }
  return Status();
}

}  // namespace ksana_llm
