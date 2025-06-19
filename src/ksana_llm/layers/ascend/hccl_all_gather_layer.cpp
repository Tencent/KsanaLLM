/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/hccl_all_gather_layer.h"

namespace ksana_llm {

template <typename T>
Status HcclAllGatherLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context,
                                   int rank) {
  BaseLayer::Init(parameters, context, rank);
  context_ = context;
  rank_ = rank;

  atb::infer::AllGatherParam all_gather_param;
  all_gather_param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  all_gather_param.rank = rank;
  all_gather_param.hcclComm = context_->ext->GetHCCLComm()[rank_];
  all_gather_param.rankSize = context_->GetComputeStreams().size();
  atb_all_gather_executor_.Init(rank, all_gather_param);

  atb::infer::TransposeParam permute_param;
  permute_param.perm.push_back(1);
  permute_param.perm.push_back(0);
  permute_param.perm.push_back(2);
  atb_permute_executor_.Init(rank, permute_param);
  return Status();
}

template <typename T>
Status HcclAllGatherLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  output_tensors[0].dtype = input_tensors[0].dtype;
  size_t tp_size = context_->GetTensorParallelSize();
  if (tp_size == 1) {
    return Status();
  }
  size_t h = input_tensors[0].shape[0];
  size_t w_per = input_tensors[0].shape[1];
  output_tensors[0].shape = {h, tp_size, w_per};
  std::vector<size_t> internal_tensor_shape = {tp_size, h, w_per};
  // all gather op
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_all_gather_executor_.ResetVariantPack();
  atb_all_gather_executor_.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                          static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
  atb_all_gather_executor_.SetOutputTensor(input_tensors[1].GetPtr<void>(), internal_tensor_shape,
                                           static_cast<aclDataType>(DataType(input_tensors[1].dtype)));
  atb_all_gather_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  // permute op
  atb_permute_executor_.ResetVariantPack();
  atb_permute_executor_.SetInputTensor(input_tensors[1].GetPtr<void>(), internal_tensor_shape,
                                       static_cast<aclDataType>(DataType(input_tensors[1].dtype)));
  atb_permute_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                        static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
  atb_permute_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  output_tensors[0].shape = {h, tp_size * w_per};
  return Status();
}

template class HcclAllGatherLayer<float>;
template class HcclAllGatherLayer<float16>;
#ifdef ENABLE_BFLOAT16
template class HcclAllGatherLayer<bfloat16>;
#endif

}  // namespace ksana_llm
