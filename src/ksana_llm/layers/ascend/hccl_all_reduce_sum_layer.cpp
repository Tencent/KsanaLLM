/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/hccl_all_reduce_sum_layer.h"

namespace ksana_llm {

template <typename T>
Status HcclAllReduceSumLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                      std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  context_ = context;
  rank_ = rank;

  atb::infer::AllReduceParam all_reduce_param;
  all_reduce_param.commMode = atb::infer::CommMode::COMM_MULTI_THREAD;
  all_reduce_param.rank = rank;
  all_reduce_param.hcclComm = context_->ext->GetHCCLComm()[rank_];
  all_reduce_param.rankSize = context_->GetComputeStreams().size();
  atb_op_executor_.Init(rank, all_reduce_param);
  return Status();
}

template <typename T>
Status HcclAllReduceSumLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                         std::vector<Tensor>& output_tensors) {
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;

  if (context_->GetTensorParallelSize() > 1) {
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
        ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
    atb_op_executor_.ResetVariantPack();
    atb_op_executor_.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                    static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
    atb_op_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                     static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
    atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  } else {
    void* src = input_tensors[0].GetPtr<void>();
    void* dst = output_tensors[0].GetPtr<void>();
    MemcpyAsync(dst, src, input_tensors[0].GetTotalBytes(), MEMCPY_DEVICE_TO_DEVICE,
                context_->GetMemoryManageStreams()[rank_]);
  }
  return Status();
}

template class HcclAllReduceSumLayer<float>;
template class HcclAllReduceSumLayer<float16>;
template class HcclAllReduceSumLayer<bfloat16>;

}  // namespace ksana_llm
