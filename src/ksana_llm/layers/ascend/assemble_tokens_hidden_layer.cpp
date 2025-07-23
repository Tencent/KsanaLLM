/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/assemble_tokens_hidden_layer.h"
#include <cstdlib>

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

template <typename T>
Status AssembleTokensHiddenLayer<T>::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  atb::infer::GatherParam gather_param;
  atb_op_executor_.Init(rank, gather_param);
  return Status();
}

template <typename T>
Status AssembleTokensHiddenLayer<T>::Forward(const std::vector<Tensor>& input_tensors,
                                             std::vector<Tensor>& output_tensors) {
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].shape[0] = input_tensors[1].shape[0];
  output_tensors[0].dtype = input_tensors[0].dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
  atb_op_executor_.SetInputTensor(input_tensors[1].GetPtr<void>(), input_tensors[1].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[1].dtype)));
  atb_op_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                   static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  return Status();
}
template class AssembleTokensHiddenLayer<float>;
template class AssembleTokensHiddenLayer<float16>;
template class AssembleTokensHiddenLayer<bfloat16>;

}  // namespace ksana_llm
