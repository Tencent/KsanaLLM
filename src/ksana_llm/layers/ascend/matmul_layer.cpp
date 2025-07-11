/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/matmul_layer.h"

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename T>
Status MatMulLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  context_ = context;
  rank_ = rank;

  atb::infer::LinearParam linear_param;
  linear_param.transposeA = false;
  linear_param.transposeB = false;
  linear_param.hasBias = false;
  linear_param.outDataType = ACL_DT_UNDEFINED;
  atb_op_executor_.Init(rank, linear_param);

  return Status();
}

template <typename T>
Status MatMulLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // TODO(karlluo): support bias
  int64_t m = input_tensors[0].shape[0];
  int64_t n = (GetACLFormat(input_tensors[1].data_format) == aclFormat::ACL_FORMAT_FRACTAL_NZ)
                  ? size_t(input_tensors[1].shape[1]) * size_t(input_tensors[1].shape[3])
                  : size_t(input_tensors[1].shape[1]);
  output_tensors[0].shape = {static_cast<size_t>(m), static_cast<size_t>(n)};
  output_tensors[0].dtype = input_tensors[0].dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[0].dtype)),
                                  GetACLFormat(input_tensors[0].data_format));
  atb_op_executor_.SetInputTensor(input_tensors[1].GetPtr<void>(), input_tensors[1].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[1].dtype)),
                                  GetACLFormat(input_tensors[1].data_format));
  atb_op_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                   static_cast<aclDataType>(DataType(output_tensors[0].dtype)),
                                   GetACLFormat(output_tensors[0].data_format));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  return Status();
}

template class MatMulLayer<float>;
template class MatMulLayer<float16>;
template class MatMulLayer<bfloat16>;

}  // namespace ksana_llm
