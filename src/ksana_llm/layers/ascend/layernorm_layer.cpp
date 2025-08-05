/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/layernorm_layer.h"

#include <cstdint>

#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

Status LayernormLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  int parameter_index = 0;
  context_ = context;
  rank_ = rank;
  rms_norm_eps_ = std::any_cast<const float>(parameters[parameter_index++]);

  atb::infer::RmsNormParam rms_norm_param;
  rms_norm_param.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
  rms_norm_param.normParam.epsilon = rms_norm_eps_;
  rms_norm_param.normParam.layerNormEps = rms_norm_eps_;
  atb_op_executor_.Init(rank, rms_norm_param);

  return Status();
}

Status LayernormLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  void* lm_input_tensor_buf_ptr = input_tensors[0].GetPtr<void>();
  void* lm_weight_tensor_buf_ptr = input_tensors[1].GetPtr<void>();
  void* lm_output_tensor_buf_ptr = output_tensors[0].GetPtr<void>();
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(lm_input_tensor_buf_ptr, input_tensors[0].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
  atb_op_executor_.SetInputTensor(lm_weight_tensor_buf_ptr, input_tensors[1].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[1].dtype)));
  atb_op_executor_.SetOutputTensor(lm_output_tensor_buf_ptr, output_tensors[0].shape,
                                   static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
