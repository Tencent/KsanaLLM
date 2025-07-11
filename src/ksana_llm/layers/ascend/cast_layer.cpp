/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

#include "3rdparty/LLM_kernels/csrc/utils/ascend/atb_executor.h"
#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status CastLayer<SRC_DTYPE>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = DataType::TYPE_FP32;
  atb::infer::ElewiseParam param;
  param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
  param.outTensorType = static_cast<aclDataType>(DataType(output_tensors[0].dtype));
  llm_kernels::utils::ATBOperationExecutor atb_op_executor;
  atb_op_executor.Init(rank_, param);
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor.ResetVariantPack();
  atb_op_executor.SetInputTensor(input_tensors[0].GetPtr<void>(), input_tensors[0].shape,
                                 static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
  atb_op_executor.SetOutputTensor(output_tensors[0].GetPtr<void>() + size_t(input_tensors[1].shape[0]),
                                  output_tensors[0].shape, static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
  atb_op_executor.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  return Status();
}
template class CastLayer<float>;
template class CastLayer<float16>;
template class CastLayer<bfloat16>;
}  // namespace ksana_llm
