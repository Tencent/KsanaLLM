/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/add_layer.h"

#include <algorithm>

#include "csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

Status AddLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                      std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  atb::infer::ElewiseParam elewise_param;
  elewise_param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
  atb_op_executor_.Init(rank, elewise_param);
  return Status();
}

Status AddLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  const auto& a = input_tensors[0];
  const auto& b = input_tensors[1];
  auto& output = output_tensors[0];
  output.shape = a.shape;
  output.dtype = a.dtype;

  void* const out_ptr = output.GetPtr<void>();
  void* const a_ptr = a.GetPtr<void>();
  void* const b_ptr = b.GetPtr<void>();
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(a_ptr, a.shape, static_cast<aclDataType>(DataType(a.dtype)));

  // only bias and decode scenarios are involved, a more accurate broadcast check is required in the future.
  if (a.shape.size() == b.shape.size()) {
    atb_op_executor_.SetInputTensor(b_ptr, b.shape, static_cast<aclDataType>(DataType(b.dtype)));
  } else {
    const std::vector<size_t> reshape({b.shape[0], b.GetElementNumber() / b.shape[0]});
    atb_op_executor_.SetInputTensor(b_ptr, reshape, static_cast<aclDataType>(DataType(b.dtype)));
  }

  atb_op_executor_.SetOutputTensor(out_ptr, output.shape, static_cast<aclDataType>(DataType(output.dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());

  return Status();
}
}  // namespace ksana_llm
