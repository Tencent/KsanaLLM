/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/silu_mul_layer.h"

#include "csrc/utils/ascend/common.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"
#include "ksana_llm/utils/ascend/acl_utils.h"

namespace ksana_llm {

Status SiluMulLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                          std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);

  // NOTE(karlluo): input and weight
  constexpr int32_t INPUT_TENSOR_NUM = 2;
  constexpr int32_t OUTPUT_TENSOR_NUM = 1;
  // NOTE(karlluo): 1 buffer for silu output
  constexpr int32_t INTERNAL_TENSOR_NUM = 1;
  constexpr int32_t NODE_NUM = 2;

  uint32_t node_idx = 0;
  atb::GraphParam op_graph;
  op_graph.name = "FusedSiluMul";
  op_graph.inTensorNum = INPUT_TENSOR_NUM;
  op_graph.outTensorNum = OUTPUT_TENSOR_NUM;
  op_graph.internalTensorNum = INTERNAL_TENSOR_NUM;
  op_graph.nodes.resize(NODE_NUM);
  uint32_t tensor_idx = 0;
  uint32_t input_tensor_idx = tensor_idx++;
  uint32_t weight_tensor_idx = tensor_idx++;
  uint32_t output_tensor_idx = tensor_idx++;
  uint32_t interal_tensor_idx = tensor_idx++;
  // for silu
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::ActivationParam activation_param;
    activation_param.activationType = atb::infer::ACTIVATION_SWISH;
    atb::CreateOperation(activation_param, &op_node.operation);
    op_node.inTensorIds = {input_tensor_idx};
    op_node.outTensorIds = {interal_tensor_idx};
  }
  // for mul
  {
    atb::Node& op_node = op_graph.nodes.at(node_idx++);
    atb::infer::ElewiseParam elewise_param;
    elewise_param.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(elewise_param, &op_node.operation);
    op_node.inTensorIds = {interal_tensor_idx, weight_tensor_idx};
    op_node.outTensorIds = {output_tensor_idx};
  }
  atb_op_executor_.Init(rank, op_graph);
  atb_op_executor_.ResetVariantPack();

  return Status();
}

Status SiluMulLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors.size() != 2) {
    KLLM_THROW(fmt::format("SiluMulLayer only support 2 input tensors, but got {}", input_tensors.size()));
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  void* silu_input_buf_ptr = input_tensors[0].GetPtr<void>();
  void* silu_output_buf_ptr = output_tensors[0].GetPtr<void>();
  void* gated_weight_buf_ptr = input_tensors[1].GetPtr<void>();
  reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
      ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
  atb_op_executor_.ResetVariantPack();
  atb_op_executor_.SetInputTensor(silu_input_buf_ptr, input_tensors[0].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[0].dtype)));
  atb_op_executor_.SetInputTensor(gated_weight_buf_ptr, input_tensors[1].shape,
                                  static_cast<aclDataType>(DataType(input_tensors[1].dtype)));
  atb_op_executor_.SetOutputTensor(silu_output_buf_ptr, output_tensors[0].shape,
                                   static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
  atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  return Status();
}

}  // namespace ksana_llm
