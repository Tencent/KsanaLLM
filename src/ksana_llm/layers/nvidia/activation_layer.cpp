/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/activation_layer.h"

#include "csrc/kernels/nvidia/activation/activation.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

bool IsGatedActivation(ActivationType type) {
  if (type == ActivationType::Geglu || type == ActivationType::Swiglu) {
    return true;
  }
  return false;
}

Status ActivationLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                             std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  activation_type_ = std::any_cast<const ActivationType>(parameters[0]);
  return Status();
}

Status ActivationLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  LAYER_ForwardT(inter_data_type_, input_tensors, output_tensors);
}

template <typename T>
Status ActivationLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: input [token_num, hidden_size]
  //   1: gated_weight [hidden_size, inter_size] (optional)
  //   2: bias [hidden_size] (optional)
  //   3: gated_bias [inter_size] (optional)
  // output_tensors:
  //   0: output [token_num, inter_size] act(input + bias) * (gated_weight + gated_bias)
  // Note: when bias is provided, gated_bias must be provided.
  const void* gated_weight = (input_tensors.size() > 1 && IsGatedActivation(activation_type_))
                                 ? reinterpret_cast<const void*>(input_tensors[1].GetPtr<void>())
                                 : nullptr;
  const void* bias =
      input_tensors.size() > 2 ? reinterpret_cast<const void*>(input_tensors[2].GetPtr<void>()) : nullptr;
  const void* gated_bias = (input_tensors.size() > 3 && IsGatedActivation(activation_type_))
                               ? reinterpret_cast<const void*>(input_tensors[3].GetPtr<void>())
                               : nullptr;

  if (activation_type_ == ActivationType::Gelu || activation_type_ == ActivationType::Geglu) {
    InvokeGatedActivation<llm_kernels::nvidia::GeluActivation, T>(
        reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), bias, gated_weight, gated_bias,
        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  } else if (activation_type_ == ActivationType::Relu) {
    InvokeGatedActivation<llm_kernels::nvidia::ReluActivation, T>(
        reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), bias, gated_weight, gated_bias,
        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  } else {  // activation_type_ == ActivationType::Swiglu
    InvokeGatedActivation<llm_kernels::nvidia::SiluActivation, T>(
        reinterpret_cast<const void*>(input_tensors[0].GetPtr<void>()), bias, gated_weight, gated_bias,
        static_cast<int>(input_tensors[0].shape[0]), static_cast<int>(input_tensors[0].shape[1]),
        output_tensors[0].GetPtr<void>(), context_->GetComputeStreams()[rank_].Get());
  }
  output_tensors[0].shape = input_tensors[0].shape;
  output_tensors[0].dtype = input_tensors[0].dtype;
  return Status();
}

Status SigmoidLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  LAYER_ForwardT(inter_data_type_, input_tensors, output_tensors);
}

template <typename T>
Status SigmoidLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  if (input_tensors[0].GetPtr<void>() == output_tensors[0].GetPtr<void>()) {
    size_t size = output_tensors[0].shape[0] * output_tensors[0].shape[1];
    float scale = 1.0f;
    InvokeSigmoidActivation<T>(output_tensors[0].GetPtr<void>(), size, scale,
                               context_->GetComputeStreams()[rank_].Get());
  } else {
    KLLM_LOG_WARNING << "The sigmoid layer can directly process the tensor without needing to return another tensor.";
  }
  return Status();
}

}  // namespace ksana_llm
