/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/ffn/two_layered_ffn.h"

namespace ksana_llm {

template <typename T>
TwoLayeredFFN<T>::TwoLayeredFFN(int layer_idx, LayerCreationContext& creation_context,
                                ModelCreationConfig& model_creation_config) {
  const std::string weight_name_format = ".mlp.{}.weight";
  InitConfig(layer_idx, creation_context, weight_name_format);
  InitLayers(layer_idx, creation_context, model_creation_config, weight_name_format);
}

template <typename T>
TwoLayeredFFN<T>::TwoLayeredFFN(int layer_idx, LayerCreationContext& creation_context,
                                ModelCreationConfig& model_creation_config, const std::string& weight_name_format) {
  InitConfig(layer_idx, creation_context, weight_name_format);
  InitLayers(layer_idx, creation_context, model_creation_config, weight_name_format);
}

template <typename T>
void TwoLayeredFFN<T>::InitConfig(int layer_idx, LayerCreationContext& creation_context,
                                  const std::string& weight_name_format) {
  std::string up_gate_proj_weights_name =
      fmt::format("model.layers.{}" + weight_name_format, layer_idx, "gate_up_proj");
  if (creation_context.base_weight->GetModelWeights(up_gate_proj_weights_name).GetElementNumber() > 0) {
    fuse_gate_up_proj_ = true;
  } else {
    fuse_gate_up_proj_ = false;
  }
  std::string gate_proj_bias_weights_name = fmt::format("model.layers.{}.mlp.{}", layer_idx, "gate_proj_bias");
  std::string up_proj_bias_weights_name = fmt::format("model.layers.{}.mlp.{}", layer_idx, "up_proj_bias");
  if (creation_context.base_weight->GetModelWeights(gate_proj_bias_weights_name).GetElementNumber() > 0 &&
      creation_context.base_weight->GetModelWeights(up_proj_bias_weights_name).GetElementNumber() > 0) {
    mlp_bias_ = true;
  } else {
    mlp_bias_ = false;
  }
}

template <typename T>
void TwoLayeredFFN<T>::InitLayers(int layer_idx, LayerCreationContext& creation_context,
                                  ModelCreationConfig& model_creation_config, const std::string& weight_name_format) {
  std::string layer_prefix = fmt::format("model.layers.{}", layer_idx);
  GroupQuantBackend linear_group_quant_backend = model_creation_config.attn_config.model_config.quant_config.backend;
  if (fuse_gate_up_proj_) {
    mlp_gate_up_projs_ = std::make_shared<Linear<T>>(fmt::format(layer_prefix + weight_name_format, "gate_up_proj"),
                                                     creation_context, linear_group_quant_backend);
  } else {
    mlp_gate_projs_ = std::make_shared<Linear<T>>(fmt::format(layer_prefix + weight_name_format, "gate_proj"),
                                                  creation_context, linear_group_quant_backend);
    mlp_up_projs_ = std::make_shared<Linear<T>>(fmt::format(layer_prefix + weight_name_format, "up_proj"),
                                                creation_context, linear_group_quant_backend);
  }
  if (mlp_bias_) {
    mlp_gate_bias_tensor_ = creation_context.base_weight->GetModelWeights(
        fmt::format("model.layers.{}.mlp.{}", layer_idx, "gate_proj_bias"));
    mlp_up_bias_tensor_ =
        creation_context.base_weight->GetModelWeights(fmt::format("model.layers.{}.mlp.{}", layer_idx, "up_proj_bias"));
  }
  mlp_down_projs_ = std::make_shared<Linear<T>>(fmt::format(layer_prefix + weight_name_format, "down_proj"),
                                                creation_context, linear_group_quant_backend);
  adds_ = std::make_shared<Add<T>>(creation_context);
  silu_muls_ = std::make_shared<SiluMul<T>>(creation_context);
}

template <typename T>
Status TwoLayeredFFN<T>::Forward(std::vector<Tensor>& hidden_buffer_tensors_0,
                                 std::vector<Tensor>& reduce_buffer_tensors, const bool is_multi_token_forward,
                                 ForwardingContext& forwarding_context) {
  CREATE_BUFFER_SCOPE(hidden_buffer_tensors_1, forwarding_context.GetForwardingBuffers()->hidden_buffer_1);
  if (fuse_gate_up_proj_) {
    // Mlp gate_up_proj MatMul
    STATUS_CHECK_RETURN(mlp_gate_up_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
    STATUS_CHECK_RETURN(silu_muls_->Forward(hidden_buffer_tensors_0[0], hidden_buffer_tensors_1));
    std::swap(hidden_buffer_tensors_0, hidden_buffer_tensors_1);
  } else {
    auto& gated_buffer_ = reduce_buffer_tensors;
    // Mlp gate_proj MatMul
    STATUS_CHECK_RETURN(mlp_gate_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    if (mlp_bias_) {
      // Mlp gate_proj Bias Add
      STATUS_CHECK_RETURN(adds_->Forward(hidden_buffer_tensors_1[0], mlp_gate_bias_tensor_, hidden_buffer_tensors_1));
    }
    // Mlp up_proj MatMul 由于 gate_proj 与 up_proj 为并行关系,因此此处使用额外空间存储 matmul 结果
    STATUS_CHECK_RETURN(mlp_up_projs_->Forward(hidden_buffer_tensors_0, gated_buffer_));
    if (mlp_bias_) {
      // Mlp up_proj Bias Add
      STATUS_CHECK_RETURN(adds_->Forward(gated_buffer_[0], mlp_up_bias_tensor_, gated_buffer_));
    }
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);

    // Activation is an in-place operation, just put the output in `hidden_buffer_tensors_0`, the
    // same as the input.
    STATUS_CHECK_RETURN(silu_muls_->Forward(hidden_buffer_tensors_0[0], gated_buffer_[0], hidden_buffer_tensors_0));
  }

  // Mlp down_proj MatMul
  if (forwarding_context.GetModelCommunicator()) {
    // Put output to `reduce_buffer_tensors` to ensure that the input for custom reduce sum is
    // always in `reduce_buffer_tensors`.
    STATUS_CHECK_RETURN(mlp_down_projs_->Forward(hidden_buffer_tensors_0, reduce_buffer_tensors));
  } else {
    STATUS_CHECK_RETURN(mlp_down_projs_->Forward(hidden_buffer_tensors_0, hidden_buffer_tensors_1));
    std::swap(hidden_buffer_tensors_1, hidden_buffer_tensors_0);
  }

  return Status();
}

template class TwoLayeredFFN<float>;
template class TwoLayeredFFN<float16>;
template class TwoLayeredFFN<bfloat16>;

}  // namespace ksana_llm
