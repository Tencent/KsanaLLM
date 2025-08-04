/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/modules/basic/silu_mul.h"

#include "ksana_llm/models/base/forwarding_context.h"

namespace ksana_llm {

template <typename T>
class TwoLayeredFFN {
 public:
  TwoLayeredFFN(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config);
  TwoLayeredFFN(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                const std::string& weight_name_format);
  ~TwoLayeredFFN() {}

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& reduce_buffer_tensors,
                 const bool is_multi_token_forward, ForwardingContext& forwarding_context);

 private:
  void InitLayers(int layer_idx, LayerCreationContext& creation_context, ModelCreationConfig& model_creation_config,
                  const std::string& weight_name_format);

  void InitConfig(int layer_idx, LayerCreationContext& creation_context, const std::string& weight_name_format);

 private:
  bool fuse_gate_up_proj_ = false;
  bool mlp_bias_ = false;
  Tensor mlp_gate_bias_tensor_;
  Tensor mlp_up_bias_tensor_;
  std::shared_ptr<Add<T>> adds_;
  std::shared_ptr<SiluMul<T>> silu_muls_;
  std::shared_ptr<Linear<T>> mlp_gate_up_projs_;
  std::shared_ptr<Linear<T>> mlp_up_projs_;
  std::shared_ptr<Linear<T>> mlp_gate_projs_;
  std::shared_ptr<Linear<T>> mlp_down_projs_;
};  // namespace ksana_llm

}  // namespace ksana_llm
