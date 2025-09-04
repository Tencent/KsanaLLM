/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/modules/basic/add.h"
#include "ksana_llm/modules/basic/linear.h"
#include "ksana_llm/modules/basic/silu_mul.h"

#include "ksana_llm/models/base/forwarding_context.h"

namespace ksana_llm {

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
  bool fuse_silu_mul_ = false;  // whether to fuse silu mul into quant before gemm
  bool mlp_bias_ = false;
  bool enable_full_shared_expert_ = false;

  Tensor mlp_gate_bias_tensor_;
  Tensor mlp_up_bias_tensor_;

  std::shared_ptr<Add> adds_;
  std::shared_ptr<SiluMul> silu_muls_;
  std::shared_ptr<Linear> mlp_gate_up_projs_;
  std::shared_ptr<Linear> mlp_up_projs_;
  std::shared_ptr<Linear> mlp_gate_projs_;
  std::shared_ptr<Linear> mlp_down_projs_;
};  // namespace ksana_llm

}  // namespace ksana_llm
