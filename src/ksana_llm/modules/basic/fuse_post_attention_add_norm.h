/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/models/base/layer_creation_context.h"
#include "ksana_llm/modules/basic/layernorm.h"
#include "ksana_llm/modules/basic/add_norm.h"
#include "ksana_llm/modules/basic/add.h"

namespace ksana_llm {

class FusePostAttentionAddNorm {
 public:
  // Disable a default constructor
  FusePostAttentionAddNorm(const std::string& weight_name, float norm_eps,
                           const LayerCreationContext& creation_context);

  ~FusePostAttentionAddNorm();

  Status Forward(std::vector<Tensor>& hidden_buffer_tensors_0, std::vector<Tensor>& residual_buffer);

 protected:
  std::shared_ptr<Layernorm> layernorm_layer_;
  std::shared_ptr<Add> adds_;
  std::shared_ptr<AddNorm> add_norm_layer_;
  bool is_fusion_supported_dtype_ = false;
};
}  // namespace ksana_llm
