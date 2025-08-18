/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/layers/mem_adjuster_layer.h"
#include "ksana_llm/models/base/layer_creation_context.h"

namespace ksana_llm {

class MemAdjuster {
 public:
  explicit MemAdjuster(const LayerCreationContext& creation_context);

  ~MemAdjuster();

  Status GatherSubmatrix(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                         const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                         Tensor& workspace_tensor);

  Status DpMapCopy(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                   Tensor& workspace_tensor);

 protected:
  std::shared_ptr<MemAdjusterLayer> mem_adjuster_layer_;
};
}  // namespace ksana_llm
