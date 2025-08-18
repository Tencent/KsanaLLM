/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/mem_adjuster.h"

namespace ksana_llm {

MemAdjuster::MemAdjuster(const LayerCreationContext& creation_context) {
  mem_adjuster_layer_ = std::make_shared<MemAdjusterLayer>();
  mem_adjuster_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

MemAdjuster::~MemAdjuster() {}

Status MemAdjuster::GatherSubmatrix(const Tensor& input, Tensor& output_tensor, size_t dp_group_id,
                                       const std::vector<int>& dp_token_offset, size_t max_seq_len, size_t tp_size,
                                       Tensor& workspace_tensor) {
  return mem_adjuster_layer_->GatherSubmatrix(input, output_tensor, dp_group_id, dp_token_offset, max_seq_len, tp_size,
                                              workspace_tensor);
}

Status MemAdjuster::DpMapCopy(const Tensor& input, Tensor& output_tensor, const std::vector<int>& dp_token_offset,
                                 Tensor& workspace_tensor) {
  return mem_adjuster_layer_->DpMapCopy(input, output_tensor, dp_token_offset, workspace_tensor);
}

}  // namespace ksana_llm
