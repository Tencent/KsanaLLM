/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/mem_adjuster.h"

namespace ksana_llm {

MemAdjuster::MemAdjuster(const LayerCreationContext& creation_context) {
  mem_adjuster_layer_ = std::make_shared<MemAdjusterLayer>();
  mem_adjuster_layer_->Init({}, creation_context.runtime_config, creation_context.context, creation_context.rank);
}

MemAdjuster::~MemAdjuster() {}

Status MemAdjuster::ExtractSubMatrix(const Tensor& input_tensor, Tensor& output_tensor, size_t input_offset,
                                     size_t output_n) {
  return mem_adjuster_layer_->ExtractSubMatrix(input_tensor, output_tensor, input_offset, output_n);
}

}  // namespace ksana_llm
