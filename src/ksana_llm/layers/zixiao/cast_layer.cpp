/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/cast_layer.h"

namespace ksana_llm {

template <typename SRC_DTYPE>
Status CastLayer<SRC_DTYPE>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Status(RET_UNDEFINED_REFERENCE, "CastLayer not supported.");
}
template class CastLayer<float>;
template class CastLayer<float16>;
template class CastLayer<bfloat16>;

}  // namespace ksana_llm
