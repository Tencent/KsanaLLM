/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/modules/basic/linear.h"

namespace ksana_llm {

template <typename T>
Linear<T>::Linear(const std::string& weight_name, const LayerCreationContext<T>& creation_context,
                  const GroupQuantBackend& group_quant_backend) {
  proj_layer_ = creation_context.matmul_layer_factory->AutoCreateLayer(
      creation_context.base_weight, weight_name, creation_context.weight_type, creation_context.input_type,
      creation_context.output_type, group_quant_backend, {});
#ifdef ENABLE_ACL
  proj_layer_->Init({}, creation_context.context, creation_context.rank);
#endif

  weight_ = creation_context.base_weight->GetModelWeights(weight_name);
}

template <typename T>
Linear<T>::~Linear() {}

template <typename T>
Status Linear<T>::Forward(Tensor input_tensor, std::vector<Tensor>& output_tensors) {
  STATUS_CHECK_RETURN(proj_layer_->Forward({input_tensor, weight_}, output_tensors));
  return Status();
}

template <typename T>
Status Linear<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  return Forward(input_tensors[0], output_tensors);
}

template class Linear<float>;
template class Linear<float16>;
#ifdef ENABLE_BFLOAT16
template class Linear<bfloat16>;
#endif

}  // namespace ksana_llm
