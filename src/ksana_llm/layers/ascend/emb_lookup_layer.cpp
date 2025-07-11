/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"
#include "ksana_llm/kernels/ascend/kernel_wrapper.h"

namespace ksana_llm {

template <typename T>
Status EmbLookupLayer<T>::Init(const std::vector<std::any>& parameters, std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, context, rank);
  size_t parameter_index = 0ul;
  if (parameter_index < parameters.size()) {
    emb_scale_ = std::any_cast<const T>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    pos_weight_ = std::any_cast<void*>(parameters[parameter_index++]);
  }
  atb::infer::GatherParam gather_param;
  atb_op_executor_.Init(rank, gather_param);
  return Status();
}

template <typename T>
Status EmbLookupLayer<T>::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // weigth_shape = input_tensors[2].
  // input_tensors:
  //   0: input_ids
  //   1: offset
  //   2: prefix (not be used)
  //   3: emb_weight
  //   4: pos (optional)
  // output_tensors:
  //   0: emb_output
  int total_seq_len = input_tensors[0].shape[0];
  int hidden_units = input_tensors[3].shape[1];
  output_tensors[0].shape = {static_cast<size_t>(total_seq_len), static_cast<size_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[3].dtype;

  Tensor input_ids = input_tensors[0];
  Tensor embedding_table = input_tensors[3];

  if (input_tensors.size() > 4) {
    KLLM_THROW("Not supported position embedding on NPU.");
  } else {
    reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_))
        ->SetExecuteStream(context_->GetComputeStreams()[rank_].Get());
    atb_op_executor_.ResetVariantPack();
    atb_op_executor_.SetInputTensor(embedding_table.GetPtr<void>(), embedding_table.shape,
                                    static_cast<aclDataType>(DataType(embedding_table.dtype)));
    atb_op_executor_.SetInputTensor(input_ids.GetPtr<void>(), input_ids.shape,
                                    static_cast<aclDataType>(DataType(input_ids.dtype)));
    atb_op_executor_.SetOutputTensor(output_tensors[0].GetPtr<void>(), output_tensors[0].shape,
                                     static_cast<aclDataType>(DataType(output_tensors[0].dtype)));
    atb_op_executor_.Run(reinterpret_cast<atb::Context*>(GetRuntimeContext(rank_)), GetWorkSpaceFunc());
  }

  return Status();
}
template class EmbLookupLayer<float>;
template class EmbLookupLayer<float16>;
template class EmbLookupLayer<bfloat16>;
}  // namespace ksana_llm
