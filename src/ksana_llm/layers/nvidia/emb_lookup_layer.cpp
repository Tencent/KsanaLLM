/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"
#include "csrc/kernels/nvidia/rotary_embedding/rotary_embedding.h"
#include "ksana_llm/kernels/nvidia/kernel_wrapper.h"

namespace ksana_llm {

Status EmbLookupLayer::Init(const std::vector<std::any>& parameters, const RuntimeConfig& runtime_config,
                            std::shared_ptr<Context> context, int rank) {
  BaseLayer::Init(parameters, runtime_config, context, rank);
  size_t parameter_index = 0ul;
  if (parameter_index < parameters.size()) {
    emb_scale_ = std::any_cast<const float>(parameters[parameter_index++]);
  }
  if (parameter_index < parameters.size()) {
    pos_weight_ = std::any_cast<void*>(parameters[parameter_index++]);
  }
  return Status();
}

Status EmbLookupLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  LAYER_ForwardT(inter_data_type_, input_tensors, output_tensors);
}

template <typename T>
Status EmbLookupLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: input_ids [token_num]
  //   1: ids_offsets [batch_size + 1]
  //   2: prefix_offsets [batch_size + 1]
  //   3: emb_weight [vocab_size, hidden_units]
  //   4: steps [token_num] (optional)
  // output_tensors:
  //   0: emb_output [token_num, hidden_units]
  int vocab_size = input_tensors[3].shape[0];
  int hidden_units = input_tensors[3].shape[1];
  int bs = input_tensors[1].shape[0] - 1;
  int token_num = input_tensors[0].shape[0];
  int vocab_id = 0;
  const void* steps = input_tensors.size() > 4 ? input_tensors[4].GetPtr<void>() : nullptr;

  LookupEmbedding<T>(input_tensors[0].GetPtr<void>(), input_tensors[1].GetPtr<void>(), input_tensors[2].GetPtr<void>(),
                     input_tensors[3].GetPtr<void>(), pos_weight_, steps, output_tensors[0].GetPtr<void>(), emb_scale_,
                     vocab_size, hidden_units, bs, vocab_id, context_->GetComputeStreams()[rank_].Get());
  output_tensors[0].shape = {static_cast<size_t>(token_num), static_cast<size_t>(hidden_units)};
  output_tensors[0].dtype = input_tensors[3].dtype;
  return Status();
}

}  // namespace ksana_llm
