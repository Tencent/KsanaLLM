/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/emb_lookup_layer.h"

namespace ksana_llm {

Status CpuEmbLookupLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  LAYER_ForwardT(inter_data_type_, input_tensors, output_tensors);
}

template <typename T>
Status CpuEmbLookupLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  // input_tensors:
  //   0: input_ids [token_num]
  //   1: cpu_buffer [token_num, hidden_units]
  //   2: emb_weight [vocab_size, hidden_units]
  // output_tensors:
  //   0: emb_output [token_num, hidden_units]
  size_t token_num = input_tensors[0].shape[0];
  size_t hidden_units = input_tensors[2].shape[1];

  int* cpu_input_tokens = reinterpret_cast<int*>(input_tensors[0].GetPtr<void>());
  T* cpu_data_ptr = reinterpret_cast<T*>(input_tensors[1].GetPtr<void>());
  T* emb_weight_ptr = reinterpret_cast<T*>(input_tensors[2].GetPtr<void>());
  for (size_t i = 0; i < token_num; i++) {
    memcpy(cpu_data_ptr + i * hidden_units, emb_weight_ptr + cpu_input_tokens[i] * hidden_units,
           hidden_units * sizeof(T));
  }
  MemcpyAsync(output_tensors[0].GetPtr<void>(), cpu_data_ptr, token_num * hidden_units * sizeof(T),
              MEMCPY_HOST_TO_DEVICE, context_->GetComputeStreams()[rank_]);

  output_tensors[0].dtype = input_tensors[2].dtype;
  output_tensors[0].shape = {token_num, hidden_units};
  return Status();
}

}  // namespace ksana_llm
