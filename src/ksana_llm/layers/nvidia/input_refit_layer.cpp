/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/layers/input_refit_layer.h"

namespace ksana_llm {

Status InputRefitLayer::Forward(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  DISPATCH_BY_3_DTYPE(inter_data_type_, ForwardT, input_tensors, output_tensors);
}

template <typename T>
Status InputRefitLayer::ForwardT(const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) {
  size_t pos_num = input_tensors[1].shape[0];
  if (pos_num <= 0) return Status();
  void* output_ptr = output_tensors[0].GetPtr<void>();
  size_t hidden_units = output_tensors[0].shape[1];
  int64_t* cpu_input_refit_pos_pair = reinterpret_cast<int64_t*>(input_tensors[0].GetPtr<void>());
  void** cpu_input_refit_emb_fp32_ptr = reinterpret_cast<void**>(input_tensors[1].GetPtr<void>());
  auto float32_options = torch::TensorOptions().dtype(torch::kFloat32);
  auto output_options = float32_options;
  // Switch the output options based on the data type of the output tensor
  switch (output_tensors[0].dtype) {
    case TYPE_BF16:
      output_options = torch::TensorOptions().dtype(c10::kBFloat16);
      break;
    case TYPE_FP16:
      output_options = torch::TensorOptions().dtype(torch::kFloat16);
      break;
    case TYPE_FP32:
      output_options = torch::TensorOptions().dtype(torch::kFloat32);
      break;
    default:
      break;
  }
  for (size_t i = 0; i < pos_num; i++) {
    int64_t pos = cpu_input_refit_pos_pair[i * 2];
    int64_t len = cpu_input_refit_pos_pair[i * 2 + 1];
    torch::Tensor tensor_fp32 = torch::from_blob(cpu_input_refit_emb_fp32_ptr[i], {len}, float32_options);
    torch::Tensor cast_tensor = tensor_fp32.to(output_options);
    // Ensure that the data is not released before completing the data transfer to the GPU.
    cast_tensor_vec_.push_back(cast_tensor);
    // Copy the cast tensor data to the output tensor.
    if (pos * hidden_units + len <= output_tensors[0].GetElementNumber()) {
      MemcpyAsync(output_ptr + pos * hidden_units * sizeof(T), cast_tensor.data_ptr(), len * sizeof(T),
                  MEMCPY_HOST_TO_DEVICE, context_->GetComputeStreams()[rank_]);
    }
  }
  return Status();
}

}  // namespace ksana_llm
