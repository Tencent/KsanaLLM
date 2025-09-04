/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/common_model_weight_loader.h"
#include "ksana_llm/kernels/permute.h"

namespace ksana_llm {

CommonModelWeightLoader::CommonModelWeightLoader(std::shared_ptr<BaseModelConfig> model_config,
                                                 std::shared_ptr<Environment> env, std::shared_ptr<Context> context)
    : model_config_(model_config), env_(env), context_(context) {
  permute_buffers_.resize(context_->GetTensorParallelSize());
  tensor_buffers_.resize(context_->GetTensorParallelSize());
}
Status CommonModelWeightLoader::LoadMhaWeights(const std::string& host_weight_name, const Tensor& host_weight_tensor,
                                               std::unordered_map<std::string, Tensor>& device_model_weights,
                                               int dev_rank, size_t num_heads, size_t num_kv_heads,
                                               size_t size_per_head) {
  Tensor dev_tensor;
  SplitOptTrans(host_weight_tensor, dev_tensor, dev_rank, context_->GetTensorParallelSize(), false);
  const std::string query_key_value_name =
      host_weight_name.substr(0, host_weight_name.find(".self_attn.")) + ".self_attn.query_key_value.weight";
  if (device_model_weights.find(query_key_value_name) == device_model_weights.end()) {
    size_t query_key_value_shape_0 = num_heads * size_per_head + num_kv_heads * size_per_head * 2;
    query_key_value_shape_0 = DivRoundUp(query_key_value_shape_0, context_->GetTensorParallelSize());
    Tensor query_key_value = Tensor(MemoryLocation::LOCATION_DEVICE, dev_tensor.dtype,
                                    {query_key_value_shape_0, dev_tensor.shape[1]}, dev_rank);
    device_model_weights[query_key_value_name] = query_key_value;
  }

  Tensor& query_key_value_tensor = device_model_weights.at(query_key_value_name);
  if (host_weight_name.find(".q_proj.") != std::string::npos) {
    MemcpyAsync(query_key_value_tensor.GetPtr<void>(), dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  } else if (host_weight_name.find(".k_proj.") != std::string::npos) {
    size_t offset = num_heads * size_per_head * dev_tensor.shape[1];
    offset = DivRoundUp(offset, context_->GetTensorParallelSize());
    offset *= GetTypeSize(dev_tensor.dtype);
    MemcpyAsync(query_key_value_tensor.GetPtr<void>() + offset, dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  } else if (host_weight_name.find(".v_proj.") != std::string::npos) {
    size_t offset = (num_heads * size_per_head + num_kv_heads * size_per_head) * dev_tensor.shape[1];
    offset = DivRoundUp(offset, context_->GetTensorParallelSize());
    offset *= GetTypeSize(dev_tensor.dtype);
    MemcpyAsync(query_key_value_tensor.GetPtr<void>() + offset, dev_tensor.GetPtr<void>(), dev_tensor.GetTotalBytes(),
                MEMCPY_DEVICE_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  }

  return Status();
}

Tensor& CommonModelWeightLoader::GetTempTensor(const std::vector<size_t>& shape, DataType data_type, int dev_rank) {
  const size_t key = std::accumulate(shape.begin(), shape.end(), GetTypeSize(data_type), std::multiplies<size_t>());
  if (tensor_buffers_[dev_rank].find(key) == tensor_buffers_[dev_rank].end()) {
    tensor_buffers_[dev_rank][key] = Tensor(MemoryLocation::LOCATION_DEVICE, data_type, shape, dev_rank);
  }
  Tensor& temp_tensor = tensor_buffers_[dev_rank].at(key);
  temp_tensor.dtype = data_type;
  temp_tensor.shape = shape;
  return temp_tensor;
}

Status CommonModelWeightLoader::PermuteWeight(Tensor& input_tensor, const std::vector<size_t>& permutation,
                                              int dev_rank) {
  if (input_tensor.shape.size() != permutation.size()) {
    return Status(RetCode::RET_INVALID_ARGUMENT, "Permutation size must be equal to input tensor rank.");
  }
  const size_t key = input_tensor.GetTotalBytes();
  if (permute_buffers_[dev_rank].find(key) == permute_buffers_[dev_rank].end()) {
    permute_buffers_[dev_rank][key] =
        Tensor(MemoryLocation::LOCATION_DEVICE, input_tensor.dtype, input_tensor.shape, dev_rank);
  }
  Tensor& permute_tensor = permute_buffers_[dev_rank].at(key);
  permute_tensor.dtype = input_tensor.dtype;
  permute_tensor.shape = input_tensor.shape;
  Permute(input_tensor, permute_tensor, permutation, context_->GetMemoryManageStreams()[dev_rank]);
  for (size_t i = 0; i < permutation.size(); i++) {
    permute_tensor.shape[i] = input_tensor.shape[permutation[i]];
  }
  std::swap(input_tensor, permute_tensor);
  return Status();
}

Status CommonModelWeightLoader::SplitOptTrans(const Tensor& weight_tensor, Tensor& output_tensor, int dev_rank,
                                              size_t para_size, bool transpose) {
  std::vector<size_t> slice_shape = {static_cast<size_t>(DivRoundUp(weight_tensor.shape[0], para_size)),
                                     weight_tensor.shape[1]};
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, weight_tensor.dtype, slice_shape, dev_rank);

  size_t slice_offset = dev_tensor.GetTotalBytes() * (dev_rank % para_size);
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  if (static_cast<size_t>(dev_rank) == para_size - 1) {
    slice_bytes = weight_tensor.GetTotalBytes() - slice_offset;
  }

  MemcpyKind memcpy_kind =
      weight_tensor.location == MemoryLocation::LOCATION_HOST ? MEMCPY_HOST_TO_DEVICE : MEMCPY_DEVICE_TO_DEVICE;
  MemcpyAsync(dev_tensor.GetPtr<void>(), weight_tensor.GetPtr<void>() + slice_offset, slice_bytes, memcpy_kind,
              context_->GetMemoryManageStreams()[dev_rank]);
  if (transpose) {
    PermuteWeight(dev_tensor, {1, 0}, dev_rank);
  }
  output_tensor = dev_tensor;

  return Status();
}

Status CommonModelWeightLoader::TransSplitOptTrans(const Tensor& weight_tensor, Tensor& output_tensor, int dev_rank,
                                                   size_t para_size, bool transpose) {
  Tensor& full_dev_tensor = GetTempTensor(weight_tensor.shape, weight_tensor.dtype, dev_rank);

  MemcpyAsync(full_dev_tensor.GetPtr<void>(), weight_tensor.GetPtr<void>(), weight_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  PermuteWeight(full_dev_tensor, {1, 0}, dev_rank);

  SplitOptTrans(full_dev_tensor, output_tensor, dev_rank, para_size, transpose);

  return Status();
}

}  // namespace ksana_llm