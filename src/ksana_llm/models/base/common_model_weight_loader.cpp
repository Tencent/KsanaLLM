/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/common_model_weight_loader.h"
#include "ksana_llm/kernels/permute.h"

#ifdef ENABLE_CUDA
#  include "ksana_llm/kernels/nvidia/basic_kernel_wrapper.h"
#endif

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

Status CommonModelWeightLoader::Permute2D(Tensor& input_tensor, int dev_rank) {
  if (input_tensor.shape.size() == 2) {
    return PermuteWeight(input_tensor, {1, 0}, dev_rank);
  } else {
    KLLM_THROW("PermuteWeight without explicit permutation only supports 2D tensors");
  }
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

MemcpyKind CommonModelWeightLoader::GetMemcpyKind(const Tensor& from_tensor, const Tensor& to_tensor) {
  // 根据 from_tensor 和 to_tensor 的 location 自动判断 memcpy_kind
  if (from_tensor.location == MemoryLocation::LOCATION_HOST && to_tensor.location == MemoryLocation::LOCATION_HOST) {
    return MEMCPY_HOST_TO_HOST;
  } else if (from_tensor.location == MemoryLocation::LOCATION_HOST &&
             to_tensor.location == MemoryLocation::LOCATION_DEVICE) {
    return MEMCPY_HOST_TO_DEVICE;
  } else if (from_tensor.location == MemoryLocation::LOCATION_DEVICE &&
             to_tensor.location == MemoryLocation::LOCATION_HOST) {
    return MEMCPY_DEVICE_TO_HOST;
  } else if (from_tensor.location == MemoryLocation::LOCATION_DEVICE &&
             to_tensor.location == MemoryLocation::LOCATION_DEVICE) {
    return MEMCPY_DEVICE_TO_DEVICE;
  } else {
    KLLM_THROW(fmt::format("Unsupported memory copy kind: from_location={}, to_location={}", from_tensor.location,
                           to_tensor.location));
  }
}

Tensor CommonModelWeightLoader::TensorParallelSplit1D(const Tensor& input_tensor,
                                                      const std::vector<size_t>& slice_shape, int dev_rank,
                                                      size_t para_size) {
  Tensor dev_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, input_tensor.dtype, slice_shape, dev_rank);
  size_t slice_offset = dev_tensor.GetTotalBytes() * (dev_rank % para_size);
  size_t slice_bytes = dev_tensor.GetTotalBytes();
  MemcpyKind memcpy_kind = GetMemcpyKind(input_tensor, dev_tensor);
  MemcpyAsync(dev_tensor.GetPtr<void>(), input_tensor.GetPtr<void>() + slice_offset, slice_bytes, memcpy_kind,
              context_->GetMemoryManageStreams()[dev_rank]);
  return dev_tensor;
}

Tensor CommonModelWeightLoader::TensorParallelSplit2D(const Tensor& input_tensor, int dev_rank, size_t para_size,
                                                      TensorParallelMode tp_mode) {
  if (tp_mode == TensorParallelMode::RowPara) {
    KLLM_CHECK_WITH_INFO(input_tensor.shape[0] % para_size == 0, "Tensor shape[0] must be divisible by para_size");
    std::vector<size_t> slice_shape = {input_tensor.shape[0] / para_size, input_tensor.shape[1]};
    return TensorParallelSplit1D(input_tensor, slice_shape, dev_rank, para_size);
  } else if (tp_mode == TensorParallelMode::ColPara) {
    KLLM_CHECK_WITH_INFO(input_tensor.shape[1] % para_size == 0, "Tensor shape[1] must be divisible by para_size");

    Tensor& permuted_input = GetTempTensor(input_tensor.shape, input_tensor.dtype, dev_rank);
    MemcpyKind memcpy_kind = GetMemcpyKind(input_tensor, permuted_input);
    MemcpyAsync(permuted_input.GetPtr<void>(), input_tensor.GetPtr<void>(), input_tensor.GetTotalBytes(), memcpy_kind,
                context_->GetMemoryManageStreams()[dev_rank]);
    PermuteWeight(permuted_input, {1, 0}, dev_rank);

    std::vector<size_t> slice_shape = {permuted_input.shape[0] / para_size, permuted_input.shape[1]};
    Tensor split_tensor = TensorParallelSplit1D(permuted_input, slice_shape, dev_rank, para_size);

    PermuteWeight(split_tensor, {1, 0}, dev_rank);
    return split_tensor;
  } else {
    KLLM_THROW("For 2D tensor, only RowPara or ColPara is supported");
  }
}

Tensor CommonModelWeightLoader::TensorParallelSplit(const Tensor& input_tensor, int dev_rank, size_t para_size,
                                                    TensorParallelMode tp_mode) {
  if (input_tensor.shape.size() == 1) {
    KLLM_CHECK_WITH_INFO(tp_mode == TensorParallelMode::RowPara, "For 1D tensor, only RowPara is supported");
    KLLM_CHECK_WITH_INFO(input_tensor.shape[0] % para_size == 0, "Tensor shape[0] must be divisible by para_size");
    std::vector<size_t> slice_shape = {input_tensor.shape[0] / para_size};
    return TensorParallelSplit1D(input_tensor, slice_shape, dev_rank, para_size);
  } else if (input_tensor.shape.size() == 2) {
    return TensorParallelSplit2D(input_tensor, dev_rank, para_size, tp_mode);
  } else {
    KLLM_THROW("Only 1D and 2D tensors are supported");
  }
}

Status CommonModelWeightLoader::Concat(const std::vector<Tensor>& inputs_tensor, Tensor& output, int dev_rank) {
  // 计算shape
  std::vector<size_t> shape = {0, inputs_tensor[0].shape[1]};
  for (size_t i = 0; i < inputs_tensor.size(); i++) {
    shape[0] += inputs_tensor[i].shape[0];
  }
  // 创建tensor（目标tensor始终在device上）
  Tensor concat = Tensor(MemoryLocation::LOCATION_DEVICE, inputs_tensor[0].dtype, shape, dev_rank);
  // 逐个拷贝，根据源tensor的location自动判断拷贝方向
  size_t offset = 0;
  for (size_t i = 0; i < inputs_tensor.size(); i++) {
    MemcpyKind memcpy_kind = GetMemcpyKind(inputs_tensor[i], concat);
    MemcpyAsync(concat.GetPtr<void>() + offset, inputs_tensor[i].GetPtr<void>(), inputs_tensor[i].GetTotalBytes(),
                memcpy_kind, context_->GetMemoryManageStreams()[dev_rank]);
    offset += inputs_tensor[i].GetTotalBytes();
  }
  output = concat;
  return Status();
}

Tensor CommonModelWeightLoader::MoveToDevice(const Tensor& host_tensor, int dev_rank) {
  Tensor output_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, host_tensor.dtype, host_tensor.shape, dev_rank);
  MemcpyAsync(output_tensor.GetPtr<void>(), host_tensor.GetPtr<void>(), host_tensor.GetTotalBytes(),
              MEMCPY_HOST_TO_DEVICE, context_->GetMemoryManageStreams()[dev_rank]);
  return output_tensor;
}

Status CommonModelWeightLoader::AutoMergeWeight(const std::vector<std::string>& inputs_weight_name,
                                                const std::string& merge_weight_name,
                                                std::unordered_map<std::string, Tensor>& dev_weights_map,
                                                int dev_rank) {
  std::vector<Tensor> input_tensors;
  for (const auto& name : inputs_weight_name) {
    if (dev_weights_map.find(name) == dev_weights_map.end()) {
      return Status(RetCode::RET_INVALID_ARGUMENT, fmt::format("Weight {} not found in dev_weights_map", name));
    }
    input_tensors.push_back(dev_weights_map.at(name));
  }

  Tensor merge_weight_tensor;
  STATUS_CHECK_RETURN(Concat(input_tensors, merge_weight_tensor, dev_rank));
  dev_weights_map[merge_weight_name] = merge_weight_tensor;

  for (const auto& name : inputs_weight_name) {
    dev_weights_map.erase(name);
  }
  return Status();
}

Status CommonModelWeightLoader::CastDeviceTensorType(Tensor& input_tensor, DataType new_dtype, int dev_rank) {
// Temporarily not support cast for Ascend and tops device
#ifdef ENABLE_ACL
  KLLM_THROW(fmt::format("Unsupported tensor cast for Ascend device"));
#elif ENABLE_TOPS
  KLLM_THROW(fmt::format("Unsupported tensor cast for Tops device"));
#endif
  if (input_tensor.dtype == new_dtype) {
    return Status();
  }
#ifdef ENABLE_CUDA
  if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_FP16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToHalf(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP32 && new_dtype == DataType::TYPE_BF16) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::FloatToBFloat16(
        input_tensor.GetPtr<float>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<bfloat16>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToFloat(input_tensor.GetPtr<float16>(),
                                                           input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
                                                           context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP32) {
    Tensor new_tensor = Tensor(MemoryLocation::LOCATION_DEVICE, new_dtype, input_tensor.shape, dev_rank);
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToFloat(
        input_tensor.GetPtr<bfloat16>(), input_tensor.GetElementNumber(), new_tensor.GetPtr<float>(),
        context_->GetMemoryManageStreams()[dev_rank].Get()));
    input_tensor = new_tensor;
  } else if (input_tensor.dtype == DataType::TYPE_BF16 && new_dtype == DataType::TYPE_FP16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::BFloat16ToHalf(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else if (input_tensor.dtype == DataType::TYPE_FP16 && new_dtype == DataType::TYPE_BF16) {
    input_tensor.dtype = new_dtype;
    // Inplace cast
    CUDA_CHECK_LAST_ERROR(llm_kernels::nvidia::HalfToBFloat16(input_tensor.GetPtr<void>(),
                                                              input_tensor.GetElementNumber(),
                                                              context_->GetMemoryManageStreams()[dev_rank].Get()));
  } else {
    KLLM_THROW(fmt::format("Unsupported tensor cast from type {} to {}", input_tensor.dtype, new_dtype));
  }
#endif
  return Status();
}

torch::Tensor CommonModelWeightLoader::GetTorchTensorFromTensor(const Tensor& tensor) {
  // 根据tensor的location确定device类型
  torch::Device device = (tensor.location == MemoryLocation::LOCATION_HOST)
                             ? torch::Device(torch::kCPU)
                             : torch::Device(torch::kCUDA, tensor.device_id);

  // 创建TensorOptions，设置device和dtype
  auto options = torch::TensorOptions().device(device).dtype(GetTorchTypeFromDataType(tensor.dtype));

  // 将shape从vector<size_t>转换为vector<int64_t>
  std::vector<int64_t> tensor_shape(tensor.shape.begin(), tensor.shape.end());

  // 使用from_blob创建torch::Tensor
  return torch::from_blob(tensor.GetPtr<void>(), tensor_shape, options);
}

Tensor CommonModelWeightLoader::GetTensorFromTorchTensor(const torch::Tensor& torch_tensor) {
  KLLM_CHECK_WITH_INFO(torch_tensor.is_contiguous(), "torch::Tensor must be contiguous");

  // 确定内存位置和设备ID
  MemoryLocation location =
      (torch_tensor.device().type() == torch::kCPU) ? MemoryLocation::LOCATION_HOST : MemoryLocation::LOCATION_DEVICE;
  int device_id = (torch_tensor.device().type() == torch::kCUDA) ? torch_tensor.device().index() : 0;

  // 转换数据类型和shape
  DataType dtype = GetDataTypeFromTorchType(torch_tensor.scalar_type());
  std::vector<size_t> shape(torch_tensor.sizes().begin(), torch_tensor.sizes().end());

  // 创建Tensor并拷贝数据
  Tensor tensor(location, dtype, shape, device_id);
  MemcpyKind memcpy_kind = (location == MemoryLocation::LOCATION_HOST) ? MEMCPY_HOST_TO_HOST : MEMCPY_DEVICE_TO_DEVICE;
  MemcpyAsync(tensor.GetPtr<void>(), torch_tensor.data_ptr(), tensor.GetTotalBytes(), memcpy_kind,
              context_->GetMemoryManageStreams()[device_id]);
  return tensor;
}

}  // namespace ksana_llm