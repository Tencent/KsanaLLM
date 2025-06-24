/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/models/base/buffer_manager.h"

namespace ksana_llm {

TensorBuffer* BufferManager::CreateBufferTensor(const std::string& name, const std::vector<size_t> shape,
                                                const DataType dtype, const MemoryLocation location, Stream* stream) {
  // Create a tensor with the specified parameters
  Tensor tensor(location, dtype, shape, rank_, nullptr, stream);

  // Create a TensorBuffer to manage the tensor
  auto buffer = std::make_unique<TensorBuffer>(name, tensor);

  // Update the total buffer size
  total_buffer_size_ += tensor.GetTotalBytes();

  // Add the buffer to the heap and return a pointer to it
  TensorBuffer* buffer_ptr = buffer.get();
  buffer_tensor_heap_.push_back(std::move(buffer));

  KLLM_LOG_DEBUG << fmt::format("Rank[{}] Create Buffer[{}]: {}", rank_, name, tensor.GetTotalBytes());
  return buffer_ptr;
}

Status BufferManager::ReleaseBufferTensors() {
  buffer_tensor_heap_.clear();
  total_buffer_size_ = 0ul;
  return Status();
}

}  // namespace ksana_llm
