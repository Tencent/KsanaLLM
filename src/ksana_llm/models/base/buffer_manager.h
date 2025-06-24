/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once
#include <atomic>
#include <mutex>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

/**
 * @brief TensorBuffer class manages tensors created by BufferManager::CreateBufferTensor.
 */
class TensorBuffer {
 public:
  TensorBuffer() = default;
  TensorBuffer(const std::string& name, const Tensor& tensor) : name_(name), in_use_(false) { tensors_[0] = tensor; }

  // Get Tensors for layer output, values in tensor will be changed
  std::vector<Tensor>& GetTensors() {
    bool expected = false;
    if (!in_use_.compare_exchange_strong(expected, true)) {
      throw std::runtime_error(name_ + " is in use");
    }
    return tensors_;
  }

  void FreeTensors() {
    bool expected = true;
    if (!in_use_.compare_exchange_strong(expected, false)) {
      throw std::runtime_error(name_ + " is not in use");
    }
  }

  bool IsInUse() const { return in_use_.load(); }
  const std::string& GetName() { return name_; }

 private:
  std::string name_;
  std::vector<Tensor> tensors_{1};   // The managed tensor
  std::atomic<bool> in_use_{false};  // Flag indicating if the tensor is currently in use (atomic for thread safety)
};

class TensorBufferScope {
 public:
  TensorBufferScope(TensorBuffer* buf, const std::string& info) : buf_(buf), tensors_(buf->GetTensors()), info_(info) {}
  ~TensorBufferScope() { buf_->FreeTensors(); }

  std::vector<Tensor>& GetTensors() { return tensors_; }

 private:
  TensorBuffer* buf_;
  std::vector<Tensor>& tensors_;
  std::string info_;
};

#define CREATE_BUFFER_SCOPE(buffer_name, tensor_buffer_name)                                                         \
  TensorBufferScope buffer_name##_scope(tensor_buffer_name, std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
  auto& buffer_name = buffer_name##_scope.GetTensors();

// TODO(robertyuan): in the future, BufferManager handle all buffers usage, other than just creating buffers.
class BufferManager {
 public:
  // Disable a default constructor
  BufferManager() {}

  ~BufferManager() = default;

  void SetRank(int rank) { rank_ = rank; }

  /**
   * @brief Create a TensorBuffer object
   *
   * @param shape The shape of the tensor
   * @param dtype The data type of the tensor
   * @param location The memory location of the tensor
   * @return TensorBuffer* Pointer to the created TensorBuffer, nullptr if creation failed
   */
  TensorBuffer* CreateBufferTensor(const std::string& name, const std::vector<size_t> shape, const DataType dtype,
                                   const MemoryLocation location = MemoryLocation::LOCATION_DEVICE,
                                   Stream* stream = nullptr);

  /**
   * @brief Release all buffer tensors
   *
   * @return Status Operation status
   */
  Status ReleaseBufferTensors();

  /**
   * @brief Get the total memory used by buffer tensors
   *
   * @return size_t Total memory in bytes
   */
  const size_t GetBufferTensorsMemoryUsed() { return total_buffer_size_; }

 private:
  int rank_;
  size_t total_buffer_size_{0ul};

  // Record all buffer tensors
  std::vector<std::unique_ptr<TensorBuffer>> buffer_tensor_heap_;
};

}  // namespace ksana_llm
