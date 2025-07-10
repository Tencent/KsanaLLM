/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "c10/core/ScalarType.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

class Tensor;

// Make sure the value could not be assigned.
template <typename T>
class NotAssignable {
 public:
  friend class Tensor;

  // NOCC:runtime/explicit(Desinged like this for elegant)
  NotAssignable(T val);
  NotAssignable<T>& operator=(T val);
  NotAssignable(const NotAssignable<T>& other);
  operator T() const;

 private:
  void Set(const T& val);

  const T& Get() const;
  T& Get();

  void SetErrorMessage(const std::string& error_message);

 private:
  T value_;
  std::string error_message_;
};

// Trigger a check if the value is assigned.
template <typename T>
class TypeWithCheck {
 public:
  TypeWithCheck();

  // NOCC:runtime/explicit(Desinged like this for elegant)
  TypeWithCheck(T val);

  void SetChecker(std::function<void()> checker);

  TypeWithCheck<T>& operator=(T val);
  TypeWithCheck<T>& operator+=(T val);
  TypeWithCheck<T>& operator-=(T val);

  TypeWithCheck(const TypeWithCheck<T>& other);

  operator T() const;

 private:
  T value_;
  std::function<void()> checker_ = nullptr;
};

// Wrapper of shape type, trigger if the shape of any dim of shape is assigned.
class ShapeTypeWithCheck {
 public:
  friend class Tensor;
  ShapeTypeWithCheck();

  // NOCC:runtime/explicit(Designed like this for elegant)
  ShapeTypeWithCheck(const std::vector<size_t> val);

  ~ShapeTypeWithCheck();

  void SetChecker(std::function<void()> checker);

  ShapeTypeWithCheck& operator=(std::vector<size_t> val);

  TypeWithCheck<size_t>& operator[](std::vector<size_t>::size_type index);

  const TypeWithCheck<size_t>& operator[](std::vector<size_t>::size_type index) const;

  operator std::vector<size_t>() const;

  std::vector<TypeWithCheck<size_t>>::iterator begin();
  std::vector<TypeWithCheck<size_t>>::const_iterator begin() const;
  std::vector<TypeWithCheck<size_t>>::iterator end();
  std::vector<TypeWithCheck<size_t>>::const_iterator end() const;

  std::vector<size_t>::size_type size() const;

  bool empty() const;

  void resize(std::vector<size_t>::size_type count);

  void resize(std::vector<size_t>::size_type count, const std::vector<size_t>::value_type& value);

  TypeWithCheck<size_t>& back();
  const TypeWithCheck<size_t>& back() const;

  TypeWithCheck<size_t>& front();
  const TypeWithCheck<size_t>& front() const;

  std::vector<TypeWithCheck<size_t>>::iterator erase(std::vector<TypeWithCheck<size_t>>::iterator pos);

  std::vector<TypeWithCheck<size_t>>::iterator insert(std::vector<TypeWithCheck<size_t>>::const_iterator pos,
                                                      const TypeWithCheck<size_t>& value);

  bool operator==(const ShapeTypeWithCheck& other);

 private:
  std::vector<TypeWithCheck<size_t>> val_;
  std::function<void()> checker_ = nullptr;
};

// The tensor define, only support contigous memory layout.
class Tensor {
 public:
  // Initialize a empty tensor.
  Tensor();

  // Initialize the tensor, if data_ptr is not null, it will be used as data buffer.
  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id = -1,
         void* data_ptr = nullptr, Stream* stream = nullptr);

  ~Tensor();

  Tensor(const Tensor& other) { AssignMembers(other); }

  Tensor& operator=(const Tensor& other) {
    if (this != &other) {
      // Free underlying memory until no more tensor instances referenced.
      if (reference_.use_count() == 1) {
        FreeMemory();
      }
      AssignMembers(other);
    }
    return *this;
  }

 public:
  // Whether two tensor is equal.
  bool Equal(const Tensor& other) const;

  // Get the element number of this tensor.
  size_t GetElementNumber() const;

  // Get the byte of this tensor dtype.
  size_t GetDTypeSize() const;

  // Get the total bytes of this tensor.
  size_t GetTotalBytes() const;

  // Get pointer of block
  template <typename T>
  inline T* GetPtr() const {
    return reinterpret_cast<T*>(data_ptr.Get() + static_cast<size_t>(offset));
  }

  // Get tensor meta in string.
  std::string ToString() const;

  // Save to npy format file
  void SaveToNpyFile(const std::string& file_path);

  // Load tensor from npy file.
  void LoadFromNpyFile(const std::string& file_path);

 public:
  // The memory location, host or device.
  NotAssignable<MemoryLocation> location = MemoryLocation::LOCATION_UNKNOWN;

  // The rank of device, meaningless for host memory.
  NotAssignable<int> device_id = -1;

  // The data type of current tensor.
  TypeWithCheck<DataType> dtype = DataType::TYPE_INVALID;

  // The shape of current tensor.
  ShapeTypeWithCheck shape;

  // The underlying memory address.
  NotAssignable<void*> data_ptr = nullptr;

  // The data format, for ascend only now.
  DataFormat data_format = DataFormat::FORMAT_DEFAULT;

  // The offset based on data ptr.
  TypeWithCheck<size_t> offset = 0;

 private:
  // Free tensor memory.
  void FreeMemory();

  // Allocate tensor memory.
  void AllocateMemory();

  // Get location in string.
  std::string GetLocationString() const;

  // Assign every members.
  void AssignMembers(const Tensor& other);

  // Set check logic of current tensor.
  void InitializeChecker();

 private:
  // Whether the data buffer is shared with others, the shared buffer will not be free.
  bool is_shared_buffer_ = false;

  std::shared_ptr<int> reference_ = nullptr;

  // Check tensor memory size.
  std::function<void()> checker_ = nullptr;

  // The max buffer size, used to check tensor validation.
  size_t max_buffer_size_ = 0;

  // NOTE(karlluo): for NVIDIA GPU ref
  // https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
  // for Huawei NPU ref
  // https://support.enflame-tech.com/onlinedoc_dev_3.3/_static/topsplatform_html/3-guide
  // /programing_guide/content/source/memory_model.html
  // create device memory space with stream and memory pool as extra memory management.
  Stream* stream_{nullptr};

 public:
  // TODO(yancyliu): The following number should be removed later.
  // ////////////////////////////////////////////////////////////////
  Tensor* scales = nullptr;
  Tensor* zeros = nullptr;

  // g_idx indicates the scales row number corresponding to each row of weight
  Tensor* g_idx = nullptr;
  // perm is converted from g_idx, perm=torch.argsort(g_idx), perm is used in marlin backend to support gptq-desc
  Tensor* perm = nullptr;

  Tensor* input_scales = nullptr;
  Tensor* weight_scales = nullptr;

  void Fill(float f);
};

}  // namespace ksana_llm
