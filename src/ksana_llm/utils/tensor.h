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
#include "ksana_llm/utils/dynamic_memory_pool.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"

namespace ksana_llm {

// The tensor define, only support contigous memory layout.
class Tensor {
 public:
  // Initialize a empty tensor.
  explicit Tensor(const std::string& name = "");

  // Initialize the tensor, if data_ptr is not null, it will be used as data buffer.
  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id, void* data_ptr,
         Stream* stream, bool lazy_allocate = false, const std::string& name = "");

  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, bool lazy_allocate = false,
         const std::string& name = "")
      : Tensor(location, dtype, shape, -1, nullptr, nullptr, lazy_allocate, name) {}

  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id,
         bool lazy_allocate = false, const std::string& name = "")
      : Tensor(location, dtype, shape, device_id, nullptr, nullptr, lazy_allocate, name) {}

  Tensor(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id, void* data_ptr,
         bool lazy_allocate = false, const std::string& name = "")
      : Tensor(location, dtype, shape, device_id, data_ptr, nullptr, lazy_allocate, name) {}

  ~Tensor();

  Tensor(const Tensor& other);

  Tensor& operator=(const Tensor& other);

 public:
  // Allocate and release buffer memory.
  void Acquire();
  void Release();

  // Release and reallocate buffer memory.
  void ReallocateMemory(MemoryLocation location, DataType dtype, const std::vector<size_t>& shape, int device_id);

  // Whether two tensor is equal.
  bool Equal(const Tensor& other) const;

  // Get the element number of this tensor.
  size_t GetElementNumber() const;

  // Get the byte of this tensor dtype.
  size_t GetDTypeSize() const;

  // Get the total bytes of this tensor.
  size_t GetTotalBytes() const;

  // Get a view of the current tensor that shares the underlying storage, possibly with a different shape and offset
  Tensor GetView(const std::vector<size_t>& shape, const size_t offset = 0) const;

  // Get pointer of block
  template <typename T>
  inline T* GetPtr(bool check_empty = true) const {
    return reinterpret_cast<T*>(GetPtrImpl(check_empty));
  }

  // Get tensor meta in string.
  std::string ToString() const;

  // Save to npy format file
  void SaveToNpyFile(const std::string& file_path);

  // Load tensor from npy file.
  void LoadFromNpyFile(const std::string& file_path);

 public:
  // Identify a tensor, for debug and readable.
  std::string name;

  // The memory location, host or device.
  MemoryLocation location = MemoryLocation::LOCATION_UNKNOWN;

  // The rank of device, meaningless for host memory.
  int device_id = -1;

  // The data type of current tensor.
  DataType dtype = DataType::TYPE_INVALID;

  // The shape of current tensor.
  std::vector<size_t> shape;

  // The data format, for ascend only now.
  DataFormat data_format = DataFormat::FORMAT_DEFAULT;

 private:
  // Get location in string.
  std::string GetLocationString() const;

  // Assign every members.
  void AssignMembers(const Tensor& other);

  // The implementation of GetPtr.
  uint8_t* GetPtrImpl(bool check_empty) const;

  // Implement of allocation and release.
  void AcquireImpl();
  void ReleaseImpl();

 private:
  // The underlying memory address.
  void* data_ptr = nullptr;

  // Whether the data buffer is shared with others, the shared buffer will not be free.
  bool is_shared_buffer_ = false;

  std::shared_ptr<int> reference_ = nullptr;

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

  // input_alpha is used as a coefficient for input
  Tensor* input_alpha = nullptr;

  void Fill(float f);
};

// Whether whether memory is out of bound.
class MemoryChecker {
 public:
  struct MemoryMeta {
    void* head_ptr;
    size_t head_bytes;

    void* tail_ptr;
    size_t tail_bytes;

    uint8_t expect_value;
  };

 public:
  MemoryChecker();

  // Add & remove memory block from check list.
  static void AddMemoryBlock(const std::string& name, int rank, void* head_ptr, size_t head_bytes, void* tail_ptr,
                             size_t tail_bytes, uint8_t expect_value = 0);

  static void RemoveMemoryBlock(const std::string& name, int rank);

  // Check whether emory
  static void CheckMemory(int rank, bool synchronize_device);

  // Whether memory checker is enabled.
  static bool Enabled();

 private:
  static bool CheckMemoryValueImpl(void* check_ptr, size_t check_bytes, uint8_t expect_value);

 private:
  // rank id to memory check meta.
  static inline std::vector<std::unordered_map<std::string, MemoryMeta>> check_memory_map_;

  inline static bool enabled_ = (std::getenv("ENABLE_MEMORY_CHECK") != nullptr);
};

}  // namespace ksana_llm
