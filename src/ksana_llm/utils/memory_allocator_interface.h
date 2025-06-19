/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

// The memory utils interface, used to mock memory operation for testing.
class MemoryAllocatorInterface {
 public:
  ~MemoryAllocatorInterface() {}

  // Set current device id.
  virtual void SetDevice(int device_id) = 0;

  // Get current device id.
  virtual void GetDevice(int* device_id) = 0;

  // Malloc device memory.
  virtual void Malloc(void** dev_ptr, size_t size) = 0;
  virtual void MallocAsync(void** dev_ptr, size_t size, Stream stream) = 0;

  // Memset
  virtual void MemsetAsync(void* dev_ptr, int value, size_t count, Stream stream) = 0;
  virtual void Memset(void* dev_ptr, int value, size_t count) = 0;

  // Memcopy
  virtual void MemcpyAsync(void* dst, const void* src, size_t count, enum MemcpyKind kind, Stream stream) = 0;
  virtual void Memcpy(void* dst, const void* src, size_t count, enum MemcpyKind kind) = 0;

  // Free device memory.
  virtual void Free(void* dev_ptr) = 0;
  virtual void FreeAsync(void* dev_ptr, Stream stream) = 0;

  // Malloc host memory.
  virtual void HostAlloc(void** host_ptr, size_t size) = 0;

  // Free host memory.
  virtual void HostFree(void* host_ptr) = 0;
};

}  // namespace ksana_llm
