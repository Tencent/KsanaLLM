/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>

#include "ksana_llm/utils/memory_allocator_interface.h"

namespace ksana_llm {

// The default memory allocator.
class MemoryAllocator : public MemoryAllocatorInterface {
 public:
  virtual ~MemoryAllocator() {}

  // Set current device id.
  virtual void SetDevice(int device_id) override;

  // Get current device id.
  virtual void GetDevice(int* device_id) override;

  // Malloc device memory.
  virtual void Malloc(void** dev_ptr, size_t size) override;
  virtual void MallocAsync(void** dev_ptr, size_t size, Stream stream) override;

  // Memset
  virtual void MemsetAsync(void* dev_ptr, int value, size_t count, Stream stream) override;
  virtual void Memset(void* dev_ptr, int value, size_t count) override;

  // Memcopy
  virtual void MemcpyAsync(void* dst, const void* src, size_t count, enum MemcpyKind kind, Stream stream) override;
  virtual void Memcpy(void* dst, const void* src, size_t count, enum MemcpyKind kind) override;

  // Free device memory.
  virtual void Free(void* dev_ptr) override;
  virtual void FreeAsync(void* dev_ptr, Stream stream) override;

  // Malloc host memory.
  virtual void HostAlloc(void** host_ptr, size_t size) override;

  // Free host memory.
  virtual void HostFree(void* host_ptr) override;
};

}  // namespace ksana_llm
