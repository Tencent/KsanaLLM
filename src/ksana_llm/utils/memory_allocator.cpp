/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/memory_allocator.h"

#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

void MemoryAllocator::SetDevice(int device_id) { ::ksana_llm::SetDevice(device_id); }

void MemoryAllocator::GetDevice(int* device_id) { ::ksana_llm::GetDevice(device_id); }

void MemoryAllocator::Malloc(void** dev_ptr, size_t size) { ::ksana_llm::Malloc(dev_ptr, size); }

void MemoryAllocator::MallocAsync(void** dev_ptr, size_t size, Stream stream) {
  ::ksana_llm::MallocAsync(dev_ptr, size, stream);
}

void MemoryAllocator::MemsetAsync(void* dev_ptr, int value, size_t count, Stream stream) {
  ::ksana_llm::MemsetAsync(dev_ptr, value, count, stream);
}

void MemoryAllocator::Memset(void* dev_ptr, int value, size_t count) { ::ksana_llm::Memset(dev_ptr, value, count); }

void MemoryAllocator::MemcpyAsync(void* dst, const void* src, size_t count, enum MemcpyKind kind, Stream stream) {
  ::ksana_llm::MemcpyAsync(dst, src, count, kind, stream);
}

void MemoryAllocator::Memcpy(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  ::ksana_llm::Memcpy(dst, src, count, kind);
}

void MemoryAllocator::Free(void* dev_ptr) { ::ksana_llm::Free(dev_ptr); }

void MemoryAllocator::FreeAsync(void* dev_ptr, Stream stream) { ::ksana_llm::FreeAsync(dev_ptr, stream); }

void MemoryAllocator::HostAlloc(void** host_ptr, size_t size) { ::ksana_llm::HostAlloc(host_ptr, size); }

void MemoryAllocator::HostFree(void* host_ptr) { ::ksana_llm::HostFree(host_ptr); }

}  // namespace ksana_llm
