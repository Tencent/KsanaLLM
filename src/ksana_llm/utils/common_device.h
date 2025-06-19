/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <cstddef>
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

template <int T>
struct StreamTypeTraits {
  typedef void value_type;
};

template <int T>
class StreamT {
 public:
  typename StreamTypeTraits<T>::value_type& Get();
};

template <int T>
struct EventTypeTraits {
  typedef void value_type;
};

template <int T>
class EventT {
 public:
  typename EventTypeTraits<T>::value_type& Get();
};

// Create event
template <int T>
void EventCreateT(EventT<T>* event);

// Create event with flags.
template <int T>
void EventCreateWithFlagsT(EventT<T>* event, unsigned int flags);

// Destroy event
template <int T>
void EventDestroyT(EventT<T> event);

// Synchronize stream.
template <int T>
void StreamSynchronizeT(StreamT<T> stream) {}

// Synchronize device.
template <int T>
void DeviceSynchronizeT() {}

// Set current device.
template <int T>
void SetDeviceT(int device_id) {}

// Get current device.
template <int T>
void GetDeviceT(int* device_id) {}

// Get device count.
template <int T>
void GetDeviceCountT(int* count) {}

// Record a event in stream.
template <int T>
void EventRecordT(EventT<T> event, StreamT<T> stream) {}

// Query event status
template <int T>
bool EventQueryT(EventT<T> event) {
  return true;
}

// Block stream until event finished.
template <int T>
void StreamWaitEventT(StreamT<T> stream, EventT<T> event);

template <int T>
void EventSynchronizeT(EventT<T> event);

template <int T>
void EventElapsedTimeT(float* ms, EventT<T> start, EventT<T> end);

// Get hbm memory info.
template <int T>
void MemGetInfoT(size_t* free, size_t* total);

// alloc memory on device.
template <int T>
void MallocT(void** dev_ptr, size_t size);

// free memory on device.
template <int T>
void FreeT(void* dev_ptr);

// alloc memory on host.
template <int T>
void HostAllocT(void** host_ptr, size_t size);

// free memoy on host
template <int T>
void FreeHostT(void* host_ptr);

// malloc memory on device
template <int T>
void MallocAsyncT(void** dev_ptr, size_t size, StreamT<T> stream);

// free memory on device
template <int T>
void FreeAsyncT(void* dev_ptr, StreamT<T> stream);

// Initializes or sets device memory to a value.
template <int T>
void MemsetAsyncT(void* dev_ptr, int value, size_t count, StreamT<T> stream);

template <int T>
void MemsetT(void* dev_ptr, int value, size_t count);

// The memory copy kind.
enum MemcpyKind {
  MEMCPY_HOST_TO_HOST = 0,
  MEMCPY_HOST_TO_DEVICE = 1,
  MEMCPY_DEVICE_TO_HOST = 2,
  MEMCPY_DEVICE_TO_DEVICE = 3,
};

// Copies data between host and device.
template <int T>
void MemcpyAsyncT(void* dst, const void* src, size_t count, enum MemcpyKind kind, StreamT<T> stream);

template <int T>
void Memcpy2DT(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
               enum MemcpyKind kind);

template <int T>
void Memcpy2DAsyncT(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                    enum MemcpyKind kind, StreamT<T> stream);

template <int T>
void MemcpyT(void* dst, const void* src, size_t count, enum MemcpyKind kind);

// Get the common data type from device data type.
template <int T>
class GetDataTypeT {
 public:
  template <class U>
  static DataType impl();
};

template <int T>
void* GetRuntimeContextT(int device_id);

}  // namespace ksana_llm
