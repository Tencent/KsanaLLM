/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/common_device.h"

namespace ksana_llm {

template <>
struct StreamTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef cudaStream_t value_type;
};

template <>
class StreamT<DEVICE_TYPE_NVIDIA> {
 public:
  explicit StreamT(int device_id);

  // Get the cuda stream by reference..
  cudaStream_t& Get();

  // Destroy the stream.
  void Destroy();

 private:
  // the related device with this stream.
  int device_id_;

  // The cuda stream.
  cudaStream_t cuda_stream_;
};

template <>
struct EventTypeTraits<DEVICE_TYPE_NVIDIA> {
  typedef cudaEvent_t value_type;
};

template <>
class EventT<DEVICE_TYPE_NVIDIA> {
 public:
  // Get the cuda event by reference.
  cudaEvent_t& Get();

 private:
  // The cuda event.
  cudaEvent_t cuda_event_;
};

template <>
void EventCreateT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA>* event);

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_NVIDIA>* event, unsigned int flags);

template <>
void EventDestroyT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event);

template <>
void StreamSynchronizeT<DEVICE_TYPE_NVIDIA>(StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void DeviceSynchronizeT<DEVICE_TYPE_NVIDIA>();

template <>
void SetDeviceT<DEVICE_TYPE_NVIDIA>(int device_id);

template <>
void GetDeviceT<DEVICE_TYPE_NVIDIA>(int* device_id);

template <>
void GetDeviceCountT<DEVICE_TYPE_NVIDIA>(int* count);

template <>
void EventRecordT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event, StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
bool EventQueryT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event);

template <>
void StreamWaitEventT<DEVICE_TYPE_NVIDIA>(StreamT<DEVICE_TYPE_NVIDIA> stream, EventT<DEVICE_TYPE_NVIDIA> event);

template <>
void EventSynchronizeT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event);

template <>
void EventElapsedTimeT<DEVICE_TYPE_NVIDIA>(float* ms, EventT<DEVICE_TYPE_NVIDIA> start, EventT<DEVICE_TYPE_NVIDIA> end);

template <>
void MemGetInfoT<DEVICE_TYPE_NVIDIA>(size_t* free, size_t* total);

template <>
void MallocT<DEVICE_TYPE_NVIDIA>(void** dev_ptr, size_t size);

template <>
void FreeT<DEVICE_TYPE_NVIDIA>(void* dev_ptr);

template <>
void HostAllocT<DEVICE_TYPE_NVIDIA>(void** host_ptr, size_t size);

template <>
void FreeHostT<DEVICE_TYPE_NVIDIA>(void* host_ptr);

template <>
void HostAllocMappedT<DEVICE_TYPE_NVIDIA>(void** host_ptr, void** device_ptr, size_t size);

template <>
void FreeHostMappedT<DEVICE_TYPE_NVIDIA>(void* host_ptr, void* device_ptr);

template <>
void MallocAsyncT<DEVICE_TYPE_NVIDIA>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void FreeAsyncT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void MemsetAsyncT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void MemsetT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, int value, size_t count);

template <>
void MemcpyAsyncT<DEVICE_TYPE_NVIDIA>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void Memcpy2DT<DEVICE_TYPE_NVIDIA>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind);

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_NVIDIA>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_NVIDIA> stream);

template <>
void MemcpyT<DEVICE_TYPE_NVIDIA>(void* dst, const void* src, size_t count, enum MemcpyKind kind);

template <>
class GetDataTypeT<DEVICE_TYPE_NVIDIA> {
 public:
  template <class U>
  static DataType impl();

 private:
  template <class U>
  static DataType GetFloatType();

  template <class U>
  static DataType GetIntType();
};

template <>
void* GetRuntimeContextT<DEVICE_TYPE_NVIDIA>(int device_id);

}  // namespace ksana_llm
