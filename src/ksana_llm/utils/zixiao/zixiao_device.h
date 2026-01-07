/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "ksana_llm/utils/common_device.h"
#include "ksana_llm/utils/zixiao/tops_utils.h"

namespace ksana_llm {

class ZiXiaoDeviceContextManager {
 public:
  ZiXiaoDeviceContextManager();
  ~ZiXiaoDeviceContextManager();
};

template <>
struct StreamTypeTraits<DEVICE_TYPE_ZIXIAO> {
  typedef topsStream_t value_type;
};

template <>
class StreamT<DEVICE_TYPE_ZIXIAO> {
 public:
  explicit StreamT(int device_id);

  topsStream_t& Get();

  // Destroy the stream.
  void Destroy();

 private:
  // the related device with this stream.
  int device_id_;

  // The tops stream.
  topsStream_t tops_stream_;
};

template <>
struct EventTypeTraits<DEVICE_TYPE_ZIXIAO> {
  typedef topsEvent_t value_type;
};

template <>
class EventT<DEVICE_TYPE_ZIXIAO> {
 public:
  // Get the cuda event by reference.
  topsEvent_t& Get();

 private:
  // The cuda event.
  topsEvent_t tops_event_;
};

template <>
void EventCreateT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO>* event);

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_ZIXIAO>* event, unsigned int flags);

template <>
void EventDestroyT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event);

template <>
void StreamSynchronizeT<DEVICE_TYPE_ZIXIAO>(StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void DeviceSynchronizeT<DEVICE_TYPE_ZIXIAO>();

template <>
void SetDeviceT<DEVICE_TYPE_ZIXIAO>(int device_id);

template <>
void GetDeviceT<DEVICE_TYPE_ZIXIAO>(int* device_id);

template <>
void GetDeviceCountT<DEVICE_TYPE_ZIXIAO>(int* count);

template <>
void EventRecordT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event, StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void StreamWaitEventT<DEVICE_TYPE_ZIXIAO>(StreamT<DEVICE_TYPE_ZIXIAO> stream, EventT<DEVICE_TYPE_ZIXIAO> event);

template <>
void EventSynchronizeT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event);

template <>
void EventElapsedTimeT<DEVICE_TYPE_ZIXIAO>(float* ms, EventT<DEVICE_TYPE_ZIXIAO> start, EventT<DEVICE_TYPE_ZIXIAO> end);

template <>
void MemGetInfoT<DEVICE_TYPE_ZIXIAO>(size_t* free, size_t* total);

template <>
void MallocT<DEVICE_TYPE_ZIXIAO>(void** dev_ptr, size_t size);

template <>
void FreeT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr);

template <>
void HostAllocT<DEVICE_TYPE_ZIXIAO>(void** host_ptr, size_t size);

template <>
void FreeHostT<DEVICE_TYPE_ZIXIAO>(void* host_ptr);

template <>
void HostAllocMappedT<DEVICE_TYPE_ZIXIAO>(void** host_ptr, void** device_ptr, size_t size);

template <>
void FreeHostMappedT<DEVICE_TYPE_ZIXIAO>(void* host_ptr, void* device_ptr);

template <>
void MallocAsyncT<DEVICE_TYPE_ZIXIAO>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void FreeAsyncT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void MemsetAsyncT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void MemsetT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, int value, size_t count);

template <>
void MemcpyAsyncT<DEVICE_TYPE_ZIXIAO>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void Memcpy2DT<DEVICE_TYPE_ZIXIAO>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind);

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_ZIXIAO>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_ZIXIAO> stream);

template <>
void MemcpyT<DEVICE_TYPE_ZIXIAO>(void* dst, const void* src, size_t count, enum MemcpyKind kind);

template <>
class GetDataTypeT<DEVICE_TYPE_ZIXIAO> {
 public:
  template <class U>
  static DataType impl();
};

template <>
void* GetRuntimeContextT<DEVICE_TYPE_ZIXIAO>(int device_id);

// Check if TOPS runtime is still available.
template <>
bool IsDeviceRuntimeAvailableT<DEVICE_TYPE_ZIXIAO>();

}  // namespace ksana_llm
