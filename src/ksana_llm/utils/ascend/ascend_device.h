/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <unordered_map>

#include "ksana_llm/utils/common_device.h"

#include "atb/context.h"

namespace ksana_llm {

class AscendDeviceContextManager {
 public:
  AscendDeviceContextManager();
  ~AscendDeviceContextManager();

  // Get aclrtContext of device_id
  aclrtContext& GetDeviceContext(int device_id);

  // Get atb::Context of device_id
  atb::Context* GetDeviceATBContext(int device_id);

 private:
  std::unordered_map<int, aclrtContext> acl_contexts_;
  // device_id => context
  // TODO(karlluo): maybe reuse aclrtContext and aclrtStream
  std::unordered_map<int, atb::Context*> acl_atb_contexts_;
};

template <>
struct StreamTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef aclrtStream value_type;
};

template <>
class StreamT<DEVICE_TYPE_ASCEND> {
 public:
  explicit StreamT(int device_id);

  aclrtStream& Get();

  // Destroy the stream.
  void Destroy();

 private:
  // the related device with this stream.
  int device_id_;

  // The acl stream.
  aclrtStream acl_stream_;
};

template <>
struct EventTypeTraits<DEVICE_TYPE_ASCEND> {
  typedef aclrtEvent value_type;
};

template <>
class EventT<DEVICE_TYPE_ASCEND> {
 public:
  // Get the cuda event by reference.
  aclrtEvent& Get();

 private:
  // The cuda event.
  aclrtEvent acl_event_;
};

template <>
void EventCreateT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND>* event);

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_ASCEND>* event, unsigned int flags);

template <>
void EventDestroyT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event);

template <>
void StreamSynchronizeT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void DeviceSynchronizeT<DEVICE_TYPE_ASCEND>();

template <>
void SetDeviceT<DEVICE_TYPE_ASCEND>(int device_id);

template <>
void GetDeviceT<DEVICE_TYPE_ASCEND>(int* device_id);

template <>
void GetDeviceCountT<DEVICE_TYPE_ASCEND>(int* count);

template <>
void EventRecordT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event, StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void StreamWaitEventT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream, EventT<DEVICE_TYPE_ASCEND> event);

template <>
void EventSynchronizeT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event);

template <>
void EventElapsedTimeT<DEVICE_TYPE_ASCEND>(float* ms, EventT<DEVICE_TYPE_ASCEND> start, EventT<DEVICE_TYPE_ASCEND> end);

template <>
void MemGetInfoT<DEVICE_TYPE_ASCEND>(size_t* free, size_t* total);

template <>
void MallocT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size);

template <>
void FreeT<DEVICE_TYPE_ASCEND>(void* dev_ptr);

template <>
void HostAllocT<DEVICE_TYPE_ASCEND>(void** host_ptr, size_t size);

template <>
void FreeHostT<DEVICE_TYPE_ASCEND>(void* host_ptr);

template <>
void HostAllocMappedT<DEVICE_TYPE_ASCEND>(void** host_ptr, void** device_ptr, size_t size);

template <>
void FreeHostMappedT<DEVICE_TYPE_ASCEND>(void* host_ptr, void* device_ptr);

template <>
void MallocAsyncT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void FreeAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void MemsetAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void MemsetT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count);

template <>
void MemcpyAsyncT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void Memcpy2DT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind);

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_ASCEND> stream);

template <>
void MemcpyT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind);

template <>
class GetDataTypeT<DEVICE_TYPE_ASCEND> {
 public:
  template <class U>
  static DataType impl();
};

template <>
void* GetRuntimeContextT<DEVICE_TYPE_ASCEND>(int device_id);

// Check if ACL runtime is still available.
template <>
bool IsDeviceRuntimeAvailableT<DEVICE_TYPE_ASCEND>();

}  // namespace ksana_llm
