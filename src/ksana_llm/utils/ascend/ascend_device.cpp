/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/utils/ascend/ascend_device.h"

#include <atomic>

#include "3rdparty/LLM_kernels/csrc/utils/ascend/common.h"
#include "ksana_llm/utils/ascend/acl_utils.h"
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

// Flag to track if the Ascend runtime is still available
// This is set to false when g_context_manager is destroyed
static std::atomic<bool> g_ascend_runtime_available{false};

static AscendDeviceContextManager g_context_manager;

AscendDeviceContextManager::AscendDeviceContextManager() {
  ACL_CHECK(aclInit(nullptr));
  uint32_t dev_count = 0;
  ACL_CHECK(aclrtGetDeviceCount(&dev_count));
  for (uint32_t dev_id = 0; dev_id < dev_count; ++dev_id) {
    aclrtContext context;
    ACL_CHECK(aclrtSetDevice(dev_id));
    ACL_CHECK(aclrtCreateContext(&context, dev_id));
    acl_contexts_[dev_id] = context;
    atb::Context* atb_context_ptr = nullptr;
    ATB_CHECK_RET(atb::CreateContext(&atb_context_ptr));
    acl_atb_contexts_[dev_id] = atb_context_ptr;
  }
  // Mark runtime as available after successful initialization
  g_ascend_runtime_available.store(true, std::memory_order_release);
}

aclrtContext& AscendDeviceContextManager::GetDeviceContext(int device_id) { return acl_contexts_[device_id]; }

atb::Context* AscendDeviceContextManager::GetDeviceATBContext(int device_id) { return acl_atb_contexts_[device_id]; }

AscendDeviceContextManager::~AscendDeviceContextManager() {
  // Mark runtime as unavailable before cleanup
  g_ascend_runtime_available.store(false, std::memory_order_release);
  for (auto& [dev_id, context] : acl_contexts_) {
    ACL_CHECK(aclrtDestroyContext(context));
    ATB_CHECK_RET(DestroyContext(acl_atb_contexts_[dev_id]));
  }
}

StreamT<DEVICE_TYPE_ASCEND>::StreamT(int device_id) : device_id_(device_id) {
  ACL_CHECK(aclrtSetDevice(device_id_));
  ACL_CHECK(aclrtCreateStream(&acl_stream_));

  g_context_manager.GetDeviceATBContext(device_id)->SetExecuteStream(acl_stream_);
}

void StreamT<DEVICE_TYPE_ASCEND>::Destroy() {
  ACL_CHECK(aclrtSetDevice(device_id_));
  ACL_CHECK(aclrtDestroyStream(acl_stream_));
}

aclrtStream& StreamT<DEVICE_TYPE_ASCEND>::Get() { return acl_stream_; }

aclrtEvent& EventT<DEVICE_TYPE_ASCEND>::Get() { return acl_event_; }

template <>
void EventCreateT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND>* event) {
  // Default enable multiple stream sync, and enable timeline.
  ACL_CHECK(aclrtCreateEventWithFlag(&(event->Get()), ACL_EVENT_SYNC | ACL_EVENT_TIME_LINE));
}

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_ASCEND>* event, unsigned int flags) {
  unsigned int acl_flags = ACL_EVENT_TIME_LINE;
  if (flags & EVENT_DISABLE_TIMING) {
    acl_flags = acl_flags ^ ACL_EVENT_TIME_LINE;
  }

  // Enable sync between multiple streams.
  acl_flags = acl_flags | ACL_EVENT_SYNC;
  ACL_CHECK(aclrtCreateEventWithFlag(&(event->Get()), acl_flags));
}

template <>
void EventDestroyT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtDestroyEvent(event.Get()));
}

template <>
void StreamSynchronizeT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtSynchronizeStream(stream.Get()));
}

template <>
void DeviceSynchronizeT<DEVICE_TYPE_ASCEND>() {
  ACL_CHECK(aclrtSynchronizeDevice());
}

template <>
void SetDeviceT<DEVICE_TYPE_ASCEND>(int device_id) {
  ACL_CHECK(aclrtSetCurrentContext(g_context_manager.GetDeviceContext(device_id)));
  ACL_CHECK(aclrtSetDevice(device_id));
}

template <>
void GetDeviceT<DEVICE_TYPE_ASCEND>(int* device_id) {
  ACL_CHECK(aclrtGetDevice(device_id));
}

template <>
void GetDeviceCountT<DEVICE_TYPE_ASCEND>(int* count) {
  uint32_t tmp_count = 0;
  ACL_CHECK(aclrtGetDeviceCount(&tmp_count));
  *count = static_cast<int>(tmp_count);
}

template <>
void EventRecordT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtRecordEvent(event.Get(), stream.Get()));
}

template <>
void StreamWaitEventT<DEVICE_TYPE_ASCEND>(StreamT<DEVICE_TYPE_ASCEND> stream, EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtStreamWaitEvent(stream.Get(), event.Get()));
}

template <>
void EventSynchronizeT<DEVICE_TYPE_ASCEND>(EventT<DEVICE_TYPE_ASCEND> event) {
  ACL_CHECK(aclrtSynchronizeEvent(event.Get()));
}

template <>
void EventElapsedTimeT<DEVICE_TYPE_ASCEND>(float* ms, EventT<DEVICE_TYPE_ASCEND> start,
                                           EventT<DEVICE_TYPE_ASCEND> end) {
  ACL_CHECK(aclrtEventElapsedTime(ms, start.Get(), end.Get()));
}

template <>
void MemGetInfoT<DEVICE_TYPE_ASCEND>(size_t* free, size_t* total) {
  ACL_CHECK(aclrtGetMemInfo(ACL_HBM_MEM, free, total));
}

template <>
void MallocT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size) {
  ACL_CHECK(aclrtMalloc(dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

template <>
void FreeT<DEVICE_TYPE_ASCEND>(void* dev_ptr) {
  ACL_CHECK(aclrtFree(dev_ptr));
}

template <>
void HostAllocT<DEVICE_TYPE_ASCEND>(void** host_ptr, size_t size) {
  ACL_CHECK(aclrtMallocHost(host_ptr, size));
}

template <>
void FreeHostT<DEVICE_TYPE_ASCEND>(void* host_ptr) {
  ACL_CHECK(aclrtFreeHost(host_ptr));
}

template <>
void HostAllocMappedT<DEVICE_TYPE_ASCEND>(void** host_ptr, void** device_ptr, size_t size) {
  // Ascend does not support true mapped host memory allocation.
  // As a fallback, we allocate separate host and device memory.
  // Note: This requires explicit memory copy between host and device.
  ACL_CHECK(aclrtMallocHost(host_ptr, size));
  ACL_CHECK(aclrtMalloc(device_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

template <>
void FreeHostMappedT<DEVICE_TYPE_ASCEND>(void* host_ptr, void* device_ptr) {
  // Free both host and device memory allocated by HostAllocMappedT
  if (host_ptr) {
    ACL_CHECK(aclrtFreeHost(host_ptr));
  }
  if (device_ptr) {
    ACL_CHECK(aclrtFree(device_ptr));
  }
}

template <>
void MallocAsyncT<DEVICE_TYPE_ASCEND>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtMalloc(dev_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
}

template <>
void FreeAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtFree(dev_ptr));
}

template <>
void MemsetAsyncT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_ASCEND> stream) {
  if (count > 0) {
    ACL_CHECK(aclrtMemsetAsync(dev_ptr, count, value, count, stream.Get()));
  }
}

template <>
void MemsetT<DEVICE_TYPE_ASCEND>(void* dev_ptr, int value, size_t count) {
  if (count > 0) {
    ACL_CHECK(aclrtMemset(dev_ptr, count, value, count));
  }
}

aclrtMemcpyKind GetAclMemcpyKind(enum MemcpyKind kind) {
  switch (kind) {
    case MEMCPY_HOST_TO_HOST:
      return ACL_MEMCPY_HOST_TO_HOST;
    case MEMCPY_HOST_TO_DEVICE:
      return ACL_MEMCPY_HOST_TO_DEVICE;
    case MEMCPY_DEVICE_TO_HOST:
      return ACL_MEMCPY_DEVICE_TO_HOST;
    default:
      return ACL_MEMCPY_DEVICE_TO_DEVICE;
  }
}

template <>
void MemcpyAsyncT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_ASCEND> stream) {
  if (count > 0) {
    ACL_CHECK(aclrtMemcpyAsync(dst, count, src, count, GetAclMemcpyKind(kind), stream.Get()));
  }
}

template <>
void Memcpy2DT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind) {
  ACL_CHECK(aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, GetAclMemcpyKind(kind)));
}

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_ASCEND>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_ASCEND> stream) {
  ACL_CHECK(aclrtMemcpy2dAsync(dst, dpitch, src, spitch, width, height, GetAclMemcpyKind(kind), stream.Get()));
}

template <>
void MemcpyT<DEVICE_TYPE_ASCEND>(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  if (count > 0) {
    ACL_CHECK(aclrtMemcpy(dst, count, src, count, GetAclMemcpyKind(kind)));
  }
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl() {
  if (std::is_same<U, float>::value || std::is_same<U, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<U, aclFloat16>::value || std::is_same<U, const aclFloat16>::value) {
    return TYPE_FP16;
  } else if (std::is_same<U, int32_t>::value || std::is_same<U, const int32_t>::value) {
    return TYPE_INT32;
  } else if (std::is_same<U, int64_t>::value || std::is_same<U, const int64_t>::value) {
    return TYPE_INT64;
  } else if (std::is_same<U, int8_t>::value || std::is_same<U, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<U, uint8_t>::value || std::is_same<U, const uint8_t>::value) {
    return TYPE_UINT8;
  } else if (std::is_same<U, uint32_t>::value || std::is_same<U, const uint32_t>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<U, uint64_t>::value || std::is_same<U, const uint64_t>::value) {
    return TYPE_UINT64;
  } else if (std::is_same<U, bool>::value || std::is_same<U, const bool>::value) {
    return TYPE_BOOL;
  } else if (std::is_same<U, char>::value || std::is_same<U, const char>::value) {
    return TYPE_BYTES;
  } else if (std::is_pointer<U>::value) {
    return TYPE_POINTER;
  } else {
    return TYPE_INVALID;
  }
}

template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<float>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<aclFloat16>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<int32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<int8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<uint8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<uint32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<uint64_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<bool>();
template DataType GetDataTypeT<DEVICE_TYPE_ASCEND>::impl<char>();

template <>
void* GetRuntimeContextT<DEVICE_TYPE_ASCEND>(int device_id) {
  void* runtime_context_ptr = nullptr;
  runtime_context_ptr = reinterpret_cast<void*>(g_context_manager.GetDeviceATBContext(device_id));
  return runtime_context_ptr;
}

template <>
bool IsDeviceRuntimeAvailableT<DEVICE_TYPE_ASCEND>() {
  // Check if the Ascend runtime (g_context_manager) is still valid
  return g_ascend_runtime_available.load(std::memory_order_acquire);
}

}  // namespace ksana_llm
