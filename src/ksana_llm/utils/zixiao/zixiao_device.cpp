/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/zixiao/zixiao_device.h"

#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

static ZiXiaoDeviceContextManager g_context_manager;

static constexpr uint64_t ZIXIAO_MEMCPY_FLAG = 0ul;

ZiXiaoDeviceContextManager::ZiXiaoDeviceContextManager() {}

ZiXiaoDeviceContextManager::~ZiXiaoDeviceContextManager() {}

StreamT<DEVICE_TYPE_ZIXIAO>::StreamT(int device_id) : device_id_(device_id) {
  TOPS_CHECK(topsSetDevice(device_id_));
  TOPS_CHECK(topsStreamCreate(&tops_stream_));
}

void StreamT<DEVICE_TYPE_ZIXIAO>::Destroy() {
  TOPS_CHECK(topsSetDevice(device_id_));
  TOPS_CHECK(topsStreamDestroy(tops_stream_));
}

topsStream_t& StreamT<DEVICE_TYPE_ZIXIAO>::Get() { return tops_stream_; }

topsEvent_t& EventT<DEVICE_TYPE_ZIXIAO>::Get() { return tops_event_; }

template <>
void EventCreateT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO>* event) {
  // Default enable multiple stream sync, and enable timeline.
  TOPS_CHECK(topsEventCreate(&(event->Get())));
}

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_ZIXIAO>* event, unsigned int flags) {
  TOPS_CHECK(topsEventCreate(&(event->Get())));
}

template <>
void EventDestroyT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event) {
  TOPS_CHECK(topsEventDestroy(event.Get()));
}

template <>
void StreamSynchronizeT<DEVICE_TYPE_ZIXIAO>(StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  TOPS_CHECK(topsStreamSynchronize(stream.Get()));
}

template <>
void DeviceSynchronizeT<DEVICE_TYPE_ZIXIAO>() {
  TOPS_CHECK(topsDeviceSynchronize());
}

template <>
void SetDeviceT<DEVICE_TYPE_ZIXIAO>(int device_id) {
  TOPS_CHECK(topsSetDevice(device_id));
}

template <>
void GetDeviceT<DEVICE_TYPE_ZIXIAO>(int* device_id) {
  TOPS_CHECK(topsGetDevice(device_id));
}

template <>
void GetDeviceCountT<DEVICE_TYPE_ZIXIAO>(int* count) {
  TOPS_CHECK(topsGetDeviceCount(count));
}

template <>
void EventRecordT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event, StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  TOPS_CHECK(topsEventRecord(event.Get(), stream.Get()));
}

template <>
void StreamWaitEventT<DEVICE_TYPE_ZIXIAO>(StreamT<DEVICE_TYPE_ZIXIAO> stream, EventT<DEVICE_TYPE_ZIXIAO> event) {
  return;
}

template <>
void EventSynchronizeT<DEVICE_TYPE_ZIXIAO>(EventT<DEVICE_TYPE_ZIXIAO> event) {
  TOPS_CHECK(topsEventSynchronize(event.Get()));
}

template <>
void EventElapsedTimeT<DEVICE_TYPE_ZIXIAO>(float* ms, EventT<DEVICE_TYPE_ZIXIAO> start,
                                           EventT<DEVICE_TYPE_ZIXIAO> end) {
  TOPS_CHECK(topsEventElapsedTime(ms, start.Get(), end.Get()));
}

template <>
void MemGetInfoT<DEVICE_TYPE_ZIXIAO>(size_t* free, size_t* total) {
  TOPS_CHECK(topsMemGetInfo(free, total));
}

template <>
void MallocT<DEVICE_TYPE_ZIXIAO>(void** dev_ptr, size_t size) {
  TOPS_CHECK(topsMalloc(dev_ptr, size));
}

template <>
void FreeT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr) {
  TOPS_CHECK(topsFree(dev_ptr));
}

template <>
void HostAllocT<DEVICE_TYPE_ZIXIAO>(void** host_ptr, size_t size) {
  TOPS_CHECK(topsHostMalloc(host_ptr, size));
}

template <>
void FreeHostT<DEVICE_TYPE_ZIXIAO>(void* host_ptr) {
  TOPS_CHECK(topsHostFree(host_ptr));
}

template <>
void HostAllocMappedT<DEVICE_TYPE_ZIXIAO>(void** host_ptr, void** device_ptr, size_t size) {
  throw std::runtime_error("HostAllocMappedT is not supported in Zixiao device.");
}

template <>
void FreeHostMappedT<DEVICE_TYPE_ZIXIAO>(void* host_ptr, void* device_ptr) {
  throw std::runtime_error("FreeHostMappedT is not supported in Zixiao device.");
}

template <>
void MallocAsyncT<DEVICE_TYPE_ZIXIAO>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  TOPS_CHECK(topsMallocAsync(dev_ptr, size, stream.Get(), ZIXIAO_MEMCPY_FLAG));
}

template <>
void FreeAsyncT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  TOPS_CHECK(topsFreeAsync(dev_ptr, stream.Get()));
}

template <>
void MemsetAsyncT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  if (count > 0) {
    TOPS_CHECK(topsMemsetAsync(dev_ptr, value, count, stream.Get()));
  }
}

template <>
void MemsetT<DEVICE_TYPE_ZIXIAO>(void* dev_ptr, int value, size_t count) {
  if (count > 0) {
    TOPS_CHECK(topsMemset(dev_ptr, value, count));
  }
}

topsMemcpyKind GetTopsMemcpyKind(enum MemcpyKind kind) {
  switch (kind) {
    case MEMCPY_HOST_TO_HOST:
      return topsMemcpyHostToHost;
    case MEMCPY_HOST_TO_DEVICE:
      return topsMemcpyHostToDevice;
    case MEMCPY_DEVICE_TO_HOST:
      return topsMemcpyDeviceToHost;
    default:
      return topsMemcpyDeviceToDevice;
  }
}

template <>
void MemcpyAsyncT<DEVICE_TYPE_ZIXIAO>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  if (count > 0) {
    TOPS_CHECK(topsMemcpyAsync(dst, src, count, GetTopsMemcpyKind(kind), stream.Get()));
  }
}

template <>
void Memcpy2DT<DEVICE_TYPE_ZIXIAO>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind) {
  for (size_t row_idx = 0; row_idx < height; ++row_idx) {
    TOPS_CHECK(topsMemcpy(dst + row_idx * dpitch, src + row_idx * spitch, width, GetTopsMemcpyKind(kind)));
  }
}

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_ZIXIAO>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_ZIXIAO> stream) {
  for (size_t row_idx = 0; row_idx < height; ++row_idx) {
    TOPS_CHECK(
        topsMemcpyAsync(dst + row_idx * dpitch, src + row_idx * spitch, width, GetTopsMemcpyKind(kind), stream.Get()));
  }
}

template <>
void MemcpyT<DEVICE_TYPE_ZIXIAO>(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  if (count > 0) {
    TOPS_CHECK(topsMemcpy(dst, src, count, GetTopsMemcpyKind(kind)));
  }
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl() {
  if (std::is_same<U, float>::value || std::is_same<U, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<U, float16>::value || std::is_same<U, const float16>::value) {
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

template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<float>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<float16>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<int32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<int8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<uint8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<uint32_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<uint64_t>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<bool>();
template DataType GetDataTypeT<DEVICE_TYPE_ZIXIAO>::impl<char>();

template <>
void* GetRuntimeContextT<DEVICE_TYPE_ZIXIAO>(int device_id) {
  return nullptr;
}

}  // namespace ksana_llm
