/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <iostream>

#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/nvidia/cuda_utils.h"
#include "ksana_llm/utils/nvidia/nvidia_device.h"

namespace ksana_llm {

StreamT<DEVICE_TYPE_NVIDIA>::StreamT(int device_id) : device_id_(device_id) {
  CUDA_CHECK(cudaSetDevice(device_id_));
  CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
}

void StreamT<DEVICE_TYPE_NVIDIA>::Destroy() {
  CUDA_CHECK(cudaSetDevice(device_id_));
  CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
}

cudaStream_t& StreamT<DEVICE_TYPE_NVIDIA>::Get() { return cuda_stream_; }

cudaEvent_t& EventT<DEVICE_TYPE_NVIDIA>::Get() { return cuda_event_; }

template <>
void EventCreateT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA>* event) {
  CUDA_CHECK(cudaEventCreate(&(event->Get())));
}

template <>
void EventCreateWithFlagsT(EventT<DEVICE_TYPE_NVIDIA>* event, unsigned int flags) {
  CUDA_CHECK(cudaEventCreateWithFlags(&(event->Get()), flags));
}

template <>
void EventDestroyT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event) {
  CUDA_CHECK(cudaEventDestroy(event.Get()));
}

template <>
void StreamSynchronizeT<DEVICE_TYPE_NVIDIA>(StreamT<DEVICE_TYPE_NVIDIA> stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream.Get()));
}

template <>
void DeviceSynchronizeT<DEVICE_TYPE_NVIDIA>() {
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <>
void SetDeviceT<DEVICE_TYPE_NVIDIA>(int device_id) {
  CUDA_CHECK(cudaSetDevice(device_id));
}

template <>
void GetDeviceT<DEVICE_TYPE_NVIDIA>(int* device_id) {
  CUDA_CHECK(cudaGetDevice(device_id));
}

template <>
void GetDeviceCountT<DEVICE_TYPE_NVIDIA>(int* count) {
  CUDA_CHECK(cudaGetDeviceCount(count));
}

template <>
void EventRecordT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event, StreamT<DEVICE_TYPE_NVIDIA> stream) {
  CUDA_CHECK(cudaEventRecord(event.Get(), stream.Get()));
}

template <>
bool EventQueryT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event) {
  cudaError_t status = cudaEventQuery(event.Get());
  return (status == cudaSuccess);
}

template <>
void StreamWaitEventT<DEVICE_TYPE_NVIDIA>(StreamT<DEVICE_TYPE_NVIDIA> stream, EventT<DEVICE_TYPE_NVIDIA> event) {
  CUDA_CHECK(cudaStreamWaitEvent(stream.Get(), event.Get()));
}

template <>
void EventSynchronizeT<DEVICE_TYPE_NVIDIA>(EventT<DEVICE_TYPE_NVIDIA> event) {
  CUDA_CHECK(cudaEventSynchronize(event.Get()));
}

template <>
void EventElapsedTimeT<DEVICE_TYPE_NVIDIA>(float* ms, EventT<DEVICE_TYPE_NVIDIA> start,
                                           EventT<DEVICE_TYPE_NVIDIA> end) {
  CUDA_CHECK(cudaEventElapsedTime(ms, start.Get(), end.Get()));
}

template <>
void MemGetInfoT<DEVICE_TYPE_NVIDIA>(size_t* free, size_t* total) {
  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  cudaMemPool_t mempool;
  CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool, device_id));
  size_t mempool_used;
  size_t mempool_reserved;
  // cudaMemPoolAttrReservedMemCurrent is the current total physical GPU memory consumed by the pool.
  CUDA_CHECK(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrUsedMemCurrent, &mempool_used));
  // cudaMemPoolAttrUsedMemCurrent is the total size of all of the memory allocated from the pool
  // and not available for reuse.
  CUDA_CHECK(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &mempool_reserved));
  size_t mempool_free = (mempool_reserved - mempool_used);
  size_t device_free;
  CUDA_CHECK(cudaMemGetInfo(&device_free, total));
  *free = device_free + mempool_free;
}

template <>
void MallocT<DEVICE_TYPE_NVIDIA>(void** dev_ptr, size_t size) {
  CUDA_CHECK(cudaMalloc(dev_ptr, size));
}

template <>
void FreeT<DEVICE_TYPE_NVIDIA>(void* dev_ptr) {
  CUDA_CHECK(cudaFree(dev_ptr));
}

template <>
void HostAllocT<DEVICE_TYPE_NVIDIA>(void** host_ptr, size_t size) {
  CUDA_CHECK(cudaHostAlloc(host_ptr, size, cudaHostAllocDefault));
}

template <>
void FreeHostT<DEVICE_TYPE_NVIDIA>(void* host_ptr) {
  CUDA_CHECK(cudaFreeHost(host_ptr));
}

template <>
void MallocAsyncT<DEVICE_TYPE_NVIDIA>(void** dev_ptr, size_t size, StreamT<DEVICE_TYPE_NVIDIA> stream) {
  CUDA_CHECK(cudaMallocAsync(dev_ptr, size, stream.Get()));
}

template <>
void FreeAsyncT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, StreamT<DEVICE_TYPE_NVIDIA> stream) {
  CUDA_CHECK(cudaFreeAsync(dev_ptr, stream.Get()));
}

template <>
void MemsetAsyncT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, int value, size_t count, StreamT<DEVICE_TYPE_NVIDIA> stream) {
  if (count > 0) {
    CUDA_CHECK(cudaMemsetAsync(dev_ptr, value, count, stream.Get()));
  }
}

template <>
void MemsetT<DEVICE_TYPE_NVIDIA>(void* dev_ptr, int value, size_t count) {
  if (count > 0) {
    CUDA_CHECK(cudaMemset(dev_ptr, value, count));
  }
}

cudaMemcpyKind GetCudaMemcpyKind(enum MemcpyKind kind) {
  switch (kind) {
    case MEMCPY_HOST_TO_HOST:
      return cudaMemcpyHostToHost;
    case MEMCPY_HOST_TO_DEVICE:
      return cudaMemcpyHostToDevice;
    case MEMCPY_DEVICE_TO_HOST:
      return cudaMemcpyDeviceToHost;
    case MEMCPY_DEVICE_TO_DEVICE:
      return cudaMemcpyDeviceToDevice;
  }
  return cudaMemcpyHostToHost;
}

template <>
void MemcpyAsyncT<DEVICE_TYPE_NVIDIA>(void* dst, const void* src, size_t count, enum MemcpyKind kind,
                                      StreamT<DEVICE_TYPE_NVIDIA> stream) {
  if (count > 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count, GetCudaMemcpyKind(kind), stream.Get()));
  }
}

template <>
void Memcpy2DT<DEVICE_TYPE_NVIDIA>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                   size_t height, enum MemcpyKind kind) {
  CUDA_CHECK(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, GetCudaMemcpyKind(kind)));
}

template <>
void Memcpy2DAsyncT<DEVICE_TYPE_NVIDIA>(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                                        size_t height, enum MemcpyKind kind, StreamT<DEVICE_TYPE_NVIDIA> stream) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, GetCudaMemcpyKind(kind), stream.Get()));
}

template <>
void MemcpyT<DEVICE_TYPE_NVIDIA>(void* dst, const void* src, size_t count, enum MemcpyKind kind) {
  if (count > 0) {
    CUDA_CHECK(cudaMemcpyAsync(dst, src, count, GetCudaMemcpyKind(kind)));
  }
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::GetFloatType() {
  if (std::is_same<U, float>::value || std::is_same<U, const float>::value) {
    return TYPE_FP32;
  } else if (std::is_same<U, half>::value || std::is_same<U, const half>::value) {
    return TYPE_FP16;
#ifdef ENABLE_BFLOAT16
  } else if (std::is_same<U, __nv_bfloat16>::value || std::is_same<U, const __nv_bfloat16>::value) {
    return TYPE_BF16;
#endif
#ifdef ENABLE_FP8
  } else if (std::is_same<U, __nv_fp8_e4m3>::value || std::is_same<U, const __nv_fp8_e4m3>::value) {
    return TYPE_FP8_E4M3;
  } else if (std::is_same<U, __nv_fp8_e5m2>::value || std::is_same<U, const __nv_fp8_e5m2>::value) {
    return TYPE_FP8_E5M2;
#endif
  } else {
    return TYPE_INVALID;
  }
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::GetIntType() {
  if (std::is_same<U, int>::value || std::is_same<U, const int>::value) {
    return TYPE_INT32;
  } else if (std::is_same<U, int8_t>::value || std::is_same<U, const int8_t>::value) {
    return TYPE_INT8;
  } else if (std::is_same<U, uint8_t>::value || std::is_same<U, const uint8_t>::value) {
    return TYPE_UINT8;
  } else if (std::is_same<U, unsigned int>::value || std::is_same<U, const unsigned int>::value) {
    return TYPE_UINT32;
  } else if (std::is_same<U, uint64_t>::value || std::is_same<U, const uint64_t>::value) {
    return TYPE_UINT64;
  } else {
    return TYPE_INVALID;
  }
}

template <class U>
DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl() {
  // NOTE(karlluo): in order to fullfill Tencent's cyclomatic complexity rule,
  // I have to seperate function.
  DataType dtype = GetFloatType<U>();
  if (dtype != TYPE_INVALID) {
    return dtype;
  }

  dtype = GetIntType<U>();
  if (dtype != TYPE_INVALID) {
    return dtype;
  }

  if (std::is_same<U, bool>::value || std::is_same<U, const bool>::value) {
    return TYPE_BOOL;
  } else if (std::is_same<U, char>::value || std::is_same<U, const char>::value) {
    return TYPE_BYTES;
  } else if (std::is_pointer<U>::value) {
    return TYPE_POINTER;
  } else {
    return TYPE_INVALID;
  }
}

template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<float>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<half>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<int>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<int8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<uint8_t>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<unsigned int>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<uint64_t>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<bool>();
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<char>();

#ifdef ENABLE_BFLOAT16
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<__nv_bfloat16>();
#endif
#ifdef ENABLE_FP8
template DataType GetDataTypeT<DEVICE_TYPE_NVIDIA>::impl<__nv_fp8_e4m3>();
#endif

template <>
void* GetRuntimeContextT<DEVICE_TYPE_NVIDIA>(int device_id) {
  return nullptr;
}

}  // namespace ksana_llm
