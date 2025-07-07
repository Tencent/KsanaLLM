/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#ifdef ENABLE_CUDA
#  include "ksana_llm/utils/nvidia/nvidia_device.h"
#endif

#ifdef ENABLE_ACL
#  include "ksana_llm/utils/ascend/ascend_device.h"
#endif

#ifdef ENABLE_TOPS
#  include "ksana_llm/utils/zixiao/zixiao_device.h"
#endif

namespace ksana_llm {

// The stream for different device type.
using Stream = StreamT<ACTIVE_DEVICE_TYPE>;

// The event for different device type.
using Event = EventT<ACTIVE_DEVICE_TYPE>;

// The event create & destroy.
constexpr auto EventCreate = EventCreateT<ACTIVE_DEVICE_TYPE>;
constexpr auto EventDestroy = EventDestroyT<ACTIVE_DEVICE_TYPE>;

// The event create flag
#define EVENT_DEFAULT 0x00
#define EVENT_DISABLE_TIMING 0x02
constexpr auto EventCreateWithFlags = EventCreateWithFlagsT<ACTIVE_DEVICE_TYPE>;

// The synchronize function.
constexpr auto StreamSynchronize = StreamSynchronizeT<ACTIVE_DEVICE_TYPE>;
constexpr auto DeviceSynchronize = DeviceSynchronizeT<ACTIVE_DEVICE_TYPE>;

// The set/get device function.
constexpr auto SetDevice = SetDeviceT<ACTIVE_DEVICE_TYPE>;
constexpr auto GetDevice = GetDeviceT<ACTIVE_DEVICE_TYPE>;

constexpr auto GetDeviceCount = GetDeviceCountT<ACTIVE_DEVICE_TYPE>;

// The event function.
constexpr auto EventRecord = EventRecordT<ACTIVE_DEVICE_TYPE>;
constexpr auto EventQuery = EventQueryT<ACTIVE_DEVICE_TYPE>;
constexpr auto StreamWaitEvent = StreamWaitEventT<ACTIVE_DEVICE_TYPE>;

constexpr auto EventSynchronize = EventSynchronizeT<ACTIVE_DEVICE_TYPE>;
constexpr auto EventElapsedTime = EventElapsedTimeT<ACTIVE_DEVICE_TYPE>;

constexpr auto MemGetInfo = MemGetInfoT<ACTIVE_DEVICE_TYPE>;

// The malloc/free function.
constexpr auto Malloc = MallocT<ACTIVE_DEVICE_TYPE>;
constexpr auto Free = FreeT<ACTIVE_DEVICE_TYPE>;

// The alloc/free host memory.
constexpr auto HostAlloc = HostAllocT<ACTIVE_DEVICE_TYPE>;
constexpr auto HostFree = FreeHostT<ACTIVE_DEVICE_TYPE>;

// The alloc/free host memory with mapped device pointer.
constexpr auto HostAllocMapped = HostAllocMappedT<ACTIVE_DEVICE_TYPE>;
constexpr auto FreeHostMapped = FreeHostMappedT<ACTIVE_DEVICE_TYPE>;

// The malloc/free device memory.
constexpr auto MallocAsync = MallocAsyncT<ACTIVE_DEVICE_TYPE>;
constexpr auto FreeAsync = FreeAsyncT<ACTIVE_DEVICE_TYPE>;

// The memset/memcpy function.
constexpr auto MemsetAsync = MemsetAsyncT<ACTIVE_DEVICE_TYPE>;
constexpr auto MemcpyAsync = MemcpyAsyncT<ACTIVE_DEVICE_TYPE>;

constexpr auto Memset = MemsetT<ACTIVE_DEVICE_TYPE>;
constexpr auto Memcpy = MemcpyT<ACTIVE_DEVICE_TYPE>;

// The memcpy function.
constexpr auto Memcpy2D = Memcpy2DT<ACTIVE_DEVICE_TYPE>;
constexpr auto Memcpy2DAsync = Memcpy2DAsyncT<ACTIVE_DEVICE_TYPE>;

template <class U>
constexpr auto GetDataType = GetDataTypeT<ACTIVE_DEVICE_TYPE>::impl<U>;

constexpr auto GetRuntimeContext = GetRuntimeContextT<ACTIVE_DEVICE_TYPE>;

}  // namespace ksana_llm
