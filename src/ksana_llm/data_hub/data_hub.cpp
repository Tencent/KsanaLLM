/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/data_hub/data_hub.h"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_message.h"
#include "ksana_llm/distributed/packet_type.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/critical_zone.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// object pool of schedule output buffer.
ScheduleOutputPool* g_schedule_output_pool = nullptr;

// object pool of hidden unit buffer.
HiddenUnitBufferPool* g_hidden_unit_buffer_pool = nullptr;

// The current device hidden unit buffer.
std::unordered_map<size_t, HiddenUnitDeviceBuffer*> g_hidden_unit_buffer_map;
// Mutex to protect g_hidden_unit_buffer_map
std::mutex g_hidden_unit_buffer_map_mutex;

std::unordered_map<std::string, std::shared_ptr<ModelInstance>> g_model_instances;

std::vector<std::shared_ptr<CacheManagerInterface>> g_cache_managers;

// hidden unit meta.
DataType g_hidden_unit_data_type;
std::vector<size_t> g_hidden_unit_shape;

// Map from schedule_id to waiter
std::unordered_map<size_t, std::shared_ptr<Waiter>> g_waiters;
std::mutex g_waiters_mutex;

void InitializeScheduleOutputPool() { g_schedule_output_pool = new ScheduleOutputPool(); }

void InitializeHiddenUnitBufferPool() { g_hidden_unit_buffer_pool = new HiddenUnitBufferPool(); }

void DestroyScheduleOutputPool() {
  if (g_schedule_output_pool) {
    delete g_schedule_output_pool;
    g_schedule_output_pool = nullptr;
  }
}

void DestroyHiddenUnitBufferPool() {
  if (g_hidden_unit_buffer_pool) {
    delete g_hidden_unit_buffer_pool;
    g_hidden_unit_buffer_pool = nullptr;
  }
}

void SetCurrentHiddenUnitBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  if (hidden_unit_buffer != nullptr) {
    auto schedule_id = hidden_unit_buffer->schedule_id;
    {
      std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
      KLLM_CHECK_WITH_INFO(g_hidden_unit_buffer_map.find(schedule_id) == g_hidden_unit_buffer_map.end(),
                           FormatStr("schedule_id=%d exists.", schedule_id));
      KLLM_LOG_DEBUG << "set schedule_id=" << schedule_id;
      g_hidden_unit_buffer_map[schedule_id] = hidden_unit_buffer;
    }
  }
}

HiddenUnitDeviceBuffer* GetCurrentHiddenUnitBuffer(size_t schedule_id) {
  std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
  auto it = g_hidden_unit_buffer_map.find(schedule_id);
  if (it != g_hidden_unit_buffer_map.end()) {
    return it->second;
  }
  KLLM_LOG_DEBUG << "Return nullptr. schedule_id=" << schedule_id;
  return nullptr;
}

Status CopyFromHiddenUnitBuffer(Tensor& tensor, HiddenUnitDeviceBuffer* device_buffer, int rank, bool is_prefill) {
#ifdef ENABLE_ACL
  if (is_prefill) {
    tensor.shape = device_buffer->prefill_tensors[rank].shape;
    tensor.dtype = device_buffer->prefill_tensors[rank].dtype;
    return Status();
  }
#endif

  tensor.shape = device_buffer->tensors[rank].shape;
  tensor.dtype = device_buffer->tensors[rank].dtype;

  return Status();
}

Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, Tensor& tensor, int rank, bool is_prefill) {
#ifdef ENABLE_ACL
  if (is_prefill) {
    device_buffer->prefill_tensors[rank].shape = tensor.shape;
    device_buffer->prefill_tensors[rank].dtype = tensor.dtype;

    Memcpy(device_buffer->prefill_tensors[rank].template GetPtr<void>(), tensor.GetPtr<void>(), tensor.GetTotalBytes(),
           MEMCPY_DEVICE_TO_DEVICE);
    device_buffer->prefill_enabled = true;
    return Status();
  }
#endif

  device_buffer->tensors[rank].shape = tensor.shape;
  device_buffer->tensors[rank].dtype = tensor.dtype;

  Memcpy(device_buffer->tensors[rank].template GetPtr<void>(), tensor.GetPtr<void>(), tensor.GetTotalBytes(),
         MEMCPY_DEVICE_TO_DEVICE);
#ifdef ENABLE_ACL
  device_buffer->decode_enabled = true;
#endif

  return Status();
}

// Called by every gpu worker.
Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, void* device_ptr, std::vector<size_t> shape,
                              DataType dtype, int rank, bool is_prefill) {
  size_t total_bytes = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
                       GetTypeSize(dtype);
#ifdef ENABLE_ACL
  if (is_prefill) {
    device_buffer->prefill_tensors[rank].shape = shape;
    device_buffer->prefill_tensors[rank].dtype = dtype;

    Memcpy(device_buffer->prefill_tensors[rank].template GetPtr<void>(), device_ptr, total_bytes,
           MEMCPY_DEVICE_TO_DEVICE);
    device_buffer->prefill_enabled = true;

    return Status();
  }
#endif

  device_buffer->tensors[rank].shape = shape;
  device_buffer->tensors[rank].dtype = dtype;

  Memcpy(device_buffer->tensors[rank].template GetPtr<void>(), device_ptr, total_bytes, MEMCPY_DEVICE_TO_DEVICE);
#ifdef ENABLE_ACL
  device_buffer->decode_enabled = true;
#endif

  return Status();
}

// Called by every gpu worker.
Status CopyHostMemToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, void* host_ptr, std::vector<size_t> shape,
                                     DataType dtype, int rank, bool is_prefill) {
  size_t total_bytes = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
                       GetTypeSize(dtype);
#ifdef ENABLE_ACL
  if (is_prefill) {
    device_buffer->prefill_tensors[rank].shape = shape;
    device_buffer->prefill_tensors[rank].dtype = dtype;

    Memcpy(device_buffer->prefill_tensors[rank].template GetPtr<void>(), host_ptr, total_bytes,
           MEMCPY_DEVICE_TO_DEVICE);
    device_buffer->prefill_enabled = true;
    return Status();
  }
#endif

  device_buffer->tensors[rank].shape = shape;
  device_buffer->tensors[rank].dtype = dtype;

  Memcpy(device_buffer->tensors[rank].template GetPtr<void>(), host_ptr, total_bytes, MEMCPY_HOST_TO_DEVICE);
#ifdef ENABLE_ACL
  device_buffer->decode_enabled = true;
#endif

  return Status();
}

Status CopyHiddenUnitBufferToHostMem(void* host_ptr, HiddenUnitDeviceBuffer* device_buffer, std::vector<size_t> shape,
                                     DataType dtype, int rank, bool is_prefill) {
  size_t total_bytes = std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>()) *
                       GetTypeSize(dtype);
#ifdef ENABLE_ACL
  if (is_prefill) {
    device_buffer->prefill_tensors[rank].shape = shape;
    device_buffer->prefill_tensors[rank].dtype = dtype;

    Memcpy(host_ptr, device_buffer->prefill_tensors[rank].template GetPtr<void>(), total_bytes,
           MEMCPY_DEVICE_TO_DEVICE);
    return Status();
  }
#endif

  SetDevice(rank);
  device_buffer->tensors[rank].shape = shape;
  device_buffer->tensors[rank].dtype = dtype;

  Memcpy(host_ptr, device_buffer->tensors[rank].template GetPtr<void>(), total_bytes, MEMCPY_DEVICE_TO_HOST);

  return Status();
}

ScheduleOutputPool* GetScheduleOutputPool() { return g_schedule_output_pool; }

HiddenUnitBufferPool* GetHiddenUnitBufferPool() { return g_hidden_unit_buffer_pool; }

Status BroadcastScheduleOutput(ScheduleOutput* schedule_output) {
  GetScheduleOutputPool()->PutToSendQueue(schedule_output);
  return Status();
}

Status InitHiddenUnits(size_t schedule_id) {
  KLLM_LOG_DEBUG << "Enter schedule_id=" << schedule_id;
  HiddenUnitDeviceBuffer* hidden_unit_buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  if (!hidden_unit_buffer) {
    return Status(RET_RUNTIME_FAILED, "GetDeviceBuffer error, empty result.");
  }

  // Set the schedule_id for the buffer
  hidden_unit_buffer->schedule_id = schedule_id;
  SetCurrentHiddenUnitBuffer(hidden_unit_buffer);
  return Status();
}

Status SendHiddenUnits(size_t schedule_id) {
  HiddenUnitDeviceBuffer* hidden_unit_buffer = GetCurrentHiddenUnitBuffer(schedule_id);
  if (!hidden_unit_buffer) {
    return Status(RET_RUNTIME_FAILED,
                  "GetCurrentHiddenUnitBuffer error, empty result for schedule_id=" + std::to_string(schedule_id));
  }
  KLLM_LOG_DEBUG << "PutToSendQueue. schedule_id=" << hidden_unit_buffer->schedule_id
                 << ", hidden_unit_buffer=" << hidden_unit_buffer;
  GetHiddenUnitBufferPool()->PutToSendQueue(hidden_unit_buffer);
  return Status();
}

Status ResetReceiveWaiter() {
  ProfileEvent::PushEvent("ResetReceiveWaiter");
  std::lock_guard<std::mutex> lock(g_waiters_mutex);
  // Reset all waiters
  for (auto& pair : g_waiters) {
    pair.second->Reset(1);
  }
  ProfileEvent::PopEvent();
  return Status();
}

Status RecvHiddenUnits(bool do_recv, size_t schedule_id) {
  // All the model inference will call this method.
  // But only the thread that have true do_recv actually do the receiving operation.
  // Other threads are blocked until receiving operation finished, then start to do computation.

  // Get or create waiter for this schedule_id
  std::shared_ptr<Waiter> waiter;
  {
    std::lock_guard<std::mutex> lock(g_waiters_mutex);
    auto it = g_waiters.find(schedule_id);
    if (it == g_waiters.end()) {
      waiter = std::make_shared<Waiter>(1);
      g_waiters[schedule_id] = waiter;
    } else {
      waiter = it->second;
    }
  }

  if (do_recv) {
    ProfileEvent::PushEvent("RecvHiddenUnits_dorecv");
    LeaveDeviceComputingCriticalZone();
    KLLM_LOG_DEBUG << "LeaveDeviceComputingCriticalZone. schedule_id=" << schedule_id;
    // Free old hidden units.
    FreeHiddenUnits(schedule_id);
    ProfileEvent::PushEvent("GetFromDeviceRecvQueue");
    HiddenUnitDeviceBuffer* hidden_unit_buffer = GetHiddenUnitBufferPool()->GetFromDeviceRecvQueue(schedule_id);
    ProfileEvent::PopEvent();

    // TODO(robertyuan): There is a risk that others schedule id enter first. Replace with condition_variable later.
    KLLM_LOG_DEBUG << "try EnterDeviceComputingCriticalZone schedule_id=" << schedule_id
                   << ", hidden_unit_buffer=" << hidden_unit_buffer;
    ProfileEvent::PushEvent("WaitToEnterCompute");
    EnterDeviceComputingCriticalZone();
    ProfileEvent::PopEvent();

    KLLM_LOG_DEBUG << "EnterDeviceComputingCriticalZone schedule_id=" << schedule_id;
    if (!hidden_unit_buffer) {
      KLLM_LOG_ERROR << "GetFromDeviceRecvQueue error, empty result. schedule_id=" << schedule_id;
      return Status(RET_RUNTIME_FAILED, "GetFromDeviceRecvQueue error, empty result.");
    }

    // Set the schedule_id for the buffer
    assert(hidden_unit_buffer->schedule_id == schedule_id);
    SetCurrentHiddenUnitBuffer(hidden_unit_buffer);
    waiter->Notify();
    ProfileEvent::PopEvent();
    return Status();
  }

  waiter->Wait();

  return Status();
}

Status FreeHiddenUnits(size_t schedule_id) {
  ProfileEvent::PushEvent("FreeHiddenUnits");
  HiddenUnitDeviceBuffer* hidden_unit_buffer;
  {
    std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
    auto it = g_hidden_unit_buffer_map.find(schedule_id);
    if (it == g_hidden_unit_buffer_map.end()) {
      KLLM_CHECK_WITH_INFO(false, FormatStr("FreeHiddenUnits schedule_id=%d not exists.", schedule_id));
      return Status(RET_RUNTIME_FAILED,
                    "GetCurrentHiddenUnitBuffer error, empty result for schedule_id=" + std::to_string(schedule_id));
    }
    hidden_unit_buffer = it->second;
  }

  KLLM_LOG_DEBUG << "FreeHiddenUnits schedule_id=" << schedule_id << ", hidden_unit_buffer=" << hidden_unit_buffer;
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);

  {
    std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
    g_hidden_unit_buffer_map.erase(schedule_id);
  }
  ProfileEvent::PopEvent();
  return Status();
}

Status GetHiddenUnitMeta(std::vector<size_t>& shape, DataType& data_type) {
  shape = g_hidden_unit_shape;
  data_type = g_hidden_unit_data_type;
  return Status();
}

Status SetHiddenUnitMeta(const std::vector<size_t>& shape, DataType data_type) {
  g_hidden_unit_shape = shape;
  g_hidden_unit_data_type = data_type;
  return Status();
}

Status SetModelInstance(const std::string model_name, std::shared_ptr<ModelInstance> model_instance) {
  g_model_instances[model_name] = model_instance;
  return Status();
}

std::shared_ptr<ModelInstance> GetModelInstance(const std::string& model_name) {
  if (g_model_instances.find(model_name) == g_model_instances.end()) {
    return nullptr;
  }
  return g_model_instances[model_name];
}

void DestroyModelInstance() { g_model_instances.clear(); }

Status SetCacheManagers(const std::vector<std::shared_ptr<CacheManagerInterface>>& cache_managers) {
  g_cache_managers = cache_managers;
  return Status();
}

std::shared_ptr<CacheManagerInterface> GetCacheManager(int group_id) {
  if (group_id >= g_cache_managers.size()) {
    return nullptr;
  }
  return g_cache_managers[group_id];
}

void DestroyCacheManager() { g_cache_managers.clear(); }

}  // namespace ksana_llm
