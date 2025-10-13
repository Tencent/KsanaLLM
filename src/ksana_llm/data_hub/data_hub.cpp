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

// hidden unit meta. seems only used in worker, so maybe do not need lock
std::unordered_map<size_t, DataType> g_hidden_unit_data_type;
std::unordered_map<size_t, std::vector<size_t>> g_hidden_unit_shape;
std::mutex g_hidden_meta_mutex;

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
    auto multi_batch_id = hidden_unit_buffer->multi_batch_id;
    {
      std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
      KLLM_CHECK_WITH_INFO(g_hidden_unit_buffer_map.find(multi_batch_id) == g_hidden_unit_buffer_map.end(),
                           FormatStr("multi_batch_id=%d exists.", multi_batch_id));
      KLLM_LOG_DEBUG << "set multi_batch_id=" << multi_batch_id;
      g_hidden_unit_buffer_map[multi_batch_id] = hidden_unit_buffer;
    }
  }
}

HiddenUnitDeviceBuffer* GetCurrentHiddenUnitBuffer(size_t multi_batch_id) {
  std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
  auto it = g_hidden_unit_buffer_map.find(multi_batch_id);
  if (it != g_hidden_unit_buffer_map.end()) {
    return it->second;
  } else {
    KLLM_LOG_ERROR << "HiddenUnitBuffer multi_batch_id=" << multi_batch_id << " not found.";
    return nullptr;
  }
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

Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, Tensor& tensor, int rank, bool is_prefill,
                              Stream working_stream) {
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

  MemcpyAsync(device_buffer->tensors[rank].template GetPtr<void>(), tensor.GetPtr<void>(), tensor.GetTotalBytes(),
              MEMCPY_DEVICE_TO_DEVICE, working_stream);
  StreamSynchronize(working_stream);

#ifdef ENABLE_ACL
  device_buffer->decode_enabled = true;
#endif

  return Status();
}

// Called by every gpu worker.
Status CopyToHiddenUnitBuffer(HiddenUnitDeviceBuffer* device_buffer, void* device_ptr, std::vector<size_t> shape,
                              DataType dtype, int rank, bool is_prefill, Stream working_stream) {
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

  MemcpyAsync(device_buffer->tensors[rank].template GetPtr<void>(), device_ptr, total_bytes, MEMCPY_DEVICE_TO_DEVICE,
              working_stream);
  StreamSynchronize(working_stream);
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

Status InitHiddenUnits(size_t multi_batch_id) {
  PROFILE_EVENT_SCOPE(InitHiddenUnits, fmt::format("InitHiddenUnits_{}", multi_batch_id));
  HiddenUnitDeviceBuffer* hidden_unit_buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
  if (!hidden_unit_buffer) {
    return Status(RET_RUNTIME_FAILED, "GetDeviceBuffer error, empty result.");
  }

  // Set the multi_batch_id for the buffer
  hidden_unit_buffer->multi_batch_id = multi_batch_id;
  SetCurrentHiddenUnitBuffer(hidden_unit_buffer);
  return Status();
}

Status SendHiddenUnits(size_t multi_batch_id) {
  HiddenUnitDeviceBuffer* hidden_unit_buffer = GetCurrentHiddenUnitBuffer(multi_batch_id);
  if (!hidden_unit_buffer) {
    return Status(RET_RUNTIME_FAILED, "GetCurrentHiddenUnitBuffer error, empty result for multi_batch_id=" +
                                          std::to_string(multi_batch_id));
  }
  KLLM_LOG_DEBUG << "PutToPendingSendQueue. multi_batch_id=" << hidden_unit_buffer->multi_batch_id;
  GetHiddenUnitBufferPool()->PutToPendingSendQueue(hidden_unit_buffer);
  return Status();
}

Status RecvHiddenUnits(size_t multi_batch_id) {
  FreeHiddenUnits(multi_batch_id);
  HiddenUnitDeviceBuffer* recved_hidden_unit = nullptr;

  {
    PROFILE_EVENT_SCOPE(RecvHiddenUnits_, fmt::format("RecvHiddenUnits_{}", multi_batch_id));
    time_t start_time_ms = ProfileTimer::GetCurrentTimeInMs();
    HiddenUnitDeviceBuffer* hidden_unit_buffer = GetHiddenUnitBufferPool()->GetDeviceBuffer();
    hidden_unit_buffer->multi_batch_id = multi_batch_id;
    KLLM_LOG_DEBUG << "start to recv multi_batch_id=" << hidden_unit_buffer->multi_batch_id;
    GetHiddenUnitBufferPool()->PutToPendingRecvQueue(hidden_unit_buffer);
    recved_hidden_unit = GetHiddenUnitBufferPool()->GetFromDeviceRecvedQueue(multi_batch_id);
    time_t end_time_ms = ProfileTimer::GetCurrentTimeInMs();
    KLLM_LOG_DEBUG << "recved multi_batch_id: " << multi_batch_id << ", cost " << end_time_ms - start_time_ms << "ms";
  }

  KLLM_CHECK_WITH_INFO(recved_hidden_unit->multi_batch_id == multi_batch_id, "recved multi_batch_id not match");
  SetCurrentHiddenUnitBuffer(recved_hidden_unit);
  return Status();
}

Status FreeHiddenUnits(size_t multi_batch_id) {
  PROFILE_EVENT_SCOPE(FreeHiddenUnits, "FreeHiddenUnits");
  HiddenUnitDeviceBuffer* hidden_unit_buffer;
  {
    std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
    auto it = g_hidden_unit_buffer_map.find(multi_batch_id);
    if (it == g_hidden_unit_buffer_map.end()) {
      KLLM_CHECK_WITH_INFO(false, FormatStr("FreeHiddenUnits multi_batch_id=%d not exists.", multi_batch_id));
      return Status(RET_RUNTIME_FAILED, "GetCurrentHiddenUnitBuffer error, empty result for multi_batch_id=" +
                                            std::to_string(multi_batch_id));
    }
    hidden_unit_buffer = it->second;
  }

  KLLM_LOG_DEBUG << "FreeHiddenUnits multi_batch_id=" << multi_batch_id
                 << ", hidden_unit_buffer=" << hidden_unit_buffer;
  GetHiddenUnitBufferPool()->FreeDeviceBuffer(hidden_unit_buffer);

  {
    std::lock_guard<std::mutex> lock(g_hidden_unit_buffer_map_mutex);
    g_hidden_unit_buffer_map.erase(multi_batch_id);
  }
  return Status();
}

void InitHiddenUnitsMetaInfoMap(int max_pp_batch_num) {
  std::unique_lock<std::mutex> lock(g_hidden_meta_mutex);
  for (int i = 0; i < max_pp_batch_num; ++i) {
    g_hidden_unit_shape[i] = {};
    g_hidden_unit_data_type[i] = DataType::TYPE_INVALID;
  }
}

Status GetHiddenUnitMeta(const size_t multi_batch_id, std::vector<size_t>& shape, DataType& data_type) {
  std::unique_lock<std::mutex> lock(g_hidden_meta_mutex);
  shape = g_hidden_unit_shape.at(multi_batch_id);
  data_type = g_hidden_unit_data_type.at(multi_batch_id);
  KLLM_LOG_DEBUG << "get multi_batch_id=" << multi_batch_id << ", data_type=" << data_type
                 << ", shape:" << Vector2Str(shape);
  return Status();
}

Status SetHiddenUnitMeta(const size_t multi_batch_id, const std::vector<size_t>& shape, DataType data_type) {
  std::unique_lock<std::mutex> lock(g_hidden_meta_mutex);
  KLLM_LOG_DEBUG << "set multi_batch_id=" << multi_batch_id << ", data_type=" << data_type
                 << ", shape:" << Vector2Str(shape);
  g_hidden_unit_shape[multi_batch_id] = shape;
  g_hidden_unit_data_type[multi_batch_id] = data_type;
  return Status();
}

Status SetHiddenUnitMeta(size_t multi_batch_id, ScheduleOutput* schedule_output,
                         std::shared_ptr<ModelInstance> model_instance) {
  ModelConfig model_config = model_instance->GetModelConfig();
  size_t tokens = schedule_output->hidden_token_num;  // 直接使用已计算的token数量
  return SetHiddenUnitMeta(multi_batch_id, {tokens, model_config.hidden_units}, model_config.weight_data_type);
}

Status SetHiddenUnitMeta(size_t multi_batch_id,
                         const std::vector<std::shared_ptr<WorkerInferRequest>>& worker_running_reqs,
                         std::shared_ptr<ModelInstance> model_instance) {
  ModelConfig model_config = model_instance->GetModelConfig();
  size_t tokens = 0;
  for (size_t i = 0; i < worker_running_reqs.size(); ++i) {
    tokens += worker_running_reqs[i]->forwarding_tokens.size() - worker_running_reqs[i]->kv_cached_token_num;
  }
  return SetHiddenUnitMeta(multi_batch_id, {tokens, model_config.hidden_units}, model_config.weight_data_type);
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
