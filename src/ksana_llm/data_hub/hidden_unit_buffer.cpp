/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/data_hub/hidden_unit_buffer.h"

#include "ksana_llm/profiler/profile_event.h"
#include "ksana_llm/utils/device_types.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

HiddenUnitDeviceBuffer::~HiddenUnitDeviceBuffer() {
  if (waiter) {
    waiter->Stop();
    waiter = nullptr;
  }
}
void HiddenUnitDeviceBuffer::NotifyFinished() {
  if (waiter) {
    waiter->Notify();
  }
}

HiddenUnitBufferPool::HiddenUnitBufferPool() {
  InitializeBufferSize();
}

void HiddenUnitBufferPool::PreAllocateDeviceBuffer() {
  HiddenUnitDeviceBuffer* dev_hidden_unit = new HiddenUnitDeviceBuffer();
  InitializeHiddenUnitDeviceBuffer(dev_hidden_unit);
  free_device_buffers_.Put(dev_hidden_unit);
}

Status HiddenUnitBufferPool::ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev,
                                                       HiddenUnitHostBuffer* hidden_unit_host) {
  hidden_unit_dev->multi_batch_id = hidden_unit_host->multi_batch_id;

  size_t buffer_bytes = hidden_unit_host->shape_dims[0] * hidden_unit_host->shape_dims[1] * GetTypeSize(weight_type_);
  std::vector<size_t> buffer_shape = {hidden_unit_host->shape_dims[0], hidden_unit_host->shape_dims[1]};

#ifdef ENABLE_ACL
  size_t prefill_buffer_bytes =
      hidden_unit_host->prefill_shape_dims[0] * hidden_unit_host->prefill_shape_dims[1] * GetTypeSize(weight_type_);
  std::vector<size_t> prefill_buffer_shape = {hidden_unit_host->prefill_shape_dims[0],
                                              hidden_unit_host->prefill_shape_dims[1]};
  hidden_unit_dev->decode_enabled = buffer_bytes > 0;
  hidden_unit_dev->prefill_enabled = prefill_buffer_bytes > 0;
#endif

  for (size_t i = 0; i < hidden_unit_host->tensor_parallel; ++i) {
    SetDevice(i);
    if (buffer_bytes > 0) {
      Memcpy(hidden_unit_dev->tensors[i].GetPtr<void>(), hidden_unit_host->data, buffer_bytes, MEMCPY_HOST_TO_DEVICE);
      hidden_unit_dev->tensors[i].shape = buffer_shape;
    }
#ifdef ENABLE_ACL
    if (prefill_buffer_bytes > 0) {
      Memcpy(hidden_unit_dev->prefill_tensors[i].GetPtr<void>(), hidden_unit_host->data + buffer_bytes,
             prefill_buffer_bytes, MEMCPY_HOST_TO_DEVICE);
      hidden_unit_dev->prefill_tensors[i].shape = prefill_buffer_shape;
    }
#endif
  }

  return Status();
}

Status HiddenUnitBufferPool::ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host,
                                                       HiddenUnitDeviceBuffer* hidden_unit_dev) {
  hidden_unit_host->multi_batch_id = hidden_unit_dev->multi_batch_id;
  hidden_unit_host->tensor_parallel = hidden_unit_dev->tensors.size();

  std::vector<size_t> buffer_shape = hidden_unit_dev->tensors[0].shape;
  hidden_unit_host->shape_dims[0] = buffer_shape[0];
  hidden_unit_host->shape_dims[1] = buffer_shape[1];
#ifdef ENABLE_ACL
  if (!hidden_unit_dev->decode_enabled) {
    hidden_unit_host->shape_dims[0] = 0;
    hidden_unit_host->shape_dims[1] = 0;
  }
  std::vector<size_t> prefill_buffer_shape = hidden_unit_dev->prefill_tensors[0].shape;
  hidden_unit_host->prefill_shape_dims[0] = prefill_buffer_shape[0];
  hidden_unit_host->prefill_shape_dims[1] = prefill_buffer_shape[1];

  if (!hidden_unit_dev->prefill_enabled) {
    hidden_unit_host->prefill_shape_dims[0] = 0;
    hidden_unit_host->prefill_shape_dims[1] = 0;
  }
#endif

  size_t buffer_bytes = hidden_unit_host->shape_dims[0] * hidden_unit_host->shape_dims[1] * GetTypeSize(weight_type_);
#ifdef ENABLE_ACL
  size_t prefill_buffer_bytes =
      hidden_unit_host->prefill_shape_dims[0] * hidden_unit_host->prefill_shape_dims[1] * GetTypeSize(weight_type_);
#endif
  for (size_t i = 0; i < hidden_unit_dev->tensors.size(); ++i) {
    SetDevice(i);
    if (buffer_bytes > 0) {
      Memcpy(hidden_unit_host->data, hidden_unit_dev->tensors[i].GetPtr<void>(), buffer_bytes, MEMCPY_DEVICE_TO_HOST);
    }
#ifdef ENABLE_ACL
    if (prefill_buffer_bytes > 0) {
      Memcpy(hidden_unit_host->data + buffer_bytes, hidden_unit_dev->prefill_tensors[i].GetPtr<void>(),
             prefill_buffer_bytes, MEMCPY_DEVICE_TO_HOST);
    }
#endif
    break;
  }

  return Status();
}

size_t HiddenUnitBufferPool::GetHostPacketSize(Packet* packet) {
  HiddenUnitHostBuffer* hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  size_t host_packet_size = sizeof(HiddenUnitHostBuffer) +
                            (hidden_unit->shape_dims[0] * hidden_unit->shape_dims[1] * GetTypeSize(weight_type_));
#ifdef ENABLE_ACL
  host_packet_size +=
      (hidden_unit->prefill_shape_dims[0] * hidden_unit->prefill_shape_dims[1] * GetTypeSize(weight_type_));
#endif
  return host_packet_size;
}

void HiddenUnitBufferPool::InitializeBufferSize() {
  std::unordered_map<std::string, ModelConfig> model_configs;
  Singleton<Environment>::GetInstance()->GetModelConfigs(model_configs);
  PipelineConfig pipeline_config;
  Singleton<Environment>::GetInstance()->GetPipelineConfig(pipeline_config);
  comm_type_ = pipeline_config.pipeline_para_comm_type;

  // Skip if no model config.
  if (model_configs.empty()) {
    KLLM_LOG_ERROR << "No model_config provided.";
    return;
  }

  ModelConfig model_config = model_configs.begin()->second;

  weight_type_ = model_config.weight_data_type;
  tensor_para_size_ = model_config.tensor_para_size;
  max_token_num_ = model_config.max_step_token_num;
  hidden_unit_size_ = model_config.size_per_head * model_config.head_num;

  KLLM_LOG_INFO << "HiddenUnitBufferPool::InitializeBufferSize weight_type:" << weight_type_
                << ", tensor_para_size:" << tensor_para_size_ << ", max_token_num:" << max_token_num_
                << ", hidden_unit_size:" << hidden_unit_size_;
}

Status HiddenUnitBufferPool::InitializeHiddenUnitDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  hidden_unit_buffer->tensors.resize(tensor_para_size_);
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_tensors.resize(tensor_para_size_);
#endif
  for (int rank = 0; rank < tensor_para_size_; ++rank) {
    SetDevice(rank);
    hidden_unit_buffer->tensors[rank] =
        Tensor(MemoryLocation::LOCATION_DEVICE, weight_type_, {max_token_num_, hidden_unit_size_}, rank);
#ifdef ENABLE_ACL
    hidden_unit_buffer->prefill_tensors[rank] =
        Tensor(MemoryLocation::LOCATION_DEVICE, weight_type_, {max_token_num_, hidden_unit_size_}, rank);
#endif
  }

#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  hidden_unit_buffer->comm_type = comm_type_;
  hidden_unit_buffer->waiter = std::make_shared<Waiter>(1);

  KLLM_LOG_DEBUG << "HiddenUnitBufferPool::InitializeHiddenUnitDeviceBuffe, shape:"
                 << Vector2Str(std::vector<size_t>(hidden_unit_buffer->tensors[0].shape))
                 << ", max_token_num:" << max_token_num_ << ", hidden_unit_size:" << hidden_unit_size_
                 << ", Pipeline Parallel Communication Type: " << static_cast<int32_t>(comm_type_);

  return Status();
}

HiddenUnitDeviceBuffer* HiddenUnitBufferPool::GetDeviceBuffer() {
  if (free_device_buffers_.Empty()) {
    KLLM_LOG_INFO << "HiddenUnitBufferPool Create device buffer, should called only once.";
    HiddenUnitDeviceBuffer* hidden_unit = new HiddenUnitDeviceBuffer();
    InitializeHiddenUnitDeviceBuffer(hidden_unit);
    return hidden_unit;
  }

  return free_device_buffers_.Get();
}

// Free the hidden unit buffer to object pool.
Status HiddenUnitBufferPool::FreeDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  free_device_buffers_.Put(hidden_unit_buffer);
  return Status();
}

Packet* HiddenUnitBufferPool::GetHostBuffer() {
  if (free_host_buffers_.Empty()) {
    size_t extra_size = max_token_num_ * hidden_unit_size_ * GetTypeSize(weight_type_);
    size_t packet_body_size = sizeof(HiddenUnitHostBuffer) + extra_size;
    Packet* packet = reinterpret_cast<Packet*>(malloc(sizeof(Packet) + packet_body_size));
    if (packet == nullptr) {
      KLLM_LOG_ERROR << "GetHostBuffer error, allocate memory failed.";
      return packet;
    }

    packet->type = PacketType::DATA_REQ_HIDDEN_UNIT;
    packet->size = packet_body_size;

    HiddenUnitHostBuffer* hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);

    hidden_unit->shape_dims[0] = max_token_num_;
    hidden_unit->shape_dims[1] = hidden_unit_size_;
    hidden_unit->tensor_parallel = tensor_para_size_;

    return packet;
  }

  return free_host_buffers_.Get();
}

Status HiddenUnitBufferPool::FreeHostBuffer(Packet* hidden_unit_buffer) {
  free_host_buffers_.Put(hidden_unit_buffer);
  return Status();
}

Status HiddenUnitBufferPool::PutToHostRecvQueue(Packet* packet) {
  recv_host_buffers_.Put(packet);
  return Status();
}

Packet* HiddenUnitBufferPool::GetFromHostRecvQueue() { return recv_host_buffers_.Get(); }

Status HiddenUnitBufferPool::PutToPendingSendQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  auto send_done_waiter = hidden_unit->waiter;
  pending_send_device_buffers_.Put(hidden_unit);
  send_done_waiter->Wait();
  send_done_waiter->Reset(1);
  return Status();
}

HiddenUnitDeviceBuffer* HiddenUnitBufferPool::GetFromPendingSendQueue() { return pending_send_device_buffers_.Get(); }

Status HiddenUnitBufferPool::PutToPendingRecvQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  auto recv_done_waiter = hidden_unit->waiter;
  pending_recv_device_buffers_.Put(hidden_unit);
  recv_done_waiter->Wait();
  recv_done_waiter->Reset(1);
  return Status();
}

HiddenUnitDeviceBuffer* HiddenUnitBufferPool::GetFromPendingRecvQueue() { return pending_recv_device_buffers_.Get(); }

// After put the buffer to the pendding queue, then the buffer is valid, so need to get the buffer again.
Status HiddenUnitBufferPool::PutToDeviceRecvedQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  recved_device_buffers_.Put(hidden_unit->multi_batch_id, hidden_unit);
  return Status();
}
HiddenUnitDeviceBuffer* HiddenUnitBufferPool::GetFromDeviceRecvedQueue(size_t multi_batch_id) {
  return recved_device_buffers_.Get(multi_batch_id);
}

Status HiddenUnitBufferPool::Stop() {
  free_device_buffers_.Stop();
  recved_device_buffers_.Stop();

  recv_host_buffers_.Stop();
  pending_send_device_buffers_.Stop();
  pending_recv_device_buffers_.Stop();
  free_host_buffers_.Stop();

  is_stopped_ = true;

  return Status();
}

bool HiddenUnitBufferPool::Stopped() { return is_stopped_; }
}  // namespace ksana_llm
