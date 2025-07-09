/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include <condition_variable>
#include <memory>
#include <vector>
#include "ksana_llm/data_hub/expert_data_hub.h"

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/blocking_queue.h"

#include "ksana_llm/utils/barrier.h"
#include "ksana_llm/utils/expert_parallel_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

ExpertParallelHiddenUnitBufferPool::ExpertParallelHiddenUnitBufferPool() {
  //
  barrier_ = std::make_shared<Barrier>(1);

  // recv_thread
  recv_waiter_ = std::make_shared<Waiter>(1);

  // send_gpu
  send_waiter_ = std::make_shared<Waiter>(1);

  InitializeBufferSize();
}

void ExpertParallelHiddenUnitBufferPool::PreAllocateDeviceBuffer() {
  HiddenUnitDeviceBuffer* dev_hidden_unit = new HiddenUnitDeviceBuffer();
  InitializeHiddenUnitDeviceBuffer(dev_hidden_unit);
  free_device_buffers_.Put(dev_hidden_unit);

  HiddenUnitDeviceBuffer* comm_meta_dev_hidden_unit = new HiddenUnitDeviceBuffer();
  InitializeCommMetaHiddenUnitDeviceBuffer(comm_meta_dev_hidden_unit);
  free_comm_meta_device_buffers_.Put(comm_meta_dev_hidden_unit);
}

Status ExpertParallelHiddenUnitBufferPool::ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev,
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
    // Not need.
    //  SetDevice(i);

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

Status ExpertParallelHiddenUnitBufferPool::ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host,
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
#endif

#ifdef ENABLE_ACL
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
    // SetDevice(i);
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

size_t ExpertParallelHiddenUnitBufferPool::GetHostPacketSize(Packet* packet) {
  HiddenUnitHostBuffer* hidden_unit = reinterpret_cast<HiddenUnitHostBuffer*>(packet->body);
  size_t host_packet_size = sizeof(HiddenUnitHostBuffer) +
                            (hidden_unit->shape_dims[0] * hidden_unit->shape_dims[1] * GetTypeSize(weight_type_));
#ifdef ENABLE_ACL
  host_packet_size +=
      (hidden_unit->prefill_shape_dims[0] * hidden_unit->prefill_shape_dims[1] * GetTypeSize(weight_type_));
#endif
  return host_packet_size;
}

void ExpertParallelHiddenUnitBufferPool::InitializeBufferSize() {
  ModelConfig model_config;
  Status status = Singleton<Environment>::GetInstance()->GetModelConfig(model_config);
  if (!status.OK()) {
    KLLM_THROW("InitializeBufferSize failed. status:" + status.ToString());
  }
  ExpertParallelConfig expert_parallel_config;
  Singleton<Environment>::GetInstance()->GetExpertParallelConfig(expert_parallel_config);
  DistributedCommunicationType comm_type = expert_parallel_config.expert_para_comm_type;
  SetCommType(comm_type);

  weight_type_ = model_config.weight_data_type;
  tensor_para_size_ = model_config.tensor_para_size;
  max_token_num_ = model_config.max_step_token_num;
  hidden_unit_size_ = model_config.size_per_head * model_config.head_num;
  expert_para_size_ = expert_parallel_config.expert_para_size;
  comm_meta_size_ = sizeof(expert_parallel_comm_meta);

  barrier_->Init(tensor_para_size_);

  KLLM_LOG_DEBUG << "HiddenUnitBufferPool::InitializeBufferSize weight_type:" << weight_type_
                 << ", tensor_para_size:" << tensor_para_size_ << ", max_token_num:" << max_token_num_
                 << ", hidden_unit_size:" << hidden_unit_size_ << ", expert_para_size:" << expert_para_size_
                 << ", comm_meta_size_:" << comm_meta_size_;
}

Status ExpertParallelHiddenUnitBufferPool::InitializeHiddenUnitDeviceBuffer(
    HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  // only need expert_para_size_
  hidden_unit_buffer->tensors.resize(tensor_para_size_);
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_tensors.resize(tensor_para_size_);
#endif
#ifdef ENABLE_CUDA
  int device_id;
  GetDevice(&device_id);
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
// rank 0 allocate device
#ifdef ENABLE_CUDA
  SetDevice(device_id);
#endif

#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  hidden_unit_buffer->comm_type = comm_type_;

  KLLM_LOG_DEBUG << "ExpertParallelHiddenUnitBufferPool::InitializeHiddenUnitDeviceBuffe, shape:"
                 << Vector2Str(std::vector<size_t>(hidden_unit_buffer->tensors[0].shape))
                 << ", max_token_num:" << max_token_num_ << ", hidden_unit_size:" << hidden_unit_size_
                 << ", Communication Type: " << static_cast<int32_t>(comm_type_);

  return Status();
}

Status ExpertParallelHiddenUnitBufferPool::InitializeCommMetaHiddenUnitDeviceBuffer(
    HiddenUnitDeviceBuffer* hidden_unit_buffer) {
  // only need expert_para_size_
  hidden_unit_buffer->tensors.resize(tensor_para_size_);
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_tensors.resize(tensor_para_size_);
#endif
  int device_id;
  GetDevice(&device_id);
  for (int rank = 0; rank < tensor_para_size_; ++rank) {
    SetDevice(rank);
    hidden_unit_buffer->tensors[rank] =
        Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_UINT8, {comm_meta_size_}, rank);
#ifdef ENABLE_ACL
    hidden_unit_buffer->prefill_tensors[rank] =
        Tensor(MemoryLocation::LOCATION_DEVICE, DataType::TYPE_UINT8, {comm_meta_size_}, rank);
#endif
  }
  SetDevice(device_id);

#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  hidden_unit_buffer->comm_type = comm_type_;

  KLLM_LOG_DEBUG << "ExpertParallelHiddenUnitBufferPool::InitializeCommMetaHiddenUnitDeviceBuffer, shape:"
                 << Vector2Str(std::vector<size_t>(hidden_unit_buffer->tensors[0].shape))
                 << ", max_token_num:" << max_token_num_ << ", hidden_unit_size:" << hidden_unit_size_
                 << ", Communication Type: " << static_cast<int32_t>(comm_type_);

  return Status();
}

// Rank of 0 do bufer allocation, and other ranks get the returned the same device buffer.
HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetDeviceBuffer(int rank) {
  HiddenUnitDeviceBuffer* hidden_unit;
  if (rank == 0) {
    hidden_unit = free_device_buffers_.NonBlockingGet();

    // Create new device buffer for sending.
    if (hidden_unit == nullptr) {
      KLLM_LOG_INFO
          << "ExpertParallelHiddenUnitBufferPool::GetDeviceBuffer Create device buffer, should called only once.";
      hidden_unit = new HiddenUnitDeviceBuffer();
      InitializeHiddenUnitDeviceBuffer(hidden_unit);
    }

    SetCurrentExpertRecvHiddenUnitBuffer(hidden_unit);
  }
  // rank == 0, allocate memory.
  GetExpertHiddenUnitBufferPool()->ExpertBarrier();
  return GetCurrentExpertRecvHiddenUnitBuffer();
}

//
HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetDeviceBufferSingle() {
  HiddenUnitDeviceBuffer* hidden_unit;

  hidden_unit = free_device_buffers_.NonBlockingGet();

  // Create new device buffer for sending.
  if (hidden_unit == nullptr) {
    KLLM_LOG_INFO
        << "ExpertParallelHiddenUnitBufferPool::GetDeviceBuffer Create device buffer, should called only once.";
    hidden_unit = new HiddenUnitDeviceBuffer();
    InitializeHiddenUnitDeviceBuffer(hidden_unit);
  }
  return hidden_unit;
}

HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetCommMetaDeviceBuffer(int rank) {
  HiddenUnitDeviceBuffer* hidden_unit;

  if (rank == 0) {
    hidden_unit = free_comm_meta_device_buffers_.NonBlockingGet();

    // Create new device buffer for sending.
    if (hidden_unit == nullptr) {
      KLLM_LOG_INFO << "ExpertParallelHiddenUnitBufferPool::GetCommMetaDeviceBuffer Create device buffer, should "
                       "called only once.";
      hidden_unit = new HiddenUnitDeviceBuffer();
      InitializeCommMetaHiddenUnitDeviceBuffer(hidden_unit);
    }

    SetCurrentExpertRecvCommMetaHiddenUnitBuffer(hidden_unit);
  }
  GetExpertHiddenUnitBufferPool()->ExpertBarrier();
  return GetCurrentExpertRecvCommMetaHiddenUnitBuffer();
}

HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetCommMetaDeviceBufferSingle() {
  HiddenUnitDeviceBuffer* hidden_unit;

  hidden_unit = free_comm_meta_device_buffers_.NonBlockingGet();

  // Create new device buffer for sending.
  if (hidden_unit == nullptr) {
    KLLM_LOG_INFO << "ExpertParallelHiddenUnitBufferPool::GetCommMetaDeviceBuffer Create device buffer, should "
                     "called only once.";
    hidden_unit = new HiddenUnitDeviceBuffer();
    InitializeCommMetaHiddenUnitDeviceBuffer(hidden_unit);
  }

  return hidden_unit;
}

// Free the hidden unit buffer to object pool.
Status ExpertParallelHiddenUnitBufferPool::FreeDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  free_device_buffers_.Put(hidden_unit_buffer);
  return Status();
}

// Free the hidden unit buffer to object pool.
Status ExpertParallelHiddenUnitBufferPool::FreeCommMetaDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer) {
#ifdef ENABLE_ACL
  hidden_unit_buffer->prefill_enabled = false;
  hidden_unit_buffer->decode_enabled = false;
#endif

  free_comm_meta_device_buffers_.Put(hidden_unit_buffer);
  return Status();
}

Packet* ExpertParallelHiddenUnitBufferPool::GetHostBuffer() {
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

Status ExpertParallelHiddenUnitBufferPool::FreeHostBuffer(Packet* hidden_unit_buffer) {
  free_host_buffers_.Put(hidden_unit_buffer);
  return Status();
}

Status ExpertParallelHiddenUnitBufferPool::PutToHostRecvQueue(Packet* packet) {
  recv_host_buffers_.Put(packet);
  return Status();
}

Packet* ExpertParallelHiddenUnitBufferPool::GetFromHostRecvQueue() { return recv_host_buffers_.Get(); }

Status ExpertParallelHiddenUnitBufferPool::PutToDeviceRecvQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  recv_device_buffers_.Put(hidden_unit);
  return Status();
}

void ExpertParallelHiddenUnitBufferPool::WaitUtilReadyToRecv() {
  recv_waiter_->Wait();
  recv_waiter_->Reset(1);
}

void ExpertParallelHiddenUnitBufferPool::NotifySendFinished() { send_waiter_->Notify(); }

void ExpertParallelHiddenUnitBufferPool::NotifyDeviceRecv() {
  recv_waiter_->Notify();
  return;
}

// Notify recv thread and get data.
HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::AsyncGetFromDeviceRecvQueue() {
  recv_waiter_->Notify();
  return recv_device_buffers_.NonBlockingGet();
}

HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetFromDeviceRecvQueue() {
  recv_waiter_->Notify();
  return recv_device_buffers_.Get();
}

// Notify recv thread and get data.
HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::AsyncGetFromDeviceRecvCommMetaQueue() {
  recv_waiter_->Notify();
  return recv_comm_meta_device_buffers_.NonBlockingGet();
}

HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetFromDeviceRecvCommMetaQueue() {
  recv_waiter_->Notify();
  return recv_comm_meta_device_buffers_.Get();
}

Status ExpertParallelHiddenUnitBufferPool::PutToSendQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  send_device_buffers_.Put(hidden_unit);
  send_waiter_->Wait();
  send_waiter_->Reset(1);
  return Status();
}

Status ExpertParallelHiddenUnitBufferPool::AsyncPutToSendQueue(HiddenUnitDeviceBuffer* hidden_unit) {
  send_device_buffers_.Put(hidden_unit);
  // send_waiter_->Wait();
  send_waiter_->Reset(1);
  return Status();
}

HiddenUnitDeviceBuffer* ExpertParallelHiddenUnitBufferPool::GetFromSendQueue() { return send_device_buffers_.Get(); }

Status ExpertParallelHiddenUnitBufferPool::Stop() {
  free_device_buffers_.Stop();
  recv_device_buffers_.Stop();

  recv_host_buffers_.Stop();
  send_device_buffers_.Stop();
  free_host_buffers_.Stop();

  send_comm_meta_device_buffers_.Stop();
  recv_comm_meta_device_buffers_.Stop();
  free_comm_meta_device_buffers_.Stop();

  recv_waiter_->Stop();
  is_stopped_ = true;

  return Status();
}

bool ExpertParallelHiddenUnitBufferPool::Stopped() { return is_stopped_; }

int ExpertParallelHiddenUnitBufferPool::ExpertBarrier() {
  barrier_->arrive_and_wait();
  return 0;
}

size_t ExpertParallelHiddenUnitBufferPool::GetBarrierSize() { return barrier_->get_thread_count(); }

size_t ExpertParallelHiddenUnitBufferPool::GetBarrierRemaining() { return barrier_->get_remaining(); }
size_t ExpertParallelHiddenUnitBufferPool::GetBarrierGeneration() { return barrier_->get_generation(); }

}  // namespace ksana_llm
