/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <vector>

#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/runtime/threadpool.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// Describe the hidden_units buffer
struct HiddenUnitDeviceBuffer {
 public:
  ~HiddenUnitDeviceBuffer();
  void NotifyFinished();

  // The unique id for one schedule step.
  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  // The device Tensor.
  std::vector<Tensor> tensors;

#ifdef ENABLE_ACL
  // Specially for NPU.
  bool prefill_enabled = false;
  bool decode_enabled = false;
  std::vector<Tensor> prefill_tensors;
#endif

  std::shared_ptr<Waiter> waiter = nullptr;
  // The communication type,
  // Can been used for pipeline parallel, expert parallel, data parallel.
  DistributedCommunicationType comm_type = DistributedCommunicationType::DEFAULT;
  // In scatter mode, rank 0 is the source of all downstream ranks
  // TODO(karlluo): support more group communication situation for example multi-to-multi
  uint32_t scatter_sender_rank = 0;
};

// Used for distributed mode,
// store the hidden units from network before copy to device.
// or hidden units from device bdefore send to network.
struct HiddenUnitHostBuffer {
  // The unique id for one schedule step.
  size_t multi_batch_id = DEFAULT_MULTI_BATCH_ID;

  // hidden unit shape, for one device, [max_token_num, hidden_unit_size]
  size_t shape_dims[2];

#ifdef ENABLE_ACL
  size_t prefill_shape_dims[2];
#endif

  // The device nummber.
  size_t tensor_parallel;

  // The data, for all devices.
  char data[0];
};

// The buffer pool used to manage hidden unit buffers.
class HiddenUnitBufferPool {
 public:
  HiddenUnitBufferPool();

  // Initialize necessary device buffer, so the block manager could use all left device memory.
  void PreAllocateDeviceBuffer();

  // Get a hidden unit buffer object, do not create any new object.
  HiddenUnitDeviceBuffer* GetDeviceBuffer();

  // Free the hidden unit buffer to object pool.
  Status FreeDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit);

  // Get and free the host buffer, create new object if needed.
  // Note: here will return a maximum size packet.
  Packet* GetHostBuffer();
  Status FreeHostBuffer(Packet* hidden_unit_buffer);

  // Put to and get from host received buffer.
  Status PutToHostRecvQueue(Packet* packet);
  Packet* GetFromHostRecvQueue();

  // After put the buffer to the pendding queue, then the buffer is valid, so need to get the buffer again.
  Status PutToDeviceRecvedQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromDeviceRecvedQueue(size_t multi_batch_id);

  // Put to and get from pending send buffer.
  Status PutToPendingSendQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromPendingSendQueue();

  // Put to and get from pending recv buffer.
  Status PutToPendingRecvQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromPendingRecvQueue();

  Status ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev, HiddenUnitHostBuffer* hidden_unit_host);
  Status ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host, HiddenUnitDeviceBuffer* hidden_unit_dev);

  // Get bytes of host buffer.
  size_t GetHostPacketSize(Packet* packet);

  // All blocked queue will be returned immediately.
  Status Stop();

  // Whether current buffer pool is stopped.
  bool Stopped();

  void SetCommType(DistributedCommunicationType comm_type) { comm_type_ = comm_type; }
  size_t GetFreeDeviceBufferSize() { return free_device_buffers_.Size(); }
  size_t GetSendDeviceBufferSize() { return pending_send_device_buffers_.Size(); }

 private:
  // Initialize hidden unit device buffer, for max possible memory size.
  virtual Status InitializeHiddenUnitDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);

  virtual void InitializeBufferSize();

 private:
  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;
  DistributedCommunicationType comm_type_ = DistributedCommunicationType::DEFAULT;

  // free device buffer, resuable.
  BlockingQueue<HiddenUnitDeviceBuffer*> free_device_buffers_;

  // received device buffer.
  BlockingQueueWithId<HiddenUnitDeviceBuffer*, size_t> recved_device_buffers_;
  // Recv buffer.
  BlockingQueue<Packet*> recv_host_buffers_;

  // Pending Send/Recv buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> pending_send_device_buffers_;
  BlockingQueue<HiddenUnitDeviceBuffer*> pending_recv_device_buffers_;

  // no used buffers.
  BlockingQueue<Packet*> free_host_buffers_;

  bool is_stopped_ = false;
};

}  // namespace ksana_llm
