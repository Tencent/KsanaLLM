/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <condition_variable>
#include <memory>
#include <vector>

#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/raw_packet.h"
#include "ksana_llm/utils/barrier.h"
#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/expert_parallel_utils.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/tensor.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

class ExpertParallelHiddenUnitBufferPool {
 public:
  ExpertParallelHiddenUnitBufferPool();
  ~ExpertParallelHiddenUnitBufferPool() {}

  // Initialize necessary device buffer, so the block manager could use all left
  // device memory.
  void PreAllocateDeviceBuffer();

  // Get a hidden unit buffer object, do not create any new object.
  HiddenUnitDeviceBuffer* GetDeviceBuffer(int rank);
  HiddenUnitDeviceBuffer* GetCommMetaDeviceBuffer(int rank);
  HiddenUnitDeviceBuffer* GetDeviceBufferSingle();
  HiddenUnitDeviceBuffer* GetCommMetaDeviceBufferSingle();

  // Free the hidden unit buffer to object pool.
  Status FreeDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit);
  Status FreeCommMetaDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit);

  // Get and free the host buffer, create new object if needed.
  // Note: here will return a maximum size packet.
  Packet* GetHostBuffer();
  Status FreeHostBuffer(Packet* hidden_unit_buffer);

  // Put to and get from host received buffer.
  Status PutToHostRecvQueue(Packet* packet);
  Packet* GetFromHostRecvQueue();

  // Put to and get from device received buffer.
  Status PutToDeviceRecvQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromDeviceRecvQueue();
  HiddenUnitDeviceBuffer* AsyncGetFromDeviceRecvQueue();

  HiddenUnitDeviceBuffer* GetFromDeviceRecvCommMetaQueue();
  HiddenUnitDeviceBuffer* AsyncGetFromDeviceRecvCommMetaQueue();

  // Put to and get from send buffer.
  Status PutToSendQueue(HiddenUnitDeviceBuffer* hidden_unit);
  Status AsyncPutToSendQueue(HiddenUnitDeviceBuffer* hidden_unit);
  HiddenUnitDeviceBuffer* GetFromSendQueue();

  Status ConvertHostBufferToDevice(HiddenUnitDeviceBuffer* hidden_unit_dev, HiddenUnitHostBuffer* hidden_unit_host);
  Status ConvertDeviceBufferToHost(HiddenUnitHostBuffer* hidden_unit_host, HiddenUnitDeviceBuffer* hidden_unit_dev);

  // Get bytes of host buffer.
  size_t GetHostPacketSize(Packet* packet);

  // All blocked queue will be returned immdiately.
  Status Stop();

  // Whether current buffer pool is stopped.
  bool Stopped();

  // Wait until computation finished, and ready to receive data from remote.
  void WaitUtilReadyToRecv();
  void NotifySendFinished();
  void SetCommType(DistributedCommunicationType comm_type) { comm_type_ = comm_type; }
  size_t GetFreeDeviceBufferSize() { return free_device_buffers_.Size(); }
  size_t GetSendDeviceBufferSize() { return send_device_buffers_.Size(); }
  size_t GetRecvDeviceBufferSize() { return recv_device_buffers_.Size(); }
  size_t GetFreeCommMetaDeviceBufferSize() { return free_comm_meta_device_buffers_.Size(); }
  size_t GetSendCommMetaDeviceBufferSize() { return send_comm_meta_device_buffers_.Size(); }
  size_t GetRecvCommMetaDeviceBufferSize() { return recv_comm_meta_device_buffers_.Size(); }
  size_t GetRecvDeviceBufferResultSize() { return recv_device_buffers_result_.Size(); }
  size_t GetRecvDeviceBufferComputeSize() { return recv_device_buffers_compute_.Size(); }

  void NotifyDeviceRecv();
  int ExpertBarrier();
  size_t GetBarrierSize();
  size_t GetBarrierRemaining();
  size_t GetBarrierGeneration();

 private:
  // Initialize hidden unit device buffer, for max possible memory size.
  virtual Status InitializeHiddenUnitDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
  Status InitializeCommMetaHiddenUnitDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit_buffer);
  virtual void InitializeBufferSize();

 private:
  DataType weight_type_;
  size_t max_token_num_;
  size_t tensor_para_size_;
  size_t hidden_unit_size_;
  size_t expert_para_size_;
  // bytes of expert_parallel_comm_meta{}
  size_t comm_meta_size_;
  DistributedCommunicationType comm_type_ = DistributedCommunicationType::DEFAULT;

  // A waiter used to notify data receiving.
  std::shared_ptr<Waiter> recv_waiter_ = nullptr;

  // Make send operation blocked until finished.
  std::shared_ptr<Waiter> send_waiter_ = nullptr;

  // free device buffer, resuable.
  BlockingQueue<HiddenUnitDeviceBuffer*> free_device_buffers_;

  // received device buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> recv_device_buffers_;
  // received device buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> recv_device_buffers_compute_;
  // received device buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> recv_device_buffers_result_;

  // Send buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> send_device_buffers_;

  // Free comm meta buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> free_comm_meta_device_buffers_;
  // Free comm meta buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> recv_comm_meta_device_buffers_;
  // Free comm meta buffer.
  BlockingQueue<HiddenUnitDeviceBuffer*> send_comm_meta_device_buffers_;

  // Recv buffer.
  BlockingQueue<Packet*> recv_host_buffers_;

  // no used buffers.
  BlockingQueue<Packet*> free_host_buffers_;

  std::shared_ptr<Barrier> barrier_ = nullptr;

  bool is_stopped_ = false;
};
}  // namespace ksana_llm
