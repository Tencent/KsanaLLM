/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <thread>
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"

#include "ksana_llm/distributed/data_channel_interface.h"
#include "ksana_llm/utils/context.h"

#include "ksana_llm/utils/blocking_queue.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

// A fast data channel implement using NCCL.
class NcclDataChannel : public DataChannelInterface {
 public:
  NcclDataChannel(HiddenUnitBufferPool* hidden_unit_buffer_pool, std::shared_ptr<Environment> env,
                  std::shared_ptr<Context> context);

  virtual ~NcclDataChannel();

  // For master node only.
  virtual Status Listen() override;

  // Close open port.
  virtual Status Close() override;

  // For normal node only.
  virtual Status Connect() override;

  // disconnect from master.
  virtual Status Disconnect() override;

  // Stop to accept any new connection.
  virtual Status Frozen() override;

 protected:
  // Convert data type to nccl data type.
#ifdef ENABLE_CUDA
  Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype);
  void SendRecvDeviceData(bool is_send, size_t multi_batch_id, int dev_id, DistributedCommunicationType comm_type,
                          uint32_t scatter_sender_rank, std::vector<Stream>& streams, void* data_ptr, int64_t count,
                          ncclDataType_t nccl_dtype);
#endif

 private:
  // Send or receive hidden unit through nccl.
  virtual Status ProcessDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit, bool is_send);

  // Thread loop.
  virtual Status ProcessRecvLoop();
  virtual Status ProcessSendLoop();

  // just mark as called ncclsend / ncclrecv functions.
  // donot actually send done or really recved the data.
  void NotifyRecvCommandLaunched(HiddenUnitDeviceBuffer* hidden_unit);
  void NotifySendCommandLaunched(HiddenUnitDeviceBuffer* hidden_unit);

  size_t GetRecordEventsBatchId(size_t multi_batch_id);
  void WaitUtilRecvFinished(HiddenUnitDeviceBuffer* hidden_unit);

  // keep send and recv in order to have better performance.
  void WaitLastRecvDone();

 private:
#ifdef ENABLE_CUDA
  ncclUniqueId nccl_unique_id_;
  // The communicators for every device.
  std::vector<ncclComm_t> communicators_;
#endif

  PipelineConfig pipeline_config_;

  // [multi_batch_id, device_id]
  std::vector<std::vector<Event>> recved_events_;

  // The rank ids of upstream and downstream device.
  std::vector<int> upstream_ranks_;
  std::vector<int> downstream_ranks_;

  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  // The context.
  std::shared_ptr<Context> context_ = nullptr;

  // The buffer pool.
  HiddenUnitBufferPool* hidden_unit_buffer_pool_ = nullptr;

  // Receive data buffer from remote.
  std::unique_ptr<std::thread> recv_thread_ = nullptr;

  // Send data buffers to remote.
  std::unique_ptr<std::thread> send_thread_ = nullptr;

  // keep send recv in order
  std::unique_ptr<Waiter> last_recv_done_waiter_ = nullptr;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
