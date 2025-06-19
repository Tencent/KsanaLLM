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

 protected:
  // Convert data type to nccl data type.
#ifdef ENABLE_CUDA
  Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype);
#endif

 private:
  // Send or receive hidden unit through nccl.
  virtual Status ProcessDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit, bool is_send);

  // Thread loop.
  virtual Status ProcessRecvLoop();
  virtual Status ProcessSendLoop();

 private:
#ifdef ENABLE_CUDA
  ncclUniqueId nccl_unique_id_;
  // The communicators for every device.
  std::vector<ncclComm_t> communicators_;
#endif

  PipelineConfig pipeline_config_;

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
  std::shared_ptr<std::thread> recv_thread_ = nullptr;

  // Send data buffers to remote.
  std::shared_ptr<std::thread> send_thread_ = nullptr;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
