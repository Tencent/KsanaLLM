/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <thread>
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/data_channel_interface.h"
#include "ksana_llm/distributed/nvidia/nccl_data_channel.h"

#ifdef ENABLE_CUDA
#  include "cuda_runtime.h"
#  include "ksana_llm/utils/nvidia/cuda_utils.h"
#  include "ksana_llm/utils/nvidia/nccl_utils.h"
#endif

namespace ksana_llm {
class ExpertParallelNcclDataChannel : public DataChannelInterface {
 public:
  ExpertParallelNcclDataChannel(ExpertParallelHiddenUnitBufferPool* expert_hidden_unit_buffer_pool,
                                std::shared_ptr<Environment> env, std::shared_ptr<Context> context);
  ~ExpertParallelNcclDataChannel();

  virtual Status Listen() override;
  // Close open port.
  virtual Status Close() override;
  virtual Status Connect() override;
  // disconnect from master.
  virtual Status Disconnect() override;
  void SetTerminate(bool val) {
    terminated_ = val;
    return;
  }

  // Stop to accept any new connection.
  virtual Status Frozen() override;

 private:
  virtual Status ProcessSendDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit);
  virtual Status ProcessRecvDeviceBuffer(HiddenUnitDeviceBuffer* hidden_unit, int node_rank);

  // Thread loop.
  // rank: the expert parallel node rank of source data.
  Status ProcessRecvLoop(int rank);
  Status ProcessSendLoop();
  Status ProcessRecvTwiceLoop(int rank);
  Status ProcessSendTwiceLoop();
#ifdef ENABLE_CUDA
  // Convert data type to nccl data type.
  Status GetNcclDataType(DataType dtype, ncclDataType_t& nccl_dtype);

  std::vector<ncclUniqueId> nccl_unique_ids_;
  ncclUniqueId nccl_unique_id_;

  // The communicators for every device.
  std::vector<std::vector<ncclComm_t> > communicators_;
  std::vector<cudaStream_t> streams_;
#endif

 private:
  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  // The context.
  std::shared_ptr<Context> context_ = nullptr;

  ExpertParallelConfig expert_parallel_config_;

  std::string data_host_;
  uint16_t data_port_;

  // The rank ids of upstream and downstream device.
  // To be deleted next version.
  std::vector<int> upstream_ranks_;
  std::vector<int> downstream_ranks_;

  // later.
  // When world_size = 4, node_0, node_1, node_2 and node_3.
  std::vector<int> dst_ranks_;
  std::vector<int> src_ranks_;

  // The buffer pool.
  ExpertParallelHiddenUnitBufferPool* hidden_unit_buffer_pool_ = nullptr;

  // Receive data buffer from remote.
  std::vector<std::shared_ptr<std::thread> > recv_thread_ = {nullptr};

  // Send data buffers to remote.
  std::shared_ptr<std::thread> send_thread_ = nullptr;

  // Whether channel is terminated.
  bool terminated_ = false;
};

}  // namespace ksana_llm
