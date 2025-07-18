/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <memory>

#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/data_hub/hidden_unit_buffer.h"
#include "ksana_llm/distributed/control_channel.h"
#include "ksana_llm/distributed/data_channel_interface.h"
#include "ksana_llm/distributed/expert_parallel_control_channel.h"
#include "ksana_llm/distributed/expert_parallel_data_channel.h"

#include "ksana_llm/utils/context.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/status.h"

namespace ksana_llm {

// manage all the distributed pipeline nodes.
class DistributedCoordinator {
 public:
  DistributedCoordinator(std::shared_ptr<Context> context, PacketCreationFunc packet_creation_fn = GetPacketObject,
                         ScheduleOutputPool* schedule_output_pool = nullptr,
                         HiddenUnitBufferPool* hidden_unit_buffer_pool = nullptr,
                         ExpertParallelHiddenUnitBufferPool* expert_parallel_hidden_unit_buffer_pool = nullptr,
                         std::shared_ptr<Environment> env = nullptr);
  ~DistributedCoordinator();

  // Initialize and destroy the distributed cluster.
  Status InitializeCluster();
  Status DestroyCluster();
  Status DestroyExpertCluster();

  // Initialize for expert parallel cluster.
  Status InitializeExpertParallelCluster();
  Status SynchronizeExpertParallelExperts();

  // Synchronize layers and block num.
  Status SynchronizeNodeLayers(size_t master_offload_layer_num);
  Status SynchronizeCacheBlockNum();
  Status Barrier();

  // Stop to accept any new connection.
  Status Frozen();

 private:
  PipelineConfig pipeline_config_;

  // The environments.
  std::shared_ptr<Environment> env_ = nullptr;

  // Global context.
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<ControlChannel> control_channel_ = nullptr;
  std::shared_ptr<DataChannelInterface> data_channel_ = nullptr;

  // Expert parallel config.
  ExpertParallelConfig expert_parallel_config_;
  std::shared_ptr<ExpertParallelControlChannel> expert_parallel_control_channel_ = nullptr;
  std::shared_ptr<DataChannelInterface> expert_parallel_data_channel_ = nullptr;
};

}  // namespace ksana_llm
