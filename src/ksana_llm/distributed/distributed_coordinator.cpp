/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/distributed/distributed_coordinator.h"

#include <stdexcept>

#include "fmt/core.h"
#include "ksana_llm/data_hub/expert_parallel_hidden_unit_buffer.h"
#include "ksana_llm/distributed/data_channel_factory.h"
#include "ksana_llm/utils/device_utils.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"
#ifdef ENABLE_CUDA
#  include "ksana_llm/distributed/nvidia/expert_parallel_nccl_data_channel.h"
#  include "ksana_llm/distributed/nvidia/nccl_data_channel.h"

#endif

namespace ksana_llm {

DistributedCoordinator::DistributedCoordinator(
    std::shared_ptr<Context> context, PacketCreationFunc packet_creation_fn, ScheduleOutputPool* schedule_output_pool,
    HiddenUnitBufferPool* hidden_unit_buffer_pool,
    ExpertParallelHiddenUnitBufferPool* expert_parallel_hidden_unit_buffer_pool, std::shared_ptr<Environment> env) {
  context_ = context;
  env_ = env ? env : Singleton<Environment>::GetInstance();

  env_->GetPipelineConfig(pipeline_config_);
  if (pipeline_config_.world_size > 1) {
    control_channel_ = std::make_shared<ControlChannel>(pipeline_config_.master_host, pipeline_config_.master_port,
                                                        pipeline_config_.world_size, pipeline_config_.node_rank,
                                                        packet_creation_fn, schedule_output_pool, env_);

#ifdef ENABLE_CUDA
    Status status = DataChannelFactory::CreateDataChannel(packet_creation_fn, hidden_unit_buffer_pool, env_, context_,
                                                          data_channel_);
    if (!status.OK()) {
      throw std::runtime_error(FormatStr("Ceate data channel error, %s", status.GetMessage()));
    }
#endif
  }

  // Crate ControlChannel and DataChannel for EP.
  env->GetExpertParallelConfig(expert_parallel_config_);
  if (expert_parallel_config_.global_expert_para_size > 1) {
    expert_parallel_control_channel_ = std::make_shared<ExpertParallelControlChannel>(
        expert_parallel_config_.expert_master_host, expert_parallel_config_.expert_master_port,
        expert_parallel_config_.expert_world_size, expert_parallel_config_.expert_node_rank, packet_creation_fn,
        schedule_output_pool, env_);

#ifdef ENABLE_CUDA
    Status status = DataChannelFactory::CreateExpertDataChannel<ExpertParallelNcclDataChannel>(
        packet_creation_fn, hidden_unit_buffer_pool, expert_parallel_hidden_unit_buffer_pool, env_, context_,
        expert_parallel_data_channel_);
    if (!status.OK()) {
      KLLM_LOG_INFO << "DistributedCoordinator Create expert_parallel_channel_ failed \n";

      throw std::runtime_error(FormatStr("Ceate data channel error, %s", status.GetMessage()));
    }
#endif

    KLLM_LOG_INFO << "DistributedCoordinator Create expert_parallel_channel_ succeed \n";
  }
  KLLM_LOG_INFO << "DistributedCoordinator() \n";
}

DistributedCoordinator::~DistributedCoordinator() {
  control_channel_.reset();
  data_channel_.reset();

  if (expert_parallel_config_.global_expert_para_size > 1) {
    expert_parallel_control_channel_.reset();
    expert_parallel_data_channel_.reset();
  }
}

// Extract function for pipeline parallel later.
Status DistributedCoordinator::InitializeCluster() {
  // Initialize pipline cluster.
  // Must invoke first, the add node method will report data port to master.
  if (pipeline_config_.world_size > 1) {
    Status status = data_channel_->Listen();
    if (!status.OK()) {
      throw std::runtime_error(fmt::format("Listen data port error: {}", status.GetMessage()));
    }

    if (context_->IsChief()) {
      status = control_channel_->Listen();
      if (!status.OK()) {
        throw std::runtime_error(fmt::format("Listen on {}:{} error: {}", pipeline_config_.master_host,
                                             pipeline_config_.master_port, status.GetMessage()));
      }
    } else {
      // Server maybe not ready, try connection at most 600 seconds.
      int try_times = 600;
      while (--try_times >= 0) {
        status = control_channel_->Connect();
        if (status.OK()) {
          break;
        }

        if (try_times == 0) {
          throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", pipeline_config_.master_host,
                                               pipeline_config_.master_port, status.GetMessage()));
        }

        KLLM_LOG_INFO << "DistributedCoordinator control channel connect failed, "
                         "try again.";
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      status = control_channel_->AddNode();
      if (!status.OK()) {
        throw std::runtime_error(
            fmt::format("Register node rank {} to master error: {}", pipeline_config_.node_rank, status.GetMessage()));
      }
    }
    // Wait until all nodes connected.
    control_channel_->Barrier();
  }

  // Initialize expert parallel cluser.
  if (expert_parallel_config_.expert_world_size > 1) {
    return InitializeExpertParallelCluster();
  } else {
    return Status();
  }
}

// For expert parallel.
Status DistributedCoordinator::InitializeExpertParallelCluster() {
  KLLM_LOG_INFO << "InitializeExpertParallelCluster.";

  // Must invoke first, the add node method will report data port to master.
  Status status = expert_parallel_data_channel_->Listen();
  if (!status.OK()) {
    throw std::runtime_error(fmt::format("ExpertParallel , listen data port error: {}", status.GetMessage()));
  }

  if (context_->IsExpertParallelChief()) {
    status = expert_parallel_control_channel_->Listen();
    if (!status.OK()) {
      throw std::runtime_error(fmt::format("Expert parallel listen on {}:{} error: {}",
                                           expert_parallel_config_.expert_master_host,
                                           expert_parallel_config_.expert_master_port, status.GetMessage()));
    }
  } else {
    // Server maybe not ready, try connection at most 600 seconds.
    int try_times = 600;
    while (--try_times >= 0) {
      status = expert_parallel_control_channel_->Connect();
      if (status.OK()) {
        break;
      }

      if (try_times == 0) {
        throw std::runtime_error(fmt::format("Expert parallel connect to {}:{} error: {}",
                                             expert_parallel_config_.expert_master_host,
                                             expert_parallel_config_.expert_master_port, status.GetMessage()));
      }

      KLLM_LOG_INFO << "DistributedCoordinator control channel of epxert "
                       "parallel connect failed, try again.";
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    status = expert_parallel_control_channel_->AddNode();
    if (!status.OK()) {
      throw std::runtime_error(fmt::format("Expert parallel register node rank {} to master error: {}",
                                           expert_parallel_config_.expert_node_rank, status.GetMessage()));
    }
  }

  // Wait until all nodes connected.
  return expert_parallel_control_channel_->Barrier();
}

Status DistributedCoordinator::SynchronizeNodeLayers(size_t master_offload_layer_num) {
  // This method tell the downstream data port of every node.
  control_channel_->SynchronizeNodeLayers(master_offload_layer_num);

  // Connect downstream node.
  int try_times = 600;
  while (--try_times >= 0) {
    Status status = data_channel_->Connect();
    if (status.OK()) {
      break;
    }

    if (try_times == 0 && !status.OK()) {
      throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", pipeline_config_.master_host,
                                           pipeline_config_.master_port, status.GetMessage()));
    }

    KLLM_LOG_INFO << "DistributedCoordinator data channel connect failed, try again.";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status();
}

Status DistributedCoordinator::SynchronizeExpertParallelExperts() {
  expert_parallel_control_channel_->SynchronizeExpertParallelExperts();
  // Connect downstream node.
  int try_times = 600;
  while (--try_times >= 0) {
    Status status = expert_parallel_data_channel_->Connect();
    if (status.OK()) {
      break;
    }

    if (try_times == 0 && !status.OK()) {
      throw std::runtime_error(fmt::format("Connect to {}:{} error: {}", expert_parallel_config_.expert_master_host,
                                           expert_parallel_config_.expert_master_port, status.GetMessage()));
    }

    KLLM_LOG_INFO << "DistributedCoordinator data channel connect failed, try again.";
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  return Status();
}

Status DistributedCoordinator::SynchronizeCacheBlockNum() { return control_channel_->SynchronizeCacheBlockNum(); }

Status DistributedCoordinator::Barrier() { return control_channel_->Barrier(); }

Status DistributedCoordinator::DestroyCluster() {
  if (!context_->IsStandalone()) {
    // Close all data channels.
    data_channel_->Disconnect();
    data_channel_->Close();

    if (context_->IsChief()) {
      control_channel_->Close();
    } else {
      control_channel_->Disconnect();
    }
  }

  return DestroyExpertCluster();
}

Status DistributedCoordinator::DestroyExpertCluster() {
  if (expert_parallel_config_.global_expert_para_size <= 1) return Status();

  // Close all data channels.
  expert_parallel_data_channel_->Disconnect();
  expert_parallel_data_channel_->Close();

  if (context_->IsExpertParallelChief()) {
    expert_parallel_control_channel_->Close();
  } else {
    expert_parallel_control_channel_->Disconnect();
  }

  return Status();
}

}  // namespace ksana_llm
