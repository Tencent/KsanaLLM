/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/distributed/distributed_coordinator.h"
#include "ksana_llm/distributed/packet_util.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class DistributedCoordinatorTest : public testing::Test {
 protected:
  void SetUp() override {
    master_env_ = std::make_shared<Environment>();
    worker_env_ = std::make_shared<Environment>();

    uint16_t master_port;
    std::string master_host;
    std::string master_interface;

    GetAvailableInterfaceAndIP(master_interface, master_host);
    GetAvailablePort(master_port);

    // Set model config.
    std::string config_file = GetTestConfigFile();
    Singleton<Environment>::GetInstance()->ParseConfig(config_file);
    master_env_->ParseConfig(config_file);
    worker_env_->ParseConfig(config_file);

    BlockManagerConfig master_block_manager_config;
    master_env_->InitializeBlockManagerConfig();
    master_env_->GetBlockManagerConfig(master_block_manager_config);
    master_block_manager_config.device_allocator_config.blocks_num = 10;
    master_block_manager_config.host_allocator_config.blocks_num = 8;
    master_env_->SetBlockManagerConfig(master_block_manager_config);

    BlockManagerConfig worker_block_manager_config;
    worker_env_->InitializeBlockManagerConfig();
    worker_env_->GetBlockManagerConfig(worker_block_manager_config);
    worker_block_manager_config.device_allocator_config.blocks_num = 6;
    worker_block_manager_config.host_allocator_config.blocks_num = 4;
    worker_env_->SetBlockManagerConfig(worker_block_manager_config);

    // default config.
    PipelineConfig default_pipeline_config;
    Singleton<Environment>::GetInstance()->GetPipelineConfig(default_pipeline_config);

    // Set master config.
    PipelineConfig master_pipeline_config;
    master_env_->GetPipelineConfig(master_pipeline_config);
    master_pipeline_config.master_host = master_host;
    master_pipeline_config.master_port = master_port;
    master_pipeline_config.world_size = 2;
    master_pipeline_config.node_rank = 0;
    master_env_->SetPipelineConfig(master_pipeline_config);

    int master_tp_para = master_env_->GetTensorParallelSize();
    int master_attn_data_parallel_size = master_env_->GetAttnDataParallelSize();
    Singleton<Environment>::GetInstance()->SetPipelineConfig(master_pipeline_config);
    master_context_ = std::make_shared<Context>(master_tp_para, master_attn_data_parallel_size);

    // Set worker config.
    PipelineConfig worker_pipeline_config;
    worker_env_->GetPipelineConfig(worker_pipeline_config);
    worker_pipeline_config.master_host = master_host;
    worker_pipeline_config.master_port = master_port;
    worker_pipeline_config.world_size = 2;
    worker_pipeline_config.node_rank = 1;
    worker_env_->SetPipelineConfig(worker_pipeline_config);

    int worker_tp_para = worker_env_->GetTensorParallelSize();
    uint32_t worker_attn_data_parallel_size = worker_env_->GetAttnDataParallelSize();
    Singleton<Environment>::GetInstance()->SetPipelineConfig(worker_pipeline_config);
    worker_context_ = std::make_shared<Context>(worker_tp_para, worker_attn_data_parallel_size);

    // Restore pipeline config.
    Singleton<Environment>::GetInstance()->SetPipelineConfig(default_pipeline_config);

    // Must initialized before create data channel instance.
    master_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
    worker_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();

    master_schedule_output_pool_ = new ScheduleOutputPool();
    worker_schedule_output_pool_ = new ScheduleOutputPool();

    // The packet creation function.
    auto master_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
      if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
        Packet* packet = master_hidden_unit_buffer_pool_->GetHostBuffer();
        packet->size = master_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
        packet->type = packet_type;
        return packet;
      }

      return GetPacketObject(packet_type, body_size);
    };

    auto worker_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
      if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
        Packet* packet = worker_hidden_unit_buffer_pool_->GetHostBuffer();
        packet->size = worker_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
        packet->type = packet_type;
        return packet;
      }

      return GetPacketObject(packet_type, body_size);
    };

    setenv("USE_TCP_DATA_CHANNEL", "1", 1);
    master_distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        master_context_, master_packet_creation_fn, master_schedule_output_pool_, master_hidden_unit_buffer_pool_,
        nullptr, master_env_);
    worker_distributed_coordinator_ = std::make_shared<DistributedCoordinator>(
        worker_context_, worker_packet_creation_fn, worker_schedule_output_pool_, worker_hidden_unit_buffer_pool_,
        nullptr, worker_env_);
  }

  void TearDown() override {
    master_distributed_coordinator_.reset();
    worker_distributed_coordinator_.reset();

    delete master_hidden_unit_buffer_pool_;
    delete worker_hidden_unit_buffer_pool_;

    delete master_schedule_output_pool_;
    delete worker_schedule_output_pool_;
  }

 protected:
  std::shared_ptr<Context> master_context_ = nullptr;
  std::shared_ptr<Context> worker_context_ = nullptr;

  std::shared_ptr<Environment> master_env_ = nullptr;
  std::shared_ptr<Environment> worker_env_ = nullptr;

  HiddenUnitBufferPool* master_hidden_unit_buffer_pool_ = nullptr;
  HiddenUnitBufferPool* worker_hidden_unit_buffer_pool_ = nullptr;
  ExpertParallelHiddenUnitBufferPool* master_ep_hidden_unit_buffer_pool_ = nullptr;
  ExpertParallelHiddenUnitBufferPool* worker_ep_hidden_unit_buffer_pool_ = nullptr;

  // The schedule output pool.
  ScheduleOutputPool* master_schedule_output_pool_ = nullptr;
  ScheduleOutputPool* worker_schedule_output_pool_ = nullptr;

  std::shared_ptr<DistributedCoordinator> master_distributed_coordinator_ = nullptr;
  std::shared_ptr<DistributedCoordinator> worker_distributed_coordinator_ = nullptr;
};

TEST_F(DistributedCoordinatorTest, TestDistributedCoordinator) {
  // Check context.
  EXPECT_TRUE(master_context_->IsChief() == true);
  EXPECT_TRUE(worker_context_->IsChief() == false);
  size_t master_offload_layer_num = 1;
  // master node.
  auto master_fn = [&]() {
    master_distributed_coordinator_->InitializeCluster();
    master_distributed_coordinator_->SynchronizeNodeLayers(master_offload_layer_num);
    master_distributed_coordinator_->SynchronizeCacheBlockNum();
    master_distributed_coordinator_->DestroyCluster();
  };
  std::thread master_thread = std::thread(master_fn);

  // worker node.
  auto worker_fn = [&]() {
    worker_distributed_coordinator_->InitializeCluster();
    worker_distributed_coordinator_->SynchronizeNodeLayers(master_offload_layer_num);
    worker_distributed_coordinator_->SynchronizeCacheBlockNum();
    worker_distributed_coordinator_->DestroyCluster();
  };
  std::thread worker_thread = std::thread(worker_fn);

  master_thread.join();
  worker_thread.join();

  // Check layers and block num.
  PipelineConfig master_pipeline_config;
  PipelineConfig worker_pipeline_config;
  master_env_->GetPipelineConfig(master_pipeline_config);
  worker_env_->GetPipelineConfig(worker_pipeline_config);

  EXPECT_EQ(master_pipeline_config.lower_layer_idx, 0);
  EXPECT_EQ(master_pipeline_config.upper_layer_idx, 15 - master_offload_layer_num);
  EXPECT_EQ(worker_pipeline_config.lower_layer_idx, 16 - master_offload_layer_num);
  EXPECT_EQ(worker_pipeline_config.upper_layer_idx, 31);

  EXPECT_EQ(master_pipeline_config.device_block_num, 6);
  EXPECT_EQ(master_pipeline_config.host_block_num, 4);
  EXPECT_EQ(worker_pipeline_config.device_block_num, 6);
  EXPECT_EQ(worker_pipeline_config.host_block_num, 4);
}

TEST_F(DistributedCoordinatorTest, TestDistributedCoordinatorForEP) {
  uint16_t master_port;
  std::string master_host;
  std::string master_interface;

  GetAvailableInterfaceAndIP(master_interface, master_host);
  GetAvailablePort(master_port);

  // Reset pipeline config.
  PipelineConfig master_pipeline_config;
  master_env_->GetPipelineConfig(master_pipeline_config);
  master_pipeline_config.world_size = 1;
  master_env_->SetPipelineConfig(master_pipeline_config);

  PipelineConfig worker_pipeline_config;
  worker_env_->GetPipelineConfig(worker_pipeline_config);
  worker_pipeline_config.world_size = 1;
  worker_env_->SetPipelineConfig(worker_pipeline_config);

  // Set expert parallel config.
  ExpertParallelConfig master_ep_config;
  ExpertParallelConfig worker_ep_config;

  master_env_->GetExpertParallelConfig(master_ep_config);
  master_ep_config.expert_master_host = master_host;
  master_ep_config.expert_master_port = master_port;
  master_ep_config.expert_world_size = 2;
  master_ep_config.expert_node_rank = 0;
  master_ep_config.expert_para_size = 1;
  master_ep_config.global_expert_para_size = 2;
#ifdef ENABLE_CUDA
  master_ep_config.nccl_unique_ids.resize(2);
#endif
  master_env_->SetExpertParallelConfig(master_ep_config);

  worker_env_->GetExpertParallelConfig(worker_ep_config);
  worker_ep_config.expert_master_host = master_host;
  worker_ep_config.expert_master_port = master_port;
  worker_ep_config.expert_world_size = 2;
  worker_ep_config.expert_node_rank = 1;
  worker_ep_config.expert_para_size = 1;
  worker_ep_config.global_expert_para_size = 2;
#ifdef ENABLE_CUDA
  worker_ep_config.nccl_unique_ids.resize(2);
#endif
  worker_env_->SetExpertParallelConfig(worker_ep_config);

  int master_tp_para = master_env_->GetTensorParallelSize();
  int master_attn_data_parallel_size = master_env_->GetAttnDataParallelSize();
  Singleton<Environment>::GetInstance()->SetExpertParallelConfig(master_ep_config);
  master_context_ = std::make_shared<Context>(master_tp_para, master_attn_data_parallel_size);

  int worker_tp_para = worker_env_->GetTensorParallelSize();
  uint32_t worker_attn_data_parallel_size = worker_env_->GetAttnDataParallelSize();
  Singleton<Environment>::GetInstance()->SetExpertParallelConfig(worker_ep_config);
  worker_context_ = std::make_shared<Context>(worker_tp_para, worker_attn_data_parallel_size);

  master_ep_hidden_unit_buffer_pool_ = new ExpertParallelHiddenUnitBufferPool();
  // worker_ep_hidden_unit_buffer_pool_ = new ExpertParallelHiddenUnitBufferPool();
  InitializeExpertHiddenUnitBufferPool();
  worker_ep_hidden_unit_buffer_pool_ = GetExpertHiddenUnitBufferPool();
  GetExpertHiddenUnitBufferPool()->SetCommType(DistributedCommunicationType::SCATTER);

  // The packet creation function.
  auto master_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
    if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
      Packet* packet = master_ep_hidden_unit_buffer_pool_->GetHostBuffer();
      packet->size = master_ep_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
      packet->type = packet_type;
      return packet;
    }

    return GetPacketObject(packet_type, body_size);
  };

  auto worker_packet_creation_fn = [&](PacketType packet_type, size_t body_size) -> Packet* {
    if (packet_type == PacketType::DATA_REQ_HIDDEN_UNIT) {
      Packet* packet = worker_ep_hidden_unit_buffer_pool_->GetHostBuffer();
      packet->size = worker_ep_hidden_unit_buffer_pool_->GetHostPacketSize(packet);
      packet->type = packet_type;
      return packet;
    }

    return GetPacketObject(packet_type, body_size);
  };

  master_distributed_coordinator_ =
      std::make_shared<DistributedCoordinator>(master_context_, master_packet_creation_fn, master_schedule_output_pool_,
                                               nullptr, master_ep_hidden_unit_buffer_pool_, master_env_);
  worker_distributed_coordinator_ =
      std::make_shared<DistributedCoordinator>(worker_context_, worker_packet_creation_fn, worker_schedule_output_pool_,
                                               nullptr, worker_ep_hidden_unit_buffer_pool_, worker_env_);

  // Check context.
  EXPECT_TRUE(master_context_->IsExpertParallelChief() == true);
  EXPECT_TRUE(worker_context_->IsExpertParallelChief() == false);

  size_t master_offload_layer_num = 1;
  // master node.
  auto master_fn = [&]() {
    master_distributed_coordinator_->InitializeCluster();
    master_distributed_coordinator_->SynchronizeExpertParallelExperts();
    master_distributed_coordinator_->DestroyCluster();
  };
  std::thread master_thread = std::thread(master_fn);

  // worker node.
  auto worker_fn = [&]() {
    worker_distributed_coordinator_->InitializeCluster();
    worker_distributed_coordinator_->SynchronizeExpertParallelExperts();
    worker_distributed_coordinator_->DestroyCluster();
  };
  std::thread worker_thread = std::thread(worker_fn);

  master_thread.join();
  worker_thread.join();
}
