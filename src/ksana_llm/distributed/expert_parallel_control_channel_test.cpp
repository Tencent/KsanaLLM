/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
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
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/data_hub/schedule_output.h"
#include "ksana_llm/distributed/control_channel.h"
#include "ksana_llm/distributed/expert_parallel_control_channel.h"

#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "test.h"

#include "ksana_llm/helpers/environment_test_helper.h"

using namespace ksana_llm;

class ExpertParallelControlChannelTest : public testing::Test {
 protected:
  void SetUp() override {
    master_env_ = std::make_shared<Environment>();
    worker_env_ = std::make_shared<Environment>();

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

    master_schedule_output_pool_ = new ScheduleOutputPool();
    worker_schedule_output_pool_ = new ScheduleOutputPool();

    std::string interface;
    GetAvailableInterfaceAndIP(interface, master_host_);
    GetAvailablePort(master_port_);

    ctrl_channel_master_ = std::make_shared<ExpertParallelControlChannel>(
        master_host_, master_port_, world_size_, 0, GetPacketObject, master_schedule_output_pool_, master_env_);
    ctrl_channel_worker_ = std::make_shared<ExpertParallelControlChannel>(
        master_host_, master_port_, world_size_, 1, GetPacketObject, worker_schedule_output_pool_, worker_env_);
  }

  void TearDown() override {
    ctrl_channel_worker_.reset();
    ctrl_channel_master_.reset();

    delete master_schedule_output_pool_;
    delete worker_schedule_output_pool_;
  }

 protected:
  std::shared_ptr<Environment> master_env_ = nullptr;
  std::shared_ptr<Environment> worker_env_ = nullptr;

  // The schedule output pool.
  ScheduleOutputPool* master_schedule_output_pool_ = nullptr;
  ScheduleOutputPool* worker_schedule_output_pool_ = nullptr;

  std::shared_ptr<ExpertParallelControlChannel> ctrl_channel_master_ = nullptr;
  std::shared_ptr<ExpertParallelControlChannel> ctrl_channel_worker_ = nullptr;

  std::string master_host_;
  uint16_t master_port_;

  size_t world_size_ = 2;
};

TEST_F(ExpertParallelControlChannelTest, TestControlChannel) {
  size_t master_device_block_num;
  size_t master_host_block_num;
  size_t worker_device_block_num;
  size_t worker_host_block_num;

  size_t master_offload_layer_num = 1;

  int16_t master_lower_layer_idx, master_upper_layer_idx, master_nextn_lower_layer_idx, master_nextn_upper_layer_idx;
  int16_t worker_lower_layer_idx, worker_upper_layer_idx, worker_nextn_lower_layer_idx, worker_nextn_upper_layer_idx;

  // master node.
  auto master_fn = [&]() {
    ExpertParallelConfig expert_parallel_config;
    master_env_->GetExpertParallelConfig(expert_parallel_config);
    expert_parallel_config.expert_world_size = 2;
    expert_parallel_config.global_expert_para_size = 2;
    expert_parallel_config.expert_para_size = 1;
#ifdef ENABLE_CUDA
    expert_parallel_config.nccl_unique_ids.resize(2);
    memset(expert_parallel_config.nccl_unique_ids[0].data(), 1, 128);
    memset(expert_parallel_config.nccl_unique_ids[0].data(), 2, 128);
#endif
    master_env_->SetExpertParallelConfig(expert_parallel_config);

    Singleton<Environment>::GetInstance()->SetExpertParallelConfig(expert_parallel_config);

    // Start master
    ctrl_channel_master_->Listen();

    // Wait all workers connected.
    ctrl_channel_master_->Barrier();

    // synchronize expert parallel info.
    ctrl_channel_master_->SynchronizeExpertParallelExperts();

    // Close master.
    ctrl_channel_master_->Close();
  };
  std::thread master_thread = std::thread(master_fn);

  // worker node.
  auto worker_fn = [&]() {
    ExpertParallelConfig expert_parallel_config;
    worker_env_->GetExpertParallelConfig(expert_parallel_config);
    expert_parallel_config.expert_world_size = 2;
    expert_parallel_config.global_expert_para_size = 2;
    expert_parallel_config.expert_para_size = 1;
#ifdef ENABLE_CUDA
    expert_parallel_config.nccl_unique_ids.resize(2);
#endif
    worker_env_->SetExpertParallelConfig(expert_parallel_config);

    Singleton<Environment>::GetInstance()->SetExpertParallelConfig(expert_parallel_config);

    // Start worker
    ctrl_channel_worker_->Connect();

    // Add worker to cluster.
    ctrl_channel_worker_->AddNode();

    // Wait all workers connected.
    ctrl_channel_worker_->Barrier();

    // Wait layer result.
    ctrl_channel_worker_->SynchronizeExpertParallelExperts();

    // Disconnect from master.
    ctrl_channel_worker_->Disconnect();
  };
  std::thread worker_thread = std::thread(worker_fn);

  master_thread.join();
  worker_thread.join();

  // Check layer range.

  ExpertParallelConfig worker_config;
  worker_env_->GetExpertParallelConfig(worker_config);
  ExpertParallelConfig master_config;
  master_env_->GetExpertParallelConfig(master_config);

#ifdef ENABLE_CUDA
  // Check block num result.
  EXPECT_EQ(worker_config.nccl_unique_ids[0], master_config.nccl_unique_ids[0]);
  EXPECT_EQ(worker_config.nccl_unique_ids[1], master_config.nccl_unique_ids[1]);
#endif
}
