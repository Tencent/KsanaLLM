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
#include "ksana_llm/data_hub/expert_data_hub.h"
#include "ksana_llm/data_hub/expert_parallel_deepep_wrapper.h"
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

// 创建一个MockExpertParallelDeepepWrapper类
class MockExpertParallelDeepepWrapper : public ExpertParallelDeepepWrapper {
 public:
  enum NodeType { MASTER, WORKER };

  explicit MockExpertParallelDeepepWrapper(NodeType type)
      : ExpertParallelDeepepWrapper(2, 1, type == MASTER ? 0 : 1, 128, 1024, 8, 64, nullptr), node_type_(type) {}

  uint8_t* GetNvshmemUniqueId() override {
    static uint8_t master_unique_id[kMaxNumRanks * kNvshmemUniqudIdSize];
    static uint8_t worker_unique_id[kMaxNumRanks * kNvshmemUniqudIdSize];
    static bool initialized = false;

    if (!initialized) {
      // 设置不同的特殊值，便于验证
      for (int i = 0; i < kMaxNumRanks * kNvshmemUniqudIdSize; i++) {
        master_unique_id[i] = 0xAA;
        worker_unique_id[i] = 0xCC;
      }
      initialized = true;
    }

    return (node_type_ == MASTER) ? master_unique_id : worker_unique_id;
  }

  // 覆盖GetIPCHandles方法，返回可控的值
  char* GetIPCHandles() override {
    static char master_ipc_handles[kMaxNumRanks * kIpcHandlesSize];
    static char worker_ipc_handles[kMaxNumRanks * kIpcHandlesSize];
    static bool initialized = false;

    if (!initialized) {
      // 设置不同的特殊值，便于验证
      for (int i = 0; i < kMaxNumRanks * kIpcHandlesSize; i++) {
        master_ipc_handles[i] = 0xBB;
        worker_ipc_handles[i] = 0xDD;
      }
      initialized = true;
    }
    return (node_type_ == MASTER) ? master_ipc_handles : worker_ipc_handles;
  }

  Status SetReady() override {
    return Status();
  }

 private:
  NodeType node_type_;
};

// 全局变量，分别存储master和worker的mock对象
std::shared_ptr<MockExpertParallelDeepepWrapper> g_master_mock_deepep_wrapper;
std::shared_ptr<MockExpertParallelDeepepWrapper> g_worker_mock_deepep_wrapper;

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

    // 初始化mock对象，确保在创建控制通道之前就已经设置好
    g_master_mock_deepep_wrapper =
        std::make_shared<MockExpertParallelDeepepWrapper>(MockExpertParallelDeepepWrapper::MASTER);
    g_worker_mock_deepep_wrapper =
        std::make_shared<MockExpertParallelDeepepWrapper>(MockExpertParallelDeepepWrapper::WORKER);

    ctrl_channel_master_ = std::make_shared<ExpertParallelControlChannel>(
        master_host_, master_port_, world_size_, 0, 2, GetPacketObject, master_schedule_output_pool_, master_env_);
    ctrl_channel_worker_ = std::make_shared<ExpertParallelControlChannel>(
        master_host_, master_port_, world_size_, 1, 2, GetPacketObject, worker_schedule_output_pool_, worker_env_);
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
    master_env_->SetExpertParallelConfig(expert_parallel_config);

    Singleton<Environment>::GetInstance()->SetExpertParallelConfig(expert_parallel_config);

    // Start master
    ctrl_channel_master_->Listen();

    // Wait all workers connected.
    ctrl_channel_master_->Barrier();

    // synchronize deepep meta info.
    ctrl_channel_master_->SetExpertParallelDeepepWrapper(g_master_mock_deepep_wrapper);
    ctrl_channel_master_->SynchronizeNvshmemUniqueId();

    // 验证master节点的结果
    uint8_t* nvshmem_unique_id = g_master_mock_deepep_wrapper->GetNvshmemUniqueId();
    char* ipc_handles = g_master_mock_deepep_wrapper->GetIPCHandles();

    // 验证master节点的nvshmem_unique_id是否包含了worker节点的数据
    // 由于SynchronizeNvshmemUniqueId函数会将worker节点的数据同步到master节点
    // 所以这里期望master节点的数据已经被修改为包含worker节点的数据
    for (size_t i = 0; i < kNvshmemUniqudIdSize; ++i) {
      EXPECT_EQ(nvshmem_unique_id[i], static_cast<uint8_t>(0xAA));
    }
    for (size_t i = 0; i < kIpcHandlesSize; ++i) {
      EXPECT_EQ(ipc_handles[i], static_cast<char>(0xBB));
      EXPECT_EQ(ipc_handles[i + kIpcHandlesSize], static_cast<char>(0xDD));
    }

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
    worker_env_->SetExpertParallelConfig(expert_parallel_config);

    Singleton<Environment>::GetInstance()->SetExpertParallelConfig(expert_parallel_config);

    // Start worker
    ctrl_channel_worker_->Connect();

    // Add worker to cluster.
    ctrl_channel_worker_->AddNode();

    // Wait all workers connected.
    ctrl_channel_worker_->Barrier();

    ctrl_channel_worker_->SetExpertParallelDeepepWrapper(g_worker_mock_deepep_wrapper);
    ctrl_channel_worker_->SynchronizeNvshmemUniqueId();
    // 验证worker节点的结果
    uint8_t* nvshmem_unique_id = g_worker_mock_deepep_wrapper->GetNvshmemUniqueId();
    char* ipc_handles = g_worker_mock_deepep_wrapper->GetIPCHandles();
    // 验证worker节点的nvshmem_unique_id是否已经被master节点的数据覆盖
    // 由于SynchronizeNvshmemUniqueId函数会将master节点的数据同步到worker节点
    // 所以这里期望worker节点的数据已经被修改为master节点的数据
    for (size_t i = 0; i < kNvshmemUniqudIdSize; ++i) {
      EXPECT_EQ(nvshmem_unique_id[i], static_cast<uint8_t>(0xAA));
    }
    for (size_t i = 0; i < kIpcHandlesSize; ++i) {
      EXPECT_EQ(ipc_handles[i], static_cast<char>(0xBB));
      EXPECT_EQ(ipc_handles[i + kIpcHandlesSize], static_cast<char>(0xDD));
    }
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
}
