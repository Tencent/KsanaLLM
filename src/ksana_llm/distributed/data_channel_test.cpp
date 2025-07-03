/* Copyright 2024 Tencent Inc.  All rights reserved.

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
#include "ksana_llm/cache_manager/prefix_cache_manager_test_helper.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/control_channel.h"

#include "ksana_llm/distributed/data_channel.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "test.h"

using namespace ksana_llm;

class DataChannelTest : public testing::Test {
 protected:
  void SetUp() override {
    InitLoguru();

    master_env_ = std::make_shared<Environment>();
    worker_env_ = std::make_shared<Environment>();

    // Set model config.
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    master_env_->ParseConfig(config_path);
    worker_env_->ParseConfig(config_path);

    // Set block manager.
    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->InitializeBlockManagerConfig();
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);

    // Must initialized before create data channel instance.
    master_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
    worker_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();

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

    master_data_channel_ =
        std::make_shared<DataChannel>(master_packet_creation_fn, master_hidden_unit_buffer_pool_, master_env_);
    worker_data_channel_ =
        std::make_shared<DataChannel>(worker_packet_creation_fn, worker_hidden_unit_buffer_pool_, worker_env_);
  }

  void TearDown() override {
    worker_data_channel_.reset();
    master_data_channel_.reset();

    delete master_hidden_unit_buffer_pool_;
    delete worker_hidden_unit_buffer_pool_;
  }

 protected:
  std::shared_ptr<Environment> master_env_ = nullptr;
  std::shared_ptr<Environment> worker_env_ = nullptr;

  HiddenUnitBufferPool* master_hidden_unit_buffer_pool_ = nullptr;
  HiddenUnitBufferPool* worker_hidden_unit_buffer_pool_ = nullptr;

  std::shared_ptr<DataChannel> master_data_channel_ = nullptr;
  std::shared_ptr<DataChannel> worker_data_channel_ = nullptr;
};

TEST_F(DataChannelTest, TestDataChannel) {
  // Start master node on available port.
  master_data_channel_->Listen();

  {
    // Get master port and write to worker config.
    PipelineConfig master_pipeline_config;
    master_env_->GetPipelineConfig(master_pipeline_config);

    PipelineConfig worker_pipeline_config;
    worker_env_->GetPipelineConfig(worker_pipeline_config);

    worker_pipeline_config.downstream_host = master_pipeline_config.data_host;
    worker_pipeline_config.downstream_port = master_pipeline_config.data_port;
    worker_env_->SetPipelineConfig(worker_pipeline_config);
  }

  // Start worker node on available port.
  worker_data_channel_->Listen();

  {
    // Get worker port and write to master config.
    PipelineConfig worker_pipeline_config;
    worker_env_->GetPipelineConfig(worker_pipeline_config);

    PipelineConfig master_pipeline_config;
    master_env_->GetPipelineConfig(master_pipeline_config);

    master_pipeline_config.downstream_host = worker_pipeline_config.data_host;
    master_pipeline_config.downstream_port = worker_pipeline_config.data_port;
    master_env_->SetPipelineConfig(master_pipeline_config);
  }

  // Connect to downstream node.
  master_data_channel_->Connect();
  worker_data_channel_->Connect();

  // Get a device buffer
  HiddenUnitDeviceBuffer* master_dev_hidden_unit = master_hidden_unit_buffer_pool_->GetDeviceBuffer();
  size_t master_schedule_id = 5;
  master_dev_hidden_unit->multi_batch_id = master_schedule_id;

  // Send from master to worker.
  KLLM_LOG_INFO << "master start send";
  master_hidden_unit_buffer_pool_->PutToPendingSendQueue(master_dev_hidden_unit);
  KLLM_LOG_INFO << "master send done";

  // Should be sent to worker, get it and check id.
  HiddenUnitDeviceBuffer* worker_dev_hidden_unit = worker_hidden_unit_buffer_pool_->GetDeviceBuffer();
  worker_dev_hidden_unit->multi_batch_id = master_schedule_id;
  KLLM_LOG_INFO << "master start recv";
  worker_hidden_unit_buffer_pool_->PutToPendingRecvQueue(worker_dev_hidden_unit);
  worker_dev_hidden_unit = worker_hidden_unit_buffer_pool_->GetFromDeviceRecvedQueue(master_schedule_id);
  KLLM_LOG_INFO << "master recv done";

  EXPECT_EQ(worker_dev_hidden_unit->multi_batch_id, master_schedule_id);

  // Change the id and send back from worker to master.
  size_t worker_schedule_id = 7;
  worker_dev_hidden_unit->multi_batch_id = worker_schedule_id;
  worker_hidden_unit_buffer_pool_->PutToPendingSendQueue(worker_dev_hidden_unit);

  // Should be sent to master, get it and check new id.
  HiddenUnitDeviceBuffer* master_dev_hidden_unit_2 = worker_hidden_unit_buffer_pool_->GetDeviceBuffer();
  master_dev_hidden_unit_2->multi_batch_id = worker_schedule_id;
  master_hidden_unit_buffer_pool_->PutToPendingRecvQueue(master_dev_hidden_unit_2);
  master_dev_hidden_unit_2 = master_hidden_unit_buffer_pool_->GetFromDeviceRecvedQueue(worker_schedule_id);
  EXPECT_EQ(master_dev_hidden_unit_2->multi_batch_id, worker_schedule_id);

  master_data_channel_->Disconnect();
  worker_data_channel_->Disconnect();

  master_data_channel_->Close();
  worker_data_channel_->Close();
}
