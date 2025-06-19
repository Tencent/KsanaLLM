/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <unistd.h>
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

#include "ksana_llm/cache_manager/prefix_cache_manager_test_helper.h"
#include "ksana_llm/data_hub/data_hub.h"
#include "ksana_llm/distributed/control_channel.h"

#include "ksana_llm/distributed/data_channel.h"
#include "ksana_llm/distributed/nvidia/nccl_data_channel.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/memory_utils.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/socket_util.h"
#include "ksana_llm/utils/status.h"
#include "ksana_llm/utils/string_utils.h"
#include "test.h"

using namespace ksana_llm;

// TODO(robertyuan): nccl only transmitted tensor, no schedule_id. Use default value.
#define SCHEDULE_ID DEFAULT_SCHEDULE_ID

class NcclDataChannelTest : public testing::Test {
 protected:
  void SetUp() override {}

  void Initialize() {
    // Set model config.
    std::filesystem::path current_path = __FILE__;
    std::filesystem::path parent_path = current_path.parent_path();
    std::filesystem::path config_path_relate = parent_path / "../../../../examples/llama7b/ksana_llm.yaml";
    std::string config_path = std::filesystem::absolute(config_path_relate).string();

    Singleton<Environment>::GetInstance()->ParseConfig(config_path);
    env_ = Singleton<Environment>::GetInstance();

    // Set block manager.
    BlockManagerConfig block_manager_config;
    Singleton<Environment>::GetInstance()->InitializeBlockManagerConfig();
    Singleton<Environment>::GetInstance()->GetBlockManagerConfig(block_manager_config);

    int tp_para = Singleton<Environment>::GetInstance()->GetTensorParallelSize();
    uint32_t attn_data_parallel_size = Singleton<Environment>::GetInstance()->GetAttnDataParallelSize();
    context_ = std::make_shared<Context>(tp_para, attn_data_parallel_size);

    // Must initialized before create data channel instance.
    hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<Environment> env_ = nullptr;
  HiddenUnitBufferPool* hidden_unit_buffer_pool_ = nullptr;
  std::shared_ptr<Context> context_ = nullptr;

  std::shared_ptr<NcclDataChannel> nccl_data_channel_ = nullptr;
  PipelineConfig pipeline_config_;
};

TEST_F(NcclDataChannelTest, TestDataChannel) {
  int fd[2];
  int ret = pipe(fd);
  if (ret == -1) {
    throw std::runtime_error("Create pipe error.");
  }

  char unique_id[128];

  const char* all_devices = getenv("CUDA_VISIBLE_DEVICES");
  std::vector<std::string> devices = Str2Vector(all_devices, ",");

  pid_t pid = fork();
  if (pid > 0) {
    close(fd[0]);

    setenv("CUDA_VISIBLE_DEVICES", devices[0].c_str(), 1);

    Initialize();

    // Get nccl unique_id from pipeline_config.
    env_->GetPipelineConfig(pipeline_config_);

    pipeline_config_.world_size = 2;
    pipeline_config_.node_rank = 0;
    env_->SetPipelineConfig(pipeline_config_);

    nccl_data_channel_ = std::make_shared<NcclDataChannel>(hidden_unit_buffer_pool_, env_, context_);

    // Create unique id and set to pipeline config.
    nccl_data_channel_->Listen();

    // Send nccl unique_id to child process.
    env_->GetPipelineConfig(pipeline_config_);
    ret = write(fd[1], pipeline_config_.nccl_unique_id, 128);
    if (ret < 0) {
      throw std::runtime_error("Write pipe error.");
    }
    close(fd[1]);

    nccl_data_channel_->Connect();

    // Get a device buffer
    HiddenUnitDeviceBuffer* dev_hidden_unit = hidden_unit_buffer_pool_->GetDeviceBuffer();

    Tensor& tensor = dev_hidden_unit->tensors[0];
    std::vector<float> buffer_data(8, 3.14);
    Memcpy(tensor.GetPtr<void>(), buffer_data.data(), buffer_data.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE);
    SetHiddenUnitMeta({1, 8}, DataType::TYPE_FP32);

    dev_hidden_unit->schedule_id = SCHEDULE_ID;
    hidden_unit_buffer_pool_->PutToSendQueue(dev_hidden_unit);

    CUDA_CHECK(cudaDeviceSynchronize());

    nccl_data_channel_.reset();

  } else {
    close(fd[1]);

    setenv("CUDA_VISIBLE_DEVICES", devices[1].c_str(), 1);

    Initialize();

    // Recv nccl unique_id from parent process.
    memset(unique_id, 0, 128);
    ret = read(fd[0], unique_id, 128);
    if (ret < 0) {
      throw std::runtime_error("Read pipe error.");
    }
    close(fd[0]);

    // Write nccl unique_id to pipeline_config.
    env_->GetPipelineConfig(pipeline_config_);
    memcpy(pipeline_config_.nccl_unique_id, unique_id, 128);

    pipeline_config_.world_size = 2;
    pipeline_config_.node_rank = 1;
    env_->SetPipelineConfig(pipeline_config_);

    nccl_data_channel_ = std::make_shared<NcclDataChannel>(hidden_unit_buffer_pool_, env_, context_);

    nccl_data_channel_->Connect();

    SetHiddenUnitMeta({1, 8}, DataType::TYPE_FP32);

    // Recv from upstream
    HiddenUnitDeviceBuffer* dev_hidden_unit = hidden_unit_buffer_pool_->GetFromDeviceRecvQueue(SCHEDULE_ID);

    CUDA_CHECK(cudaDeviceSynchronize());

    Tensor& tensor = dev_hidden_unit->tensors[0];
    std::vector<float> buffer_data(8, 0.0);
    Memcpy(buffer_data.data(), tensor.GetPtr<void>(), buffer_data.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);

    for (auto v : buffer_data) {
      EXPECT_FLOAT_EQ(v, 3.14);
    }

    nccl_data_channel_.reset();
  }
}
