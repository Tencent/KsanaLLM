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

    // Must initialized before create data channel instance.
    master_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
    worker_hidden_unit_buffer_pool_ = new HiddenUnitBufferPool();
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<Environment> env_ = nullptr;
  HiddenUnitBufferPool* master_hidden_unit_buffer_pool_ = nullptr;
  HiddenUnitBufferPool* worker_hidden_unit_buffer_pool_ = nullptr;
  std::shared_ptr<Context> master_context_ = nullptr;
  std::shared_ptr<Context> worker_context_ = nullptr;

  std::shared_ptr<NcclDataChannel> master_nccl_data_channel_ = nullptr;
  std::shared_ptr<NcclDataChannel> worker_nccl_data_channel_ = nullptr;
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

  float master_value = 3.14;
  size_t worker_value = 63;
  size_t multi_batch_num = 1;
  size_t master_multi_batch_id = 5;
  size_t worker_multi_batch_id = 4;

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

    int tp_para = Singleton<Environment>::GetInstance()->GetTensorParallelSize();
    uint32_t attn_data_parallel_size = Singleton<Environment>::GetInstance()->GetAttnDataParallelSize();
    master_context_ = std::make_shared<Context>(tp_para, attn_data_parallel_size, multi_batch_num);

    master_nccl_data_channel_ =
        std::make_shared<NcclDataChannel>(master_hidden_unit_buffer_pool_, env_, master_context_);

    // Create unique id and set to pipeline config.
    master_nccl_data_channel_->Listen();

    // Send nccl unique_id to child process.
    env_->GetPipelineConfig(pipeline_config_);
    ret = write(fd[1], pipeline_config_.nccl_unique_id, 128);
    if (ret < 0) {
      throw std::runtime_error("Write pipe error.");
    }
    close(fd[1]);

    master_nccl_data_channel_->Connect();

    // Get a device buffer
    HiddenUnitDeviceBuffer* master_dev_hidden_unit = master_hidden_unit_buffer_pool_->GetDeviceBuffer();
    Tensor& tensor = master_dev_hidden_unit->tensors[0];
    std::vector<float> buffer_data(8, 3.14);
    Memcpy(tensor.GetPtr<void>(), buffer_data.data(), buffer_data.size() * sizeof(float), MEMCPY_HOST_TO_DEVICE);
    SetHiddenUnitMeta(master_multi_batch_id, {1, 8}, DataType::TYPE_FP32);

    master_dev_hidden_unit->multi_batch_id = master_multi_batch_id;
    KLLM_LOG_INFO << "master start send";

    master_hidden_unit_buffer_pool_->PutToPendingSendQueue(master_dev_hidden_unit);
    StreamSynchronize(master_context_->GetComputeStreams()[pipeline_config_.node_rank]);
    KLLM_LOG_INFO << "master send done";

    // Recv from worker
    SetHiddenUnitMeta(worker_multi_batch_id, {1, 4}, DataType::TYPE_UINT64);
    HiddenUnitDeviceBuffer* master_recved_dev_hidden_unit = master_hidden_unit_buffer_pool_->GetDeviceBuffer();
    master_recved_dev_hidden_unit->multi_batch_id = worker_multi_batch_id;
    KLLM_LOG_INFO << "master start recv";
    master_hidden_unit_buffer_pool_->PutToPendingRecvQueue(master_recved_dev_hidden_unit);
    master_recved_dev_hidden_unit = master_hidden_unit_buffer_pool_->GetFromDeviceRecvedQueue(worker_multi_batch_id);
    KLLM_LOG_INFO << "master recv done";

    // check meta info
    EXPECT_EQ(master_recved_dev_hidden_unit->multi_batch_id, worker_multi_batch_id);
    EXPECT_EQ(master_recved_dev_hidden_unit->tensors[0].shape[0], 1);
    EXPECT_EQ(master_recved_dev_hidden_unit->tensors[0].shape[1], 4);
    EXPECT_EQ(master_recved_dev_hidden_unit->tensors[0].dtype, DataType::TYPE_UINT64);

    std::vector<size_t> recv_buffer_data(4, 0);
    Memcpy(recv_buffer_data.data(), master_recved_dev_hidden_unit->tensors[0].GetPtr<void>(),
           recv_buffer_data.size() * sizeof(size_t), MEMCPY_DEVICE_TO_HOST);
    for (auto v : recv_buffer_data) {
      EXPECT_EQ(v, worker_value);
    }

    master_nccl_data_channel_.reset();

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

    int tp_para = Singleton<Environment>::GetInstance()->GetTensorParallelSize();
    uint32_t attn_data_parallel_size = Singleton<Environment>::GetInstance()->GetAttnDataParallelSize();
    worker_context_ = std::make_shared<Context>(tp_para, attn_data_parallel_size, multi_batch_num);

    worker_nccl_data_channel_ =
        std::make_shared<NcclDataChannel>(worker_hidden_unit_buffer_pool_, env_, worker_context_);

    worker_nccl_data_channel_->Connect();

    // Recv from upstream
    SetHiddenUnitMeta(master_multi_batch_id, {1, 8}, DataType::TYPE_FP32);
    HiddenUnitDeviceBuffer* worker_recved_dev_hidden_unit = worker_hidden_unit_buffer_pool_->GetDeviceBuffer();
    worker_recved_dev_hidden_unit->multi_batch_id = master_multi_batch_id;
    KLLM_LOG_INFO << "worker start recv";
    worker_hidden_unit_buffer_pool_->PutToPendingRecvQueue(worker_recved_dev_hidden_unit);
    worker_recved_dev_hidden_unit = worker_hidden_unit_buffer_pool_->GetFromDeviceRecvedQueue(master_multi_batch_id);
    KLLM_LOG_INFO << "worker recv done";

    // check meta info
    EXPECT_EQ(worker_recved_dev_hidden_unit->multi_batch_id, master_multi_batch_id);
    EXPECT_EQ(worker_recved_dev_hidden_unit->tensors[0].shape[0], 1);
    EXPECT_EQ(worker_recved_dev_hidden_unit->tensors[0].shape[1], 8);
    EXPECT_EQ(worker_recved_dev_hidden_unit->tensors[0].dtype, DataType::TYPE_FP32);

    std::vector<float> recv_buffer_data(8, 0.0);
    Memcpy(recv_buffer_data.data(), worker_recved_dev_hidden_unit->tensors[0].GetPtr<void>(),
                recv_buffer_data.size() * sizeof(float), MEMCPY_DEVICE_TO_HOST);
    for (auto v : recv_buffer_data) {
      EXPECT_FLOAT_EQ(v, master_value);
    }

    // send to master
    HiddenUnitDeviceBuffer* worker_recved_dev_hidden_unit2 = worker_hidden_unit_buffer_pool_->GetDeviceBuffer();
    worker_recved_dev_hidden_unit2->multi_batch_id = worker_multi_batch_id;

    std::vector<size_t> worker_buffer_data(4, worker_value);
    Memcpy(worker_recved_dev_hidden_unit2->tensors[0].GetPtr<void>(), worker_buffer_data.data(),
           worker_buffer_data.size() * sizeof(size_t), MEMCPY_HOST_TO_DEVICE);
    SetHiddenUnitMeta(worker_multi_batch_id, {1, 4}, DataType::TYPE_UINT64);
    KLLM_LOG_INFO << "worker start send";
    worker_hidden_unit_buffer_pool_->PutToPendingSendQueue(worker_recved_dev_hidden_unit2);
    KLLM_LOG_INFO << "worker send done";

    worker_nccl_data_channel_.reset();
  }
}
