/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/connector/connector.h"
#include <gtest/gtest.h>
#include <memory>
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/device_types.h"

namespace ksana_llm {

class MockEnvironment : public Environment {
 public:
  MockEnvironment() {
    // 设置默认配置
    PipelineConfig pipeline_config;
    if (!GetPipelineConfig(pipeline_config).OK()) {
      pipeline_config = PipelineConfig();
    }
    pipeline_config.world_size = 1;
    pipeline_config.node_rank = 0;
    SetPipelineConfig(pipeline_config);

    // 为 TaskManager 提供有效的 block manager 配置，避免零 block_size
    BlockManagerConfig block_config;
    block_config.device_allocator_config.block_size = 1024 * 1024;  // 1MB pinned blocks
    block_config.device_allocator_config.blocks_num = 1;
    block_config.device_allocator_config.device = MemoryDevice::MEMORY_DEVICE;
    block_config.host_allocator_config.block_size = 1024 * 1024;
    block_config.host_allocator_config.blocks_num = 1;
    block_config.host_allocator_config.device = MemoryDevice::MEMORY_HOST;
    SetBlockManagerConfig(block_config);
  }
};

class ConnectorPushTaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
#if ACTIVE_DEVICE_TYPE != DEVICE_TYPE_NVIDIA
    GTEST_SKIP() << "Skipping connector test on non-NVIDIA platform: mapped host memory is only supported on NVIDIA";
#endif
    ConnectorConfig config;
    // Set required config fields for PinnedMemoryBufferPool
    config.transfer_batch = 1024;  // Must be > 0 for PinnedMemoryBufferPool
    config.device_count = 1;
    config.send_thread_num = 4;
    env_ = std::make_shared<MockEnvironment>();
    connector_ = std::make_unique<Connector>(config, /*attn_tensor_para_size=*/1, /*node_rank=*/0, env_);
  }
  std::shared_ptr<MockEnvironment> env_;
  std::unique_ptr<Connector> connector_;
};

TEST_F(ConnectorPushTaskTest, PushTaskBasic) {
  // 构造一个 TransferTask
  auto task = std::make_shared<TransferTask>();
  task->req_id = 42;
  task->tensor.block_idx = 1;
  task->tensor.layer_idx = 2;
  task->tensor.hash_device_id = 3;
  task->tensor.shape = {1, 2, 3};
  task->tensor.dtype = DataType::TYPE_FP32;
  task->addr = "127.0.0.1:50051";

  // 推送任务
  EXPECT_NO_THROW({ connector_->PushTask(task); });
}

TEST_F(ConnectorPushTaskTest, PushTaskNullptr) {
  // 推送空指针应抛出异常或安全返回
  std::shared_ptr<TransferTask> null_task;
  EXPECT_NO_THROW({ connector_->PushTask(null_task); });
}

TEST_F(ConnectorPushTaskTest, PushTaskPrefixCached) {
  // 构造一个 TransferTask
  auto task = std::make_shared<TransferTask>();
  task->req_id = 43;
  task->is_skipped_task = true;
  task->tensor.block_idx = 1;
  task->tensor.layer_idx = 2;
  task->tensor.hash_device_id = 3;
  task->tensor.shape = {1, 2, 3};
  task->tensor.dtype = DataType::TYPE_FP32;
  task->addr = "127.0.0.1:50051";

  // 推送任务
  EXPECT_NO_THROW({ connector_->PushTask(task); });
}

}  // namespace ksana_llm
