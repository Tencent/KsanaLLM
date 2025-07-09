/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/connector/connector.h"
#include <gtest/gtest.h>
#include <memory>
#include "ksana_llm/transfer/transfer_types.h"

namespace ksana_llm {

class MockEnvironment : public Environment {
 public:
  MockEnvironment() {
    // 设置默认配置
    PipelineConfig pipeline_config;
    GetPipelineConfig(pipeline_config);
    pipeline_config.world_size = 1;
    pipeline_config.node_rank = 0;
    SetPipelineConfig(pipeline_config);
  }
};

class ConnectorPushTaskTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ConnectorConfig config;
    // 可根据需要设置 config 字段
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
  task->tensor.device_idx = 3;
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

}  // namespace ksana_llm
