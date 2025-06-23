/* Copyright 2024 Tencent Inc.  All rights reserved.
 * ==============================================================================*/

#include <gtest/gtest.h>

#include "ksana_llm/transfer/transfer_engine.h"
#include "ksana_llm/utils/environment.h"
#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

// 模拟 Environment 单例
class MockEnvironment : public Environment {
 public:
  MockEnvironment() {
    // 设置默认配置
    pipeline_config_.lower_layer_idx = 0;
    pipeline_config_.upper_layer_idx = 3;

    block_manager_config_.device_allocator_config.block_size = 4096;
    block_manager_config_.device_allocator_config.kv_cache_dtype = TYPE_FP32;
  }

  Status GetPipelineConfig(PipelineConfig& pipeline_config) const {
    pipeline_config = pipeline_config_;
    return Status();
  }

  Status GetBlockManagerConfig(BlockManagerConfig& block_manager_config) {
    block_manager_config = block_manager_config_;
    return Status();
  }

  size_t GetTensorParallelSize() const { return 2; }

 private:
  PipelineConfig pipeline_config_;
  BlockManagerConfig block_manager_config_;
};

// 创建一个继承自TransferEngine的模拟类
class MockTransferEngine : public TransferEngine {
 public:
  static std::shared_ptr<MockTransferEngine> GetInstance() { return Singleton<MockTransferEngine>::GetInstance(); }

  // 覆盖Initialize方法
  void Initialize(GroupRole group_role) {
    group_role_ = group_role;

    auto env = Singleton<MockEnvironment>::GetInstance();

    // 直接使用TransferConnector单例而不是创建新实例
    connector_ = std::make_shared<TransferConnector>(ConnectorConfig{}, 2, 0, env);

    // 初始化并启动传输连接器
    connector_->Initialize(group_role);
    connector_->Start();

    // 从环境中获取配置
    env->GetPipelineConfig(pipeline_config_);
    env->GetBlockManagerConfig(block_manager_config_);
    tensor_parallel_size_ = env->GetTensorParallelSize();

    // 计算派生值
    layer_num_ = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    block_size_ = block_manager_config_.device_allocator_config.block_size;
    kv_cache_dtype_ = block_manager_config_.device_allocator_config.kv_cache_dtype;
  }
};

// 测试类
class TransferEngineTest : public testing::Test {
 protected:
  void SetUp() override {
    // 创建模拟环境的实例
    mock_env_ = Singleton<MockEnvironment>::GetInstance();

    // 获取 MockTransferEngine 实例
    transfer_engine_ = MockTransferEngine::GetInstance();
  }

  void TearDown() override {}

  std::shared_ptr<MockEnvironment> mock_env_;
  std::shared_ptr<MockTransferEngine> transfer_engine_;
};

// 测试初始化功能
TEST(TransferEngineTestInitialize, Initialize) {
  // 测试 DECODE 角色初始化，使用TransferConnector模板参数避免真实Connector造成的崩溃
  TransferEngine::GetInstance()->Initialize<Environment, TransferConnector>(GroupRole::DECODE);
  ASSERT_EQ(TransferEngine::GetInstance()->GetTransferMeta(0), nullptr);

  // 测试 PREFILL 角色初始化，使用TransferConnector模板参数避免真实Connector造成的崩溃
  TransferEngine::GetInstance()->Initialize<Environment, TransferConnector>(GroupRole::PREFILL);
  ASSERT_EQ(TransferEngine::GetInstance()->GetTransferMeta(0), nullptr);
}

// 测试添加传输元数据
TEST_F(TransferEngineTest, AddTransferMeta) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);  // 分配一些内存作为模拟块
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 验证元数据是否正确添加
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->shared_token_num, shared_token_num);
  ASSERT_EQ(meta->gpu_blocks.size(), 2);
  ASSERT_EQ(meta->gpu_blocks[0].size(), 3);

  // 清理分配的内存
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试无效请求ID的情况
TEST_F(TransferEngineTest, AddTransferMetaInvalidRequestId) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据，使用无效的请求ID
  int request_id = -1;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 验证元数据是否未添加（因为请求ID无效）
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_EQ(meta, nullptr);
  for (size_t i = 0; i < gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < gpu_blocks[i].size(); ++j) {
      free(gpu_blocks[i][j]);
    }
  }
}

// 测试空GPU块的情况
TEST_F(TransferEngineTest, AddTransferMetaEmptyBlocks) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 添加空的GPU块，但至少有一个设备和一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 验证元数据是否正确添加
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->gpu_blocks.size(), 1);

  // 清理分配的内存
  free(meta->gpu_blocks[0][0]);
}

// 测试发送功能（PREFILL角色）
TEST_F(TransferEngineTest, SendWithPrefillRole) {
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 修改 pipeline_config_ 以使 ValidateLayerIndex 返回 true
  mock_env_->pipeline_config_.lower_layer_idx = 0;
  mock_env_->pipeline_config_.upper_layer_idx = 3;

  // 重新初始化引擎以应用新的配置
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 发送特定设备和层的数据
  int device_idx = 0;
  int layer_idx = 1;  // 确保在有效范围内
  transfer_engine_->Send(device_idx, layer_idx);

  // 验证元数据是否存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 验证元数据中的shared_token_num是否正确
  ASSERT_EQ(meta->shared_token_num, shared_token_num);

  // 清理分配的内存
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试发送功能（DECODE角色）
TEST_F(TransferEngineTest, SendWithDecodeRole) {
  // 初始化引擎为DECODE角色
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 发送特定设备和层的数据
  int device_idx = 0;
  int layer_idx = 1;
  transfer_engine_->Send(device_idx, layer_idx);
  ASSERT_FALSE(transfer_engine_->IsRecvDone(123) != -1);
}

// 测试发送功能（无效层索引）
TEST_F(TransferEngineTest, SendWithInvalidLayerIndex) {
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 修改 pipeline_config_ 以明确定义有效范围
  mock_env_->pipeline_config_.lower_layer_idx = 0;
  mock_env_->pipeline_config_.upper_layer_idx = 3;

  // 重新初始化引擎以应用新的配置
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 发送无效层索引的数据，但不添加任何元数据
  int device_idx = 0;
  int invalid_layer_idx = 10;  // 超出配置的层范围
  transfer_engine_->Send(device_idx, invalid_layer_idx);

  // 验证无效层索引的发送不会导致程序崩溃
  int valid_layer_idx = 1;  // 有效范围内的层索引
  transfer_engine_->Send(device_idx, valid_layer_idx);
}

// 测试发送token功能
TEST_F(TransferEngineTest, SendTokens) {
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  std::vector<std::tuple<std::string, int, int>> reqs_tokens = {{"", 123, 456}, {"", 789, 101}};

  // 发送token
  transfer_engine_->Send(reqs_tokens);

  // 验证发送后可以检查发送状态
  // 创建测试数据并添加元数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建一个设备，一个块
  gpu_blocks.resize(2);
  gpu_blocks[0].resize(1);
  gpu_blocks[1].resize(1);
  gpu_blocks[0][0] = malloc(4096);
  gpu_blocks[1][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  gpu_blocks = meta->gpu_blocks;
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 4; ++j) {
      transfer_engine_->Send(i, j);
    }
  }

  // 检查发送是否完成
  bool is_done = transfer_engine_->IsSendDone(request_id);
  ASSERT_TRUE(is_done);

  transfer_engine_->CleanupTransferMeta(request_id);
  // 清理分配的内存
  meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_EQ(meta, nullptr);
  for (size_t i = 0; i < gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < gpu_blocks[i].size(); ++j) {
      free(gpu_blocks[i][j]);
    }
  }
}

// 测试发送token功能（DECODE角色）
TEST_F(TransferEngineTest, SendTokensWithDecodeRole) {
  // 初始化引擎为DECODE角色
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  std::vector<std::tuple<std::string, int, int>> reqs_tokens = {{"", 123, 456}, {"", 789, 101}};

  // 发送token
  transfer_engine_->Send(reqs_tokens);

  // 验证DECODE角色不会发送数据
  // 创建测试数据并添加元数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建一个设备，一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 检查发送是否完成 - 对于DECODE角色，这应该返回false
  bool is_done = transfer_engine_->IsSendDone(request_id);
  ASSERT_FALSE(is_done);

  // 清理分配的内存
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  for (size_t i = 0; i < meta->gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < meta->gpu_blocks[i].size(); ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试发送token功能（空请求）
TEST_F(TransferEngineTest, SendEmptyTokens) {
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建空的测试数据
  std::vector<std::tuple<std::string, int, int>> reqs_tokens;

  // 发送token
  transfer_engine_->Send(reqs_tokens);

  // 验证空请求不会导致程序崩溃
  std::vector<std::tuple<std::string, int, int>> non_empty_reqs_tokens = {{"", 123, 456}};
  transfer_engine_->Send(non_empty_reqs_tokens);
  ASSERT_FALSE(transfer_engine_->IsSendDone(123));
}

// 测试接收完成检查
TEST_F(TransferEngineTest, IsRecvDone) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 检查接收是否完成
  int first_token = transfer_engine_->IsRecvDone(request_id);
  ASSERT_EQ(first_token, -1);

  // 清理分配的内存
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  for (size_t i = 0; i < meta->gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < meta->gpu_blocks[i].size(); ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试接收完成检查（无效请求ID）
TEST_F(TransferEngineTest, IsRecvDoneInvalidRequestId) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 检查不存在的请求ID
  int invalid_request_id = 999;
  int first_token = transfer_engine_->IsRecvDone(invalid_request_id);

  // 验证结果（应该返回-1，因为请求ID不存在）
  ASSERT_EQ(first_token, -1);
}

// 测试发送完成检查
TEST_F(TransferEngineTest, IsSendDone) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 检查发送是否完成
  bool is_done = transfer_engine_->IsSendDone(request_id);

  // 验证结果
  ASSERT_FALSE(is_done);

  // 清理分配的内存
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  for (size_t i = 0; i < meta->gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < meta->gpu_blocks[i].size(); ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试清理传输元数据
TEST_F(TransferEngineTest, CleanupTransferMeta) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_token_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_token_num, gpu_blocks);

  // 验证元数据是否存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理元数据
  gpu_blocks = meta->gpu_blocks;
  bool result = transfer_engine_->CleanupTransferMeta(request_id);
  ASSERT_TRUE(result);

  // 验证元数据是否已清理
  meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_EQ(meta, nullptr);

  // 清理分配的内存
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      free(gpu_blocks[i][j]);
    }
  }
}

// 测试清理不存在的传输元数据
TEST_F(TransferEngineTest, CleanupNonExistentTransferMeta) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 清理不存在的元数据
  int invalid_request_id = 999;
  bool result = transfer_engine_->CleanupTransferMeta(invalid_request_id);

  // 验证结果（应该返回false，因为元数据不存在）
  ASSERT_FALSE(result);
}

}  // namespace ksana_llm