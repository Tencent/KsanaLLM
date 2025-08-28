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

    // 设置默认的chunk传输配置
    batch_scheduler_config_.transfer_layer_chunk_size = 1;
  }

  Status GetPipelineConfig(PipelineConfig& pipeline_config) const {
    pipeline_config = pipeline_config_;
    return Status();
  }

  Status GetBlockManagerConfig(BlockManagerConfig& block_manager_config) {
    block_manager_config = block_manager_config_;
    return Status();
  }

  size_t GetTransferLayerChunkSize() { return batch_scheduler_config_.transfer_layer_chunk_size; }

  // 用于测试的配置设置方法
  void SetTransferLayerChunkSize(size_t chunk_size) { batch_scheduler_config_.transfer_layer_chunk_size = chunk_size; }

 private:
  PipelineConfig pipeline_config_;
  BlockManagerConfig block_manager_config_;
  BatchSchedulerConfig batch_scheduler_config_;
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
    connector_->Initialize(group_role, device_info_manager_);
    connector_->Start();

    // 从环境中获取配置
    env->GetPipelineConfig(pipeline_config_);
    env->GetBlockManagerConfig(block_manager_config_);
    tensor_parallel_size_ = 2;
    attn_data_parallel_size_ = 2;
    // 计算派生值
    layer_num_ = pipeline_config_.upper_layer_idx - pipeline_config_.lower_layer_idx + 1;
    block_size_ = block_manager_config_.device_allocator_config.block_size;
    kv_cache_dtype_ = block_manager_config_.device_allocator_config.kv_cache_dtype;
    transfer_layer_chunk_size_ = env->GetTransferLayerChunkSize();
  }

  void InsertReciveDeviceInfo(const std::string& group_key, int adp_num, int dev_total_num) {
    device_info_manager_->Insert(group_key, adp_num, dev_total_num);
  }

  void SetSelfDeviceInfo(int adp_num, int dev_total_num) {
    attn_data_parallel_size_ = adp_num;
    tensor_parallel_size_ = dev_total_num;
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
  KLLM_LOG_INFO << "transfer_engine_ initialized";

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);  // 分配一些内存作为模拟块
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 验证元数据是否正确添加
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  ASSERT_EQ(meta->shared_block_num, shared_block_num);
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
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 验证元数据是否未添加（因为请求ID无效）
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_EQ(meta, nullptr);
  for (size_t i = 0; i < gpu_blocks.size(); ++i) {
    for (size_t j = 0; j < gpu_blocks[i].size(); ++j) {
      free(gpu_blocks[i][j]);
    }
  }
}

// 测试无效shared_block_num的情况
TEST_F(TransferEngineTest, AddTransferMetaInvalidSharedTokenNum) {
  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据，使用无效的shared_block_num
  int request_id = -1;
  size_t shared_block_num = 5;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};

  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 验证元数据是否未添加（因为shared_block_num大于块的数量）
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
  size_t shared_block_num = 1;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 添加空的GPU块，但至少有一个设备和一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

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
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 发送特定设备和层的数据
  int device_idx = 0;
  int layer_idx = 1;  // 确保在有效范围内
  transfer_engine_->InsertReciveDeviceInfo("", 2, 2);

  transfer_engine_->Send(device_idx, layer_idx);

  // 验证元数据是否存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 验证元数据中的shared_block_num是否正确
  ASSERT_EQ(meta->shared_block_num, shared_block_num);

  // 验证创建的TransferTask数量是否正确
  ASSERT_EQ(meta->transfer_tasks_deque_.size(), 3);

  // 清理分配的内存
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试P端adp > D端adp时的发送功能（PREFILL角色）
TEST_F(TransferEngineTest, SendWithPrefillAdpMoreRole) {
  int total_device_num = 2;
  int prefill_adp = 2;
  int decode_adp = 1;
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 修改 pipeline_config_ 以使 ValidateLayerIndex 返回 true
  mock_env_->pipeline_config_.lower_layer_idx = 0;
  mock_env_->pipeline_config_.upper_layer_idx = 3;

  // 重新初始化引擎以应用新的配置
  transfer_engine_->Initialize(GroupRole::PREFILL);

  transfer_engine_->SetSelfDeviceInfo(prefill_adp, total_device_num);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建两个设备，每个设备有3个块
  int prefill_atp = total_device_num / prefill_adp;
  gpu_blocks.resize(prefill_atp);
  for (size_t i = 0; i < prefill_atp; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 发送特定设备和层的数据
  int device_idx = 0;
  int layer_idx = 1;  // 确保在有效范围内
  transfer_engine_->InsertReciveDeviceInfo("", decode_adp, total_device_num);

  transfer_engine_->Send(device_idx, layer_idx);

  // 验证元数据是否存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 验证元数据中的shared_block_num是否正确
  ASSERT_EQ(meta->shared_block_num, shared_block_num);

  // 验证创建的TransferTask数量是否正确
  ASSERT_EQ(meta->transfer_tasks_deque_.size(), 6);

  // 清理分配的内存
  for (size_t i = 0; i < prefill_atp; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试P端adp < D端adp时的发送功能（PREFILL角色）
TEST_F(TransferEngineTest, SendWithPrefillAdpLessRole) {
  int total_device_num = 2;
  int prefill_adp = 1;
  int decode_adp = 2;
  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 修改 pipeline_config_ 以使 ValidateLayerIndex 返回 true
  mock_env_->pipeline_config_.lower_layer_idx = 0;
  mock_env_->pipeline_config_.upper_layer_idx = 3;

  // 重新初始化引擎以应用新的配置
  transfer_engine_->Initialize(GroupRole::PREFILL);

  transfer_engine_->SetSelfDeviceInfo(prefill_adp, total_device_num);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  int prefill_atp = total_device_num / prefill_adp;
  gpu_blocks.resize(prefill_atp);
  for (size_t i = 0; i < prefill_atp; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 发送特定设备和层的数据
  int device_idx = 0;
  int layer_idx = 1;  // 确保在有效范围内
  transfer_engine_->InsertReciveDeviceInfo("", decode_adp, total_device_num);

  transfer_engine_->Send(device_idx, layer_idx);

  device_idx = 1;
  // 第二次发送
  transfer_engine_->Send(device_idx, layer_idx);

  // 验证元数据是否存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 验证元数据中的shared_block_num是否正确
  ASSERT_EQ(meta->shared_block_num, shared_block_num);

  // 验证创建的TransferTask数量是否正确
  ASSERT_EQ(meta->transfer_tasks_deque_.size(), 3);

  // 清理分配的内存
  for (size_t i = 0; i < prefill_atp; ++i) {
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
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建两个设备，每个设备有一个块
  gpu_blocks.resize(2);
  gpu_blocks[0].resize(1);
  gpu_blocks[1].resize(1);
  gpu_blocks[0][0] = malloc(4096);
  gpu_blocks[1][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);
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
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建一个设备，一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

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
  // 保证shared_block_num不大于块的数量
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

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
  // 保证shared_block_num不大于块的数量
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

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
  size_t shared_block_num = 2;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建两个设备，每个设备有3个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(3);
    for (size_t j = 0; j < 3; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

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

// ==================== 新增的Chunk传输功能测试 ====================

// 测试chunk传输配置
TEST_F(TransferEngineTest, ChunkTransferConfiguration) {
  // 设置不同的chunk大小
  mock_env_->SetTransferLayerChunkSize(3);

  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建一个设备，一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 验证元数据存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

// 测试chunk传输的发送逻辑
TEST_F(TransferEngineTest, ChunkTransferSendLogic) {
  // 设置chunk大小为2
  mock_env_->SetTransferLayerChunkSize(2);

  // 初始化引擎为PREFILL角色
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建一个设备，一个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 测试发送不同层的数据
  // 根据chunk逻辑，只有chunk的最后一层或模型的最后一层才会触发传输

  // layer_idx = 0 (offset = 0): 不是chunk末尾，不应该发送
  transfer_engine_->Send(0, 0);

  // layer_idx = 1 (offset = 1): 是第一个chunk的末尾，应该发送
  transfer_engine_->Send(0, 1);

  // layer_idx = 2 (offset = 2): 不是chunk末尾，不应该发送
  transfer_engine_->Send(0, 2);

  // layer_idx = 3 (offset = 3): 是最后一层，应该发送
  transfer_engine_->Send(0, 3);

  // 验证元数据仍然存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

// 测试chunk传输的接收任务数量计算
TEST_F(TransferEngineTest, ChunkTransferReceiveTaskCount) {
  // 设置chunk大小为3
  mock_env_->SetTransferLayerChunkSize(3);

  // 初始化引擎为DECODE角色
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0, 1};
  // 创建2个设备，每个设备2个块
  gpu_blocks.resize(2);
  for (size_t i = 0; i < 2; ++i) {
    gpu_blocks[i].resize(2);
    for (size_t j = 0; j < 2; ++j) {
      gpu_blocks[i][j] = malloc(4096);
    }
  }

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 检查接收是否完成
  // 对于4层（layer_num_=4）和chunk_size=3，应该有2个chunk：
  // chunk1: layers 0-2 (3层)
  // chunk2: layer 3 (1层)
  // 预期任务数 = block_num(2) * chunks_per_device(2) * device_num(2) = 8
  int result = transfer_engine_->IsRecvDone(request_id);

  // 由于没有实际接收到任务，应该返回-1
  ASSERT_EQ(result, -1);

  // 清理
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      free(meta->gpu_blocks[i][j]);
    }
  }
}

// 测试边界情况：chunk大小等于层数
TEST_F(TransferEngineTest, ChunkTransferBoundaryCase_ChunkSizeEqualsLayerNum) {
  // 设置chunk大小等于层数
  mock_env_->SetTransferLayerChunkSize(4);  // layer_num_ = 4

  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 只有最后一层应该触发传输
  transfer_engine_->Send(0, 0);  // 不应该发送
  transfer_engine_->Send(0, 1);  // 不应该发送
  transfer_engine_->Send(0, 2);  // 不应该发送
  transfer_engine_->Send(0, 3);  // 应该发送（最后一层）

  // 验证元数据存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

// 测试边界情况：chunk大小大于层数
TEST_F(TransferEngineTest, ChunkTransferBoundaryCase_ChunkSizeGreaterThanLayerNum) {
  // 设置chunk大小大于层数
  mock_env_->SetTransferLayerChunkSize(10);  // layer_num_ = 4

  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 只有最后一层应该触发传输
  transfer_engine_->Send(0, 0);  // 不应该发送
  transfer_engine_->Send(0, 1);  // 不应该发送
  transfer_engine_->Send(0, 2);  // 不应该发送
  transfer_engine_->Send(0, 3);  // 应该发送（最后一层）

  // 验证元数据存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

// 测试chunk大小为1的情况（默认行为）
TEST_F(TransferEngineTest, ChunkTransferDefaultBehavior) {
  // 使用默认chunk大小1
  mock_env_->SetTransferLayerChunkSize(1);

  // 初始化引擎
  transfer_engine_->Initialize(GroupRole::PREFILL);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 每一层都应该触发传输（因为每个chunk只有1层）
  transfer_engine_->Send(0, 0);  // 应该发送
  transfer_engine_->Send(0, 1);  // 应该发送
  transfer_engine_->Send(0, 2);  // 应该发送
  transfer_engine_->Send(0, 3);  // 应该发送

  // 验证元数据存在
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

// 测试CreateTransferTasksForDecodeNode的chunk逻辑
TEST_F(TransferEngineTest, ChunkTransferCreateTasksForDecodeNode) {
  // 设置chunk大小为2
  mock_env_->SetTransferLayerChunkSize(2);

  // 初始化引擎为DECODE角色
  transfer_engine_->Initialize(GroupRole::DECODE);

  // 创建测试数据
  int request_id = 123;
  size_t shared_block_num = 0;
  std::vector<std::vector<void*>> gpu_blocks;
  std::vector<int> kv_occupied_devices = {0};
  // 创建1个设备，1个块
  gpu_blocks.resize(1);
  gpu_blocks[0].resize(1);
  gpu_blocks[0][0] = malloc(4096);

  // 添加传输元数据
  transfer_engine_->AddTransferMeta("", request_id, shared_block_num, gpu_blocks, kv_occupied_devices);

  // 获取元数据并验证
  auto meta = transfer_engine_->GetTransferMeta(request_id);
  ASSERT_NE(meta, nullptr);

  // 清理
  free(meta->gpu_blocks[0][0]);
}

}  // namespace ksana_llm