/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/task_manager.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "ksana_llm/connector/task_key.h"
#include "ksana_llm/profiler/timer.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/singleton.h"
#include "ksana_llm/utils/waiter.h"

namespace ksana_llm {

namespace {

/// Mock函数用于模拟GetTypeSize
int MockGetTypeSize(DataType dtype) {
  switch (dtype) {
    case DataType::TYPE_FP32:
      return 4;
    case DataType::TYPE_FP16:
      return 2;
    case DataType::TYPE_INT32:
      return 4;
    case DataType::TYPE_INT64:
      return 8;
    default:
      return 1;
  }
}

/// 创建测试用的TransferTask
std::shared_ptr<TransferTask> CreateTestTask(int req_id, int block_idx, int layer_idx, int hash_device_id,
                                             int tensor_size = 0, int token = 0, bool is_skipped_task = false,
                                             const std::string& addr = "127.0.0.1:50051") {
  auto task = std::make_shared<TransferTask>();
  task->req_id = req_id;
  task->tensor.block_idx = block_idx;
  task->tensor.layer_idx = layer_idx;
  task->tensor.hash_device_id = hash_device_id;
  task->token = token;
  task->addr = addr;
  task->is_skipped_task = is_skipped_task;
  task->is_completed = false;
  task->tensor.dtype = DataType::TYPE_FP32;
  task->tensor.src_ptr = nullptr;
  task->dst_ptr = nullptr;

  if (tensor_size > 0) {
    // 创建一个简单的tensor shape，让GetElementNumber返回期望的元素数量
    int elements = tensor_size / MockGetTypeSize(task->tensor.dtype);
    task->tensor.shape = {elements, 1};
  } else {
    task->tensor.shape = {};
  }

  return task;
}

// 重载GetTypeSize函数以支持测试
int GetTypeSize(DataType dtype) { return MockGetTypeSize(dtype); }

}  // namespace

class TaskManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 每个测试创建独立的TaskManager实例
    task_manager_ = std::make_shared<TaskManager>(8, 1024, 2);  // 8个分片，便于测试
    ASSERT_NE(task_manager_, nullptr);
    ASSERT_NE(task_manager_->notification_waiter_, nullptr);
  }

  void TearDown() override {
    if (task_manager_) {
      task_manager_->Shutdown();
    }
    task_manager_.reset();
  }

  // 辅助方法：创建TaskKey
  TaskKey CreateTaskKey(int req_id, int block_idx, int layer_idx, int hash_device_id, int tensor_size = 0,
                        int token = 0, bool is_skipped_task = false) {
    int decode_device_id = -1;
    int decode_device_offset = -1;
    int prefill_device_id = -1;
    int prefill_device_offset = -1;
    return TaskKey(req_id, block_idx, layer_idx, hash_device_id, tensor_size, token, is_skipped_task, decode_device_id,
                   decode_device_offset, prefill_device_id, prefill_device_offset, ProfileTimer::GetCurrentTimeInUs());
  }

  // 辅助方法：创建任务批次
  std::vector<TaskKey> CreateTaskKeyBatch(int batch_size, int start_req_id = 0) {
    std::vector<TaskKey> batch;
    for (int i = 0; i < batch_size; ++i) {
      batch.push_back(CreateTaskKey(start_req_id + i, i, 0, i % 4, 100 * (i + 1)));
    }
    return batch;
  }

  // 辅助方法：添加任务到管理器
  void AddTasksToManager(const std::vector<TaskKey>& task_keys) {
    for (const auto& key : task_keys) {
      auto task = CreateTestTask(key.req_id, key.block_idx, key.layer_idx, key.hash_device_id, key.tensor_size,
                                 key.token, key.is_skipped_task);
      task_manager_->AddTask(key, task);
    }
  }

  // 辅助方法：验证分片分布
  void VerifyShardDistribution(const std::vector<TaskKey>& keys, int circular_bucket_num) {
    std::vector<int> shard_counts(circular_bucket_num, 0);
    for (const auto& key : keys) {
      size_t shard_idx = key.GetShardIndex(circular_bucket_num);
      EXPECT_LT(shard_idx, circular_bucket_num);
      shard_counts[shard_idx]++;
    }

    // 验证至少有一些分片被使用
    int used_shards = 0;
    for (int count : shard_counts) {
      if (count > 0) used_shards++;
    }
    EXPECT_GT(used_shards, 0);
  }

  std::shared_ptr<TaskManager> task_manager_;
};

// 基础功能测试
class TaskManagerBasicTest : public TaskManagerTest {};

// 分片测试
class TaskManagerShardTest : public TaskManagerTest {};

// 性能测试
class TaskManagerPerformanceTest : public TaskManagerTest {};

// Promise同步测试
class TaskManagerPromiseTest : public TaskManagerTest {};

// 通知机制测试
class TaskManagerNotificationTest : public TaskManagerTest {};

//=============================================================================
// TaskKey结构测试 - 扩展测试新功能
//=============================================================================

TEST_F(TaskManagerBasicTest, TaskKeyBasicOperations) {
  // 测试TaskKey构造函数
  TaskKey key1;
  EXPECT_EQ(key1.req_id, 0);
  EXPECT_EQ(key1.tensor_size, 0);
  EXPECT_EQ(key1.block_idx, 0);
  EXPECT_EQ(key1.layer_idx, 0);
  EXPECT_EQ(key1.hash_device_id, 0);
  EXPECT_EQ(key1.token, 0);
  EXPECT_FALSE(key1.is_skipped_task);
  EXPECT_EQ(key1.start_time_us, 0);

  TaskKey key2(123, 1, 2, 3, 1024, 456, true, 4, 5, 6, 7, 789);
  EXPECT_EQ(key2.req_id, 123);
  EXPECT_EQ(key2.tensor_size, 1024);
  EXPECT_EQ(key2.block_idx, 1);
  EXPECT_EQ(key2.layer_idx, 2);
  EXPECT_EQ(key2.hash_device_id, 3);
  EXPECT_EQ(key2.decode_device_id, 4);
  EXPECT_EQ(key2.decode_device_offset, 5);
  EXPECT_EQ(key2.prefill_device_id, 6);
  EXPECT_EQ(key2.prefill_device_offset, 7);
  EXPECT_EQ(key2.token, 456);
  EXPECT_TRUE(key2.is_skipped_task);
  EXPECT_EQ(key2.start_time_us, 789);
}

TEST_F(TaskManagerBasicTest, TaskKeyShardIndex) {
  TaskKey key1(5, 1, 2, 3, 1024);
  TaskKey key2(13, 1, 2, 3, 1024);  // 5 + 8 = 13
  TaskKey key3(21, 1, 2, 3, 1024);  // 5 + 8*2 = 21

  // 测试GetShardIndex方法
  EXPECT_EQ(key1.GetShardIndex(8), 5);
  EXPECT_EQ(key2.GetShardIndex(8), 5);  // 13 % 8 = 5
  EXPECT_EQ(key3.GetShardIndex(8), 5);  // 21 % 8 = 5

  // 测试边界情况
  TaskKey key_boundary(0, 0, 0, 0);
  EXPECT_EQ(key_boundary.GetShardIndex(8), 0);

  TaskKey key_max(7, 0, 0, 0);
  EXPECT_EQ(key_max.GetShardIndex(8), 7);
}

TEST_F(TaskManagerBasicTest, TaskKeyEquality) {
  TaskKey key1(123, 1, 2, 3, 1024, 456, false, 1, 0, 1, 0, 789);
  TaskKey key2(123, 1, 2, 3, 1024, 999, true, 2, 1, 2, 1, 888);  // 不同的token和timestamp

  // 比较操作符应该忽略token, is_skipped_task和timestamp
  EXPECT_TRUE(key1 == key2);

  TaskKey key3(124, 1, 2, 3, 1024, 456, false, 1, 0, 1, 0, 789);  // 不同的req_id
  EXPECT_FALSE(key1 == key3);
}

TEST_F(TaskManagerBasicTest, TaskKeyPriorityComparison) {
  TaskKey key1(123, 1, 2, 3, 1024, 456, false, 1, 0, 1, 0, 1000);
  TaskKey key2(123, 1, 2, 3, 1024, 456, false, 1, 0, 1, 0, 2000);

  // 较早的时间戳应该有更高的优先级
  // priority_queue 里，operator< 实现为 start_time_us > other.start_time_us
  // 所以 key2 < key1 为真（key2更晚，优先级更低）
  EXPECT_FALSE(key1 < key2);  // key1更早，优先级更高
  EXPECT_TRUE(key2 < key1);   // key2更晚，优先级更低
}

TEST_F(TaskManagerBasicTest, TaskKeyCreateFromTransferTask) {
  auto task = CreateTestTask(123, 1, 2, 3, 400, 456);  // 400字节 = 100个FP32元素

  TaskKey key = TaskKey::CreateFromTransferTask(task);

  EXPECT_EQ(key.req_id, 123);
  EXPECT_EQ(key.block_idx, 1);
  EXPECT_EQ(key.layer_idx, 2);
  EXPECT_EQ(key.hash_device_id, 3);
  EXPECT_EQ(key.tensor_size, 400);  // 100 elements * 4 bytes
  EXPECT_EQ(key.token, 456);
  EXPECT_GT(key.start_time_us, 0);
}

TEST_F(TaskManagerBasicTest, TaskKeyCreateFromNullTask) {
  std::shared_ptr<TransferTask> null_task = nullptr;

  // 创建TaskKey时传入nullptr不应该崩溃
  TaskKey key = TaskKey::CreateFromTransferTask(null_task);
  EXPECT_EQ(key.req_id, 0);
  EXPECT_EQ(key.tensor_size, 0);
}

//=============================================================================
// 序列化和反序列化测试 - 测试新的重载方法
//=============================================================================

TEST_F(TaskManagerBasicTest, TaskKeySerialization) {
  TaskKey original(123, 1, 2, 3, 1024, 456, true, 1, 0, 2, 1, 789);

  // 测试序列化
  std::vector<uint8_t> serialized = original.Serialize();
  EXPECT_EQ(serialized.size(), sizeof(TaskKey));

  // 测试反序列化
  TaskKey deserialized = TaskKey::Deserialize(serialized);
  EXPECT_EQ(original.req_id, deserialized.req_id);
  EXPECT_EQ(original.tensor_size, deserialized.tensor_size);
  EXPECT_EQ(original.block_idx, deserialized.block_idx);
  EXPECT_EQ(original.layer_idx, deserialized.layer_idx);
  EXPECT_EQ(original.hash_device_id, deserialized.hash_device_id);
  EXPECT_EQ(original.token, deserialized.token);
  EXPECT_EQ(original.is_skipped_task, deserialized.is_skipped_task);
  EXPECT_EQ(original.start_time_us, deserialized.start_time_us);
}

TEST_F(TaskManagerBasicTest, TaskKeyBatchSerializationVector) {
  std::vector<TaskKey> original_keys;
  for (int i = 0; i < 5; ++i) {
    original_keys.emplace_back(i, i + 1, i + 2, i + 3, (i + 1) * 100, i * 10);
  }

  // 测试批量序列化
  std::vector<uint8_t> serialized = TaskKey::BatchSerialize(original_keys);
  EXPECT_EQ(serialized.size(), original_keys.size() * sizeof(TaskKey));

  // 测试批量反序列化 - vector版本
  std::vector<TaskKey> deserialized = TaskKey::DeserializeBatch(serialized);
  EXPECT_EQ(deserialized.size(), original_keys.size());

  for (size_t i = 0; i < original_keys.size(); ++i) {
    EXPECT_EQ(original_keys[i].req_id, deserialized[i].req_id);
    EXPECT_EQ(original_keys[i].tensor_size, deserialized[i].tensor_size);
    EXPECT_EQ(original_keys[i].token, deserialized[i].token);
  }
}

TEST_F(TaskManagerBasicTest, TaskKeyBatchSerializationPointer) {
  std::vector<TaskKey> original_keys;
  for (int i = 0; i < 3; ++i) {
    original_keys.emplace_back(100 + i, i + 1, i + 2, i + 3, (i + 1) * 200, i * 20);
  }

  std::vector<uint8_t> serialized = TaskKey::BatchSerialize(original_keys);

  // 测试批量反序列化 - 指针版本
  std::vector<TaskKey> deserialized =
      TaskKey::DeserializeBatch(reinterpret_cast<const char*>(serialized.data()), serialized.size());

  EXPECT_EQ(deserialized.size(), original_keys.size());

  for (size_t i = 0; i < original_keys.size(); ++i) {
    EXPECT_EQ(original_keys[i].req_id, deserialized[i].req_id);
    EXPECT_EQ(original_keys[i].tensor_size, deserialized[i].tensor_size);
  }
}

TEST_F(TaskManagerBasicTest, TaskKeyDeserializationEdgeCases) {
  // 测试空数据
  std::vector<TaskKey> empty_result1 = TaskKey::DeserializeBatch(std::vector<uint8_t>());
  EXPECT_TRUE(empty_result1.empty());

  std::vector<TaskKey> empty_result2 = TaskKey::DeserializeBatch(nullptr, 0);
  EXPECT_TRUE(empty_result2.empty());

  // 测试无效大小
  std::vector<uint8_t> invalid_data(sizeof(TaskKey) + 1);  // 不是TaskKey大小的整数倍
  std::vector<TaskKey> invalid_result1 = TaskKey::DeserializeBatch(invalid_data);
  EXPECT_TRUE(invalid_result1.empty());

  std::vector<TaskKey> invalid_result2 =
      TaskKey::DeserializeBatch(reinterpret_cast<const char*>(invalid_data.data()), invalid_data.size());
  EXPECT_TRUE(invalid_result2.empty());
}

//=============================================================================
// 构造函数和基本功能测试
//=============================================================================

TEST_F(TaskManagerBasicTest, Constructor) {
  // 测试构造函数基本功能
  EXPECT_NE(task_manager_, nullptr);
  EXPECT_EQ(task_manager_->GetTaskCount(), 0);
  EXPECT_GE(task_manager_->GetLoadFactor(), 0.0f);
  EXPECT_NE(task_manager_->notification_waiter_, nullptr);
}

TEST_F(TaskManagerBasicTest, SingletonGetInstance) {
  // 测试单例模式
  auto instance1 = TaskManager::GetInstance(4, 512);
  auto instance2 = TaskManager::GetInstance(8, 1024);  // 不同参数

  // 单例应该返回相同的实例
  EXPECT_EQ(instance1.get(), instance2.get());
}

TEST_F(TaskManagerBasicTest, SetNotificationWaiter) {
  auto custom_waiter = std::make_shared<Waiter>(2);
  task_manager_->SetNotificationWaiter(custom_waiter);

  EXPECT_EQ(task_manager_->notification_waiter_, custom_waiter);
}

//=============================================================================
// 分片功能测试
//=============================================================================

TEST_F(TaskManagerShardTest, ShardDistribution) {
  const int num_tasks = 16;
  std::vector<TaskKey> keys = CreateTaskKeyBatch(num_tasks);

  // 验证分片分布
  VerifyShardDistribution(keys, 8);

  // 验证相同req_id的任务会分配到同一分片
  std::vector<TaskKey> same_req_keys;
  for (int i = 0; i < 5; ++i) {
    same_req_keys.push_back(CreateTaskKey(100, i, 0, 0, 100));  // 相同req_id
  }

  size_t first_shard = same_req_keys[0].GetShardIndex(8);
  for (const auto& key : same_req_keys) {
    EXPECT_EQ(key.GetShardIndex(8), first_shard);
  }
}

TEST_F(TaskManagerShardTest, ShardIsolation) {
  // 创建分布在不同分片的任务
  TaskKey key1(0, 0, 0, 0, 100);  // shard 0
  TaskKey key2(1, 0, 0, 0, 100);  // shard 1
  TaskKey key3(8, 0, 0, 0, 100);  // shard 0 (8 % 8 = 0)

  auto task1 = CreateTestTask(0, 0, 0, 0, 100);
  auto task2 = CreateTestTask(1, 0, 0, 0, 100);
  auto task3 = CreateTestTask(8, 0, 0, 0, 100);

  // 添加到不同分片
  task_manager_->AddTask(key1, task1);
  task_manager_->AddTask(key2, task2);
  task_manager_->AddTask(key3, task3);

  // 验证可以独立访问
  EXPECT_NE(task_manager_->GetTask(key1), nullptr);
  EXPECT_NE(task_manager_->GetTask(key2), nullptr);
  EXPECT_NE(task_manager_->GetTask(key3), nullptr);

  // 验证key1和key3在同一分片
  EXPECT_EQ(key1.GetShardIndex(8), key3.GetShardIndex(8));
  EXPECT_NE(key1.GetShardIndex(8), key2.GetShardIndex(8));
}

//=============================================================================
// 任务管理核心功能测试
//=============================================================================

TEST_F(TaskManagerBasicTest, AddAndGetTask) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);
  auto task = CreateTestTask(1, 0, 0, 0, 100);

  // 添加任务
  task_manager_->AddTask(key, task);
  EXPECT_EQ(task_manager_->GetTaskCount(), 1);

  // 获取任务
  auto retrieved_task = task_manager_->GetTask(key);
  EXPECT_NE(retrieved_task, nullptr);
  EXPECT_EQ(retrieved_task->req_id, task->req_id);
  EXPECT_EQ(retrieved_task->tensor.block_idx, task->tensor.block_idx);
}

TEST_F(TaskManagerBasicTest, GetNonExistentTask) {
  TaskKey key = CreateTaskKey(999, 0, 0, 0);
  auto task = task_manager_->GetTask(key);
  EXPECT_EQ(task, nullptr);
}

TEST_F(TaskManagerBasicTest, GetTasksBatch) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(5);
  AddTasksToManager(keys);

  // 测试批量获取存在的任务
  std::vector<std::shared_ptr<TransferTask>> tasks = task_manager_->GetTasksBatch(keys);
  EXPECT_EQ(tasks.size(), 5);

  for (size_t i = 0; i < tasks.size(); ++i) {
    EXPECT_NE(tasks[i], nullptr);
    EXPECT_EQ(tasks[i]->req_id, keys[i].req_id);
  }

  // 测试包含不存在任务的批次
  std::vector<TaskKey> mixed_keys = keys;
  mixed_keys.push_back(CreateTaskKey(999, 0, 0, 0));  // 不存在的任务

  std::vector<std::shared_ptr<TransferTask>> mixed_tasks = task_manager_->GetTasksBatch(mixed_keys);
  EXPECT_EQ(mixed_tasks.size(), 6);
  EXPECT_EQ(mixed_tasks[5], nullptr);  // 不存在的任务应该返回nullptr
}

TEST_F(TaskManagerBasicTest, GetTasksBatchParallel) {
  // 创建大批量任务以触发并行处理
  const int large_batch_size = 20;  // 大于 circular_bucket_num * 2
  std::vector<TaskKey> keys = CreateTaskKeyBatch(large_batch_size);
  AddTasksToManager(keys);

  auto tasks = task_manager_->GetTasksBatch(keys);
  EXPECT_EQ(tasks.size(), large_batch_size);

  for (size_t i = 0; i < tasks.size(); ++i) {
    EXPECT_NE(tasks[i], nullptr);
    EXPECT_EQ(tasks[i]->req_id, keys[i].req_id);
  }
}

TEST_F(TaskManagerBasicTest, CompleteTask) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);
  auto task = CreateTestTask(1, 0, 0, 0, 100);

  task_manager_->AddTask(key, task);
  EXPECT_EQ(task_manager_->GetTaskCount(), 1);

  // 完成任务
  task_manager_->CompleteTask(key);
  EXPECT_EQ(task_manager_->GetTaskCount(), 0);

  // 尝试获取已完成的任务应该返回nullptr
  auto completed_task = task_manager_->GetTask(key);
  EXPECT_EQ(completed_task, nullptr);
}

TEST_F(TaskManagerBasicTest, CompleteNonExistentTask) {
  TaskKey key = CreateTaskKey(999, 0, 0, 0);

  // 完成不存在的任务不应该崩溃
  EXPECT_NO_THROW(task_manager_->CompleteTask(key));
}

//=============================================================================
// 处理缓冲区操作测试
//=============================================================================

TEST_F(TaskManagerBasicTest, ProcessingBufferOperations) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(5);

  // 测试空缓冲区
  EXPECT_TRUE(task_manager_->IsProcessingBufferEmpty());
  EXPECT_EQ(task_manager_->GetProcessingBufferSize(), 0);

  // 添加任务到处理缓冲区
  for (const auto& key : keys) {
    task_manager_->PutProcessingBuffer(key);
  }

  EXPECT_FALSE(task_manager_->IsProcessingBufferEmpty());
  EXPECT_EQ(task_manager_->GetProcessingBufferSize(), 5);
}

TEST_F(TaskManagerBasicTest, ProcessingBufferGetSingle) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(3);

  // 添加任务
  for (const auto& key : keys) {
    task_manager_->PutProcessingBuffer(key);
  }

  // Round-robin获取任务
  std::vector<TaskKey> retrieved;
  for (int i = 0; i < 3; ++i) {
    TaskKey key = task_manager_->GetProcessingBuffer();
    EXPECT_EQ(key.req_id, i);  // id当前是自增
    retrieved.push_back(key);
  }

  // 缓冲区应该为空
  EXPECT_TRUE(task_manager_->IsProcessingBufferEmpty());

  // 再次获取应该返回默认TaskKey
  TaskKey empty_key = task_manager_->GetProcessingBuffer();
  EXPECT_EQ(empty_key.req_id, 0);
}

TEST_F(TaskManagerBasicTest, ProcessingBufferGetBatch) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(10);

  // 添加任务
  for (const auto& key : keys) {
    task_manager_->PutProcessingBuffer(key);
  }

  // 批量获取
  std::vector<TaskKey> batch = task_manager_->GetProcessingBufferBatch(5);
  EXPECT_EQ(batch.size(), 5);
  EXPECT_EQ(task_manager_->GetProcessingBufferSize(), 5);

  // 获取剩余所有任务
  std::vector<TaskKey> remaining = task_manager_->GetProcessingBufferBatch(10);
  EXPECT_EQ(remaining.size(), 5);
  EXPECT_TRUE(task_manager_->IsProcessingBufferEmpty());
}

TEST_F(TaskManagerBasicTest, ProcessingBufferGetBatchParallel) {
  // 测试大批量并行获取
  std::vector<TaskKey> keys = CreateTaskKeyBatch(24);  // 3倍于circular_bucket_num

  for (const auto& key : keys) {
    task_manager_->PutProcessingBuffer(key);
  }

  // 大批量获取应该触发并行处理
  std::vector<TaskKey> batch = task_manager_->GetProcessingBufferBatch(20);
  EXPECT_EQ(batch.size(), 20);
}

//=============================================================================
// CreateTaskKey方法测试
//=============================================================================

TEST_F(TaskManagerBasicTest, CreateTaskKeyDeprecated) {
  auto task = CreateTestTask(123, 1, 2, 3, 400, 456, true);

  // 测试deprecated方法
  TaskKey key = task_manager_->CreateTaskKey(task);

  EXPECT_EQ(key.req_id, 123);
  EXPECT_EQ(key.block_idx, 1);
  EXPECT_EQ(key.layer_idx, 2);
  EXPECT_EQ(key.hash_device_id, 3);
  EXPECT_EQ(key.tensor_size, 400);
  EXPECT_EQ(key.token, 456);
  EXPECT_TRUE(key.is_skipped_task);
  EXPECT_GT(key.start_time_us, 0);
}

//=============================================================================
// GroupByGroupKeyAndDevice方法测试
//=============================================================================

TEST_F(TaskManagerBasicTest, GroupByGroupKeyAndDevice) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(4);

  // 创建具有不同地址的任务
  auto task1 = CreateTestTask(0, 0, 0, 0, 100, 0, false, "group1:50051");
  auto task2 = CreateTestTask(1, 1, 0, 1, 200, 0, false, "group1:50051");
  auto task3 = CreateTestTask(2, 2, 0, 0, 300, 0, false, "group2:50052");
  auto task4 = CreateTestTask(3, 3, 0, 1, 400, 0, false, "group2:50052");

  // 更新TaskKey的device_idx以匹配任务
  keys[0].hash_device_id = 0;
  keys[0].decode_device_id = 0;
  keys[0].prefill_device_id = 1;

  keys[1].hash_device_id = 1;
  keys[1].decode_device_id = 0;
  keys[1].prefill_device_id = 1;

  keys[2].hash_device_id = 0;
  keys[2].decode_device_id = 0;
  keys[2].prefill_device_id = 1;

  keys[3].hash_device_id = 1;
  keys[3].decode_device_id = 0;
  keys[3].prefill_device_id = 1;

  task_manager_->AddTask(keys[0], task1);
  task_manager_->AddTask(keys[1], task2);
  task_manager_->AddTask(keys[2], task3);
  task_manager_->AddTask(keys[3], task4);

  bool is_prefill = true;
  // For prefill
  auto grouped = task_manager_->GroupByGroupKeyAndDevice(keys, is_prefill);

  // 应该有2个组
  EXPECT_EQ(grouped.size(), 2);

  // 检查分组结果
  TaskManager::GroupDevKey key1("group1:50051", {1, 0});
  TaskManager::GroupDevKey key2("group2:50052", {1, 0});

  EXPECT_EQ(grouped[key1].size(), 2);
  EXPECT_EQ(grouped[key2].size(), 2);

  // For decode
  grouped = task_manager_->GroupByGroupKeyAndDevice(keys, !is_prefill);

  EXPECT_EQ(grouped.size(), 2);

  key1 = {"group1:50051", {0, 1}};
  key2 = {"group2:50052", {0, 1}};

  EXPECT_EQ(grouped[key1].size(), 2);
  EXPECT_EQ(grouped[key2].size(), 2);
}

TEST_F(TaskManagerBasicTest, GroupByGroupKeyAndDeviceWithMissingTask) {
  std::vector<TaskKey> keys = CreateTaskKeyBatch(2);

  // 只添加一个任务
  auto task1 = CreateTestTask(0, 0, 0, 0, 100, 0, false, "group1:50051");
  task_manager_->AddTask(keys[0], task1);

  auto grouped = task_manager_->GroupByGroupKeyAndDevice(keys, true);

  // 应该只有一个组，因为第二个任务不存在
  EXPECT_EQ(grouped.size(), 1);
}

TEST_F(TaskManagerBasicTest, GroupByGroupKeyAndDeviceWithSkippedTask) {
  TaskKey key = CreateTaskKey(0, 0, 0, 0, 100, 0, true);
  std::vector<TaskKey> keys = {key};

  // 创建跳过的任务
  auto task1 = CreateTestTask(0, 0, 0, 0, 100, 0, true, "group1:50051");
  task_manager_->AddTask(key, task1);

  // Decode节点会在这个时间点结束跳过的任务
  auto grouped = task_manager_->GroupByGroupKeyAndDevice(keys, false);

  // 验证任务是否被结束
  auto task = task_manager_->GetTask(key);
  EXPECT_EQ(task, nullptr);
}

//=============================================================================
// 统计和性能监控测试
//=============================================================================

TEST_F(TaskManagerBasicTest, GetTaskCount) {
  EXPECT_EQ(task_manager_->GetTaskCount(), 0);

  std::vector<TaskKey> keys = CreateTaskKeyBatch(5);
  AddTasksToManager(keys);

  EXPECT_EQ(task_manager_->GetTaskCount(), 5);

  // 完成一些任务
  task_manager_->CompleteTask(keys[0]);
  task_manager_->CompleteTask(keys[1]);

  EXPECT_EQ(task_manager_->GetTaskCount(), 3);
}

TEST_F(TaskManagerBasicTest, GetLoadFactor) {
  float initial_load = task_manager_->GetLoadFactor();
  EXPECT_GE(initial_load, 0.0f);

  // 添加一些任务
  std::vector<TaskKey> keys = CreateTaskKeyBatch(10);
  AddTasksToManager(keys);

  float loaded_factor = task_manager_->GetLoadFactor();
  EXPECT_GT(loaded_factor, initial_load);
  EXPECT_LE(loaded_factor, 1.0f);
}

//=============================================================================
// 循环索引测试
//=============================================================================

TEST_F(TaskManagerBasicTest, CircularIndexDistribution) {
  // 测试不同req_id的任务分布到不同的circular map中
  std::vector<TaskKey> keys;
  for (int i = 0; i < 300; ++i) {  // 超过默认的256个batch size
    keys.push_back(CreateTaskKey(i, 0, 0, 0, 100));
  }

  AddTasksToManager(keys);

  // 验证所有任务都能正确检索
  for (const auto& key : keys) {
    auto task = task_manager_->GetTask(key);
    EXPECT_NE(task, nullptr);
    EXPECT_EQ(task->req_id, key.req_id);
  }

  EXPECT_EQ(task_manager_->GetTaskCount(), 300);
}

//=============================================================================
// Promise-based同步测试
//=============================================================================

TEST_F(TaskManagerPromiseTest, AddPrefillPendingTask) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);

  // 添加等待确认的任务
  task_manager_->AddPrefillPendingTask(key);

  // 尝试激活应该失败，因为还没有decode确认
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_FALSE(activated);
}

TEST_F(TaskManagerPromiseTest, RegisterDecodeConfirmedTasks) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);

  // 先添加等待确认的任务
  task_manager_->AddPrefillPendingTask(key);

  // 注册decode确认的任务
  std::vector<TaskKey> confirmed_keys = {key};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  // 现在激活应该成功
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_TRUE(activated);
}

TEST_F(TaskManagerPromiseTest, RegisterDecodeConfirmedTasksWithoutPending) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);

  // 直接注册decode确认的任务（没有预先的pending任务）
  std::vector<TaskKey> confirmed_keys = {key};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  // 尝试激活应该成功
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_TRUE(activated);
}

TEST_F(TaskManagerPromiseTest, TryActivatePendingTaskNotConfirmed) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);

  // 尝试激活未确认的任务
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_FALSE(activated);
}

TEST_F(TaskManagerPromiseTest, RegisterDecodeConfirmedTasksWithSkippingFlag) {
  // 设置PREFILL节点的key，没有跳过标志
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100, 0, false);
  // 设置DECODE节点传来的confirmed_key，带有跳过标志
  TaskKey confirmed_key = CreateTaskKey(1, 0, 0, 0, 100, 0, true);

  // 直接注册decode确认的任务
  std::vector<TaskKey> confirmed_keys = {confirmed_key};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  // 此时key的跳过标志应该为false
  EXPECT_FALSE(key.GetIsSkippedTaskFlag());

  // 尝试激活应该成功，且key的跳过标志应该被设置为true
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_TRUE(activated);
  EXPECT_TRUE(key.GetIsSkippedTaskFlag());

  // 验证任务是否被结束
  auto task = task_manager_->GetTask(key);
  EXPECT_EQ(task, nullptr);
}

TEST_F(TaskManagerPromiseTest, CleanupExpiredTasks) {
  TaskKey key1 = CreateTaskKey(1, 0, 0, 0, 100);
  TaskKey key2 = CreateTaskKey(2, 0, 0, 0, 200);

  // 添加一些pending和confirmed任务
  task_manager_->AddPrefillPendingTask(key1);
  std::vector<TaskKey> confirmed_keys = {key2};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  // 立即清理（超时时间为0秒）
  task_manager_->CleanupExpiredTasks(0);

  // 验证任务被清理后的行为
  bool activated1 = task_manager_->TryActivatePendingTask(key1);
  bool activated2 = task_manager_->TryActivatePendingTask(key2);

  // 由于任务已过期被清理，激活应该失败
  EXPECT_FALSE(activated1);
  EXPECT_FALSE(activated2);
}

TEST_F(TaskManagerPromiseTest, CleanupNonExpiredTasks) {
  TaskKey key = CreateTaskKey(1, 0, 0, 0, 100);

  // 添加pending任务
  task_manager_->AddPrefillPendingTask(key);

  // 使用很长的超时时间，任务不应该被清理
  task_manager_->CleanupExpiredTasks(3600);  // 1小时

  // 注册确认任务
  std::vector<TaskKey> confirmed_keys = {key};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  // 任务应该仍然能够被激活
  bool activated = task_manager_->TryActivatePendingTask(key);
  EXPECT_TRUE(activated);
}

//=============================================================================
// 多线程和并发测试
//=============================================================================

TEST_F(TaskManagerPromiseTest, ConcurrentTaskOperations) {
  const int num_tasks = 100;

  std::vector<std::thread> threads;
  std::vector<TaskKey> all_keys;

  // 准备任务键
  for (int i = 0; i < num_tasks; ++i) {
    all_keys.push_back(CreateTaskKey(i, 0, 0, 0, 100 + i));
  }

  // 线程1: 添加任务
  threads.emplace_back([this, &all_keys]() {
    for (const auto& key : all_keys) {
      auto task = CreateTestTask(key.req_id, key.block_idx, key.layer_idx, key.hash_device_id, key.tensor_size);
      task_manager_->AddTask(key, task);
    }
  });

  // 线程2: 获取任务
  threads.emplace_back([this, &all_keys]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 等待任务被添加
    for (const auto& key : all_keys) {
      auto task = task_manager_->GetTask(key);
      // task可能为nullptr如果还没被添加
    }
  });

  // 线程3: 添加pending任务
  threads.emplace_back([this, &all_keys]() {
    for (size_t i = 0; i < all_keys.size() / 2; ++i) {
      task_manager_->AddPrefillPendingTask(all_keys[i]);
    }
  });

  // 线程4: 注册confirmed任务
  threads.emplace_back([this, &all_keys]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    std::vector<TaskKey> half_keys(all_keys.begin(), all_keys.begin() + all_keys.size() / 2);
    task_manager_->RegisterDecodeConfirmedTasks(half_keys);
  });

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 验证最终状态
  EXPECT_EQ(task_manager_->GetTaskCount(), num_tasks);
}

//=============================================================================
// 边界条件和错误处理测试
//=============================================================================

TEST_F(TaskManagerBasicTest, EmptyOperations) {
  // 测试空批次操作
  std::vector<TaskKey> empty_keys;
  auto empty_tasks = task_manager_->GetTasksBatch(empty_keys);
  EXPECT_TRUE(empty_tasks.empty());

  auto empty_grouped = task_manager_->GroupByGroupKeyAndDevice(empty_keys, true);
  EXPECT_TRUE(empty_grouped.empty());

  // 测试空的confirmed任务
  task_manager_->RegisterDecodeConfirmedTasks(empty_keys);
  // 不应该崩溃
}

TEST_F(TaskManagerBasicTest, LargeTaskBatch) {
  const int large_batch_size = 1000;
  std::vector<TaskKey> large_batch = CreateTaskKeyBatch(large_batch_size);
  AddTasksToManager(large_batch);

  EXPECT_EQ(task_manager_->GetTaskCount(), large_batch_size);

  // 批量获取
  auto retrieved_tasks = task_manager_->GetTasksBatch(large_batch);
  EXPECT_EQ(retrieved_tasks.size(), large_batch_size);

  for (const auto& task : retrieved_tasks) {
    EXPECT_NE(task, nullptr);
  }
}

TEST_F(TaskManagerBasicTest, TasksWithSameReqIdDifferentIndices) {
  // 测试相同req_id但不同其他索引的任务
  TaskKey key1(1, 0, 0, 0, 100);
  TaskKey key2(1, 1, 0, 0, 200);  // 相同req_id，不同block_idx
  TaskKey key3(1, 0, 1, 0, 300);  // 相同req_id，不同layer_idx

  auto task1 = CreateTestTask(1, 0, 0, 0, 100);
  auto task2 = CreateTestTask(1, 1, 0, 0, 200);
  auto task3 = CreateTestTask(1, 0, 1, 0, 300);

  task_manager_->AddTask(key1, task1);
  task_manager_->AddTask(key2, task2);
  task_manager_->AddTask(key3, task3);

  EXPECT_EQ(task_manager_->GetTaskCount(), 3);

  // 验证每个任务都能独立检索
  auto retrieved1 = task_manager_->GetTask(key1);
  auto retrieved2 = task_manager_->GetTask(key2);
  auto retrieved3 = task_manager_->GetTask(key3);

  EXPECT_NE(retrieved1, nullptr);
  EXPECT_NE(retrieved2, nullptr);
  EXPECT_NE(retrieved3, nullptr);

  EXPECT_EQ(retrieved1->tensor.block_idx, 0);
  EXPECT_EQ(retrieved2->tensor.block_idx, 1);
  EXPECT_EQ(retrieved3->tensor.layer_idx, 1);
}

//=============================================================================
// Shutdown测试
//=============================================================================

TEST_F(TaskManagerBasicTest, Shutdown) {
  // 添加一些任务
  std::vector<TaskKey> keys = CreateTaskKeyBatch(5);
  AddTasksToManager(keys);

  // 添加一些promise任务
  task_manager_->AddPrefillPendingTask(keys[0]);
  task_manager_->RegisterDecodeConfirmedTasks({keys[1]});

  EXPECT_EQ(task_manager_->GetTaskCount(), 5);

  // 执行shutdown
  task_manager_->Shutdown();

  // 验证所有状态被清理
  EXPECT_EQ(task_manager_->GetTaskCount(), 0);

  // 尝试获取任务应该返回nullptr
  for (const auto& key : keys) {
    auto task = task_manager_->GetTask(key);
    EXPECT_EQ(task, nullptr);
  }
}

//=============================================================================
// 性能测试
//=============================================================================

TEST_F(TaskManagerPerformanceTest, HighVolumeTaskOperations) {
  const int num_operations = 10000;
  auto start_time = std::chrono::high_resolution_clock::now();

  // 大量添加任务
  for (int i = 0; i < num_operations; ++i) {
    TaskKey key = CreateTaskKey(i, i % 10, i % 5, i % 3, 100 + i);
    auto task = CreateTestTask(i, i % 10, i % 5, i % 3, 100 + i);
    task_manager_->AddTask(key, task);
  }

  auto add_time = std::chrono::high_resolution_clock::now();

  // 大量检索任务
  int successful_retrievals = 0;
  for (int i = 0; i < num_operations; ++i) {
    TaskKey key = CreateTaskKey(i, i % 10, i % 5, i % 3, 100 + i);
    auto task = task_manager_->GetTask(key);
    if (task != nullptr) {
      successful_retrievals++;
    }
  }

  auto retrieve_time = std::chrono::high_resolution_clock::now();

  EXPECT_EQ(successful_retrievals, num_operations);
  EXPECT_EQ(task_manager_->GetTaskCount(), num_operations);

  // 验证性能（这些时间应该是合理的）
  auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(add_time - start_time);
  auto retrieve_duration = std::chrono::duration_cast<std::chrono::milliseconds>(retrieve_time - add_time);

  // 打印性能信息（在实际测试中可能需要移除）
  std::cout << "Added " << num_operations << " tasks in " << add_duration.count() << " ms" << std::endl;
  std::cout << "Retrieved " << num_operations << " tasks in " << retrieve_duration.count() << " ms" << std::endl;

  // 基本的性能断言（根据实际性能调整）
  EXPECT_LT(add_duration.count(), 1000);       // 添加应该在1秒内完成
  EXPECT_LT(retrieve_duration.count(), 1000);  // 检索应该在1秒内完成
}

TEST_F(TaskManagerPerformanceTest, LoadFactorUnderStress) {
  const int stress_tasks = 5000;

  float initial_load = task_manager_->GetLoadFactor();

  // 添加大量任务
  for (int i = 0; i < stress_tasks; ++i) {
    TaskKey key = CreateTaskKey(i, 0, 0, 0, 100);
    auto task = CreateTestTask(i, 0, 0, 0, 100);
    task_manager_->AddTask(key, task);
  }

  float stress_load = task_manager_->GetLoadFactor();

  // 负载因子应该增加但保持合理
  EXPECT_GT(stress_load, initial_load);
  EXPECT_LT(stress_load, 10.0f);  // 不应该过高

  std::cout << "Load factor under stress: " << stress_load << std::endl;
}
}  // namespace ksana_llm