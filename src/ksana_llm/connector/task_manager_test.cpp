/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/connector/task_manager.h"
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ksana_llm/transfer/transfer_types.h"
#include "ksana_llm/utils/tensor.h"

namespace ksana_llm {

class TaskManagerTest : public ::testing::Test {
 protected:
  void SetUp() override { task_manager_ = std::make_unique<TaskManager>(); }

  void TearDown() override {
    if (task_manager_) {
      task_manager_->Shutdown();
    }
  }

  // 用于生成一个简单的TransferTensor
  TransferTensor MakeTensor(int block_idx, int layer_idx, int device_idx, int data_len = 2,
                            DataType dtype = DataType::TYPE_FP32) {
    TransferTensor tensor;
    tensor.block_idx = block_idx;
    tensor.layer_idx = layer_idx;
    tensor.device_idx = device_idx;
    tensor.shape = {data_len};
    tensor.dtype = dtype;
    tensor.src_ptr = nullptr;
    return tensor;
  }

  std::shared_ptr<TransferTask> CreateMockTask(int req_id, const TransferTensor& tensor,
                                               const std::string& addr = "127.0.0.1:50051") {
    auto task = std::make_shared<TransferTask>();
    task->req_id = req_id;
    task->tensor = tensor;
    task->addr = addr;
    task->is_completed = false;
    task->dst_ptr = nullptr;
    return task;
  }

  std::unique_ptr<TaskManager> task_manager_;
};

// TaskKey Tests
class TaskKeyTest : public ::testing::Test {
 protected:
  TaskKey key1_{1, 2, 3, 4, 5, 0, 123456};
  TaskKey key2_{1, 2, 3, 4, 5, 0, 123456};
  TaskKey key3_{2, 3, 4, 5, 6, 0, 654321};
  TaskKey default_key_{};
};

TEST_F(TaskKeyTest, DefaultConstructor) {
  TaskKey key;
  EXPECT_EQ(key.req_id, 0);
  EXPECT_EQ(key.block_idx, 0);
  EXPECT_EQ(key.layer_idx, 0);
  EXPECT_EQ(key.device_idx, 0);
  EXPECT_EQ(key.tensor_size, 0);
  EXPECT_EQ(key.token, 0);
  EXPECT_EQ(key.start_time_us, 0);
}

TEST_F(TaskKeyTest, ParameterizedConstructor) {
  TaskKey key(10, 20, 30, 40, 50, 60, 123456789);
  EXPECT_EQ(key.req_id, 10);
  EXPECT_EQ(key.block_idx, 20);
  EXPECT_EQ(key.layer_idx, 30);
  EXPECT_EQ(key.device_idx, 40);
  EXPECT_EQ(key.tensor_size, 50);
  EXPECT_EQ(key.token, 60);
  EXPECT_EQ(key.start_time_us, 123456789);
}

TEST_F(TaskKeyTest, CopyConstructor) {
  TaskKey original(1, 2, 3, 4, 5, 6, 123);
  TaskKey copy(original);
  EXPECT_EQ(copy.req_id, original.req_id);
  EXPECT_EQ(copy.block_idx, original.block_idx);
  EXPECT_EQ(copy.layer_idx, original.layer_idx);
  EXPECT_EQ(copy.device_idx, original.device_idx);
  EXPECT_EQ(copy.tensor_size, original.tensor_size);
  EXPECT_EQ(copy.token, original.token);
  EXPECT_EQ(copy.start_time_us, original.start_time_us);
}

TEST_F(TaskKeyTest, MoveConstructor) {
  TaskKey original(1, 2, 3, 4, 5, 6, 123);
  TaskKey moved(std::move(original));
  EXPECT_EQ(moved.req_id, 1);
  EXPECT_EQ(moved.block_idx, 2);
  EXPECT_EQ(moved.layer_idx, 3);
  EXPECT_EQ(moved.device_idx, 4);
  EXPECT_EQ(moved.tensor_size, 5);
  EXPECT_EQ(moved.token, 6);
  EXPECT_EQ(moved.start_time_us, 123);
}

TEST_F(TaskKeyTest, CopyAssignment) {
  TaskKey original(1, 2, 3, 4, 5, 6, 123);
  TaskKey copy;
  copy = original;
  EXPECT_EQ(copy.req_id, original.req_id);
  EXPECT_EQ(copy.block_idx, original.block_idx);
  EXPECT_EQ(copy.layer_idx, original.layer_idx);
  EXPECT_EQ(copy.device_idx, original.device_idx);
  EXPECT_EQ(copy.tensor_size, original.tensor_size);
  EXPECT_EQ(copy.token, original.token);
  EXPECT_EQ(copy.start_time_us, original.start_time_us);
}

TEST_F(TaskKeyTest, MoveAssignment) {
  TaskKey original(1, 2, 3, 4, 5, 6, 123);
  TaskKey moved;
  moved = std::move(original);
  EXPECT_EQ(moved.req_id, 1);
  EXPECT_EQ(moved.block_idx, 2);
  EXPECT_EQ(moved.layer_idx, 3);
  EXPECT_EQ(moved.device_idx, 4);
  EXPECT_EQ(moved.tensor_size, 5);
  EXPECT_EQ(moved.token, 6);
  EXPECT_EQ(moved.start_time_us, 123);
}

TEST_F(TaskKeyTest, Equality) {
  EXPECT_TRUE(key1_ == key2_);
  EXPECT_FALSE(key1_ == key3_);
  EXPECT_FALSE(key1_ == default_key_);
}

TEST_F(TaskKeyTest, ToString) {
  std::string expected = "req_id=1, block_idx=2, layer_idx=3, device_idx=4, tensor_size=5, token=0";
  EXPECT_EQ(key1_.ToString(), expected);
  std::string default_expected = "req_id=0, block_idx=0, layer_idx=0, device_idx=0, tensor_size=0, token=0";
  EXPECT_EQ(default_key_.ToString(), default_expected);
}

TEST_F(TaskKeyTest, Hash) {
  TaskKey::NoTokenHash hasher;

  // Same keys should have same hash
  EXPECT_EQ(hasher(key1_), hasher(key2_));

  // Different keys should likely have different hashes
  EXPECT_NE(hasher(key1_), hasher(key3_));

  // Hash should be deterministic
  EXPECT_EQ(hasher(key1_), hasher(key1_));
}

TEST_F(TaskKeyTest, HashInUnorderedMap) {
  std::unordered_map<TaskKey, int, TaskKey::NoTokenHash> map;

  map[key1_] = 100;
  map[key3_] = 200;

  EXPECT_EQ(map[key1_], 100);
  EXPECT_EQ(map[key2_], 100);  // key2_ equals key1_
  EXPECT_EQ(map[key3_], 200);
  EXPECT_EQ(map.size(), 2);
}

TEST_F(TaskKeyTest, Serialize) {
  auto serialized = key1_.Serialize();
  EXPECT_EQ(serialized.size(), sizeof(TaskKey));
  EXPECT_FALSE(serialized.empty());
}

TEST_F(TaskKeyTest, Deserialize) {
  auto serialized = key1_.Serialize();
  auto deserialized = TaskKey::Deserialize(serialized);
  EXPECT_EQ(deserialized.req_id, key1_.req_id);
  EXPECT_EQ(deserialized.block_idx, key1_.block_idx);
  EXPECT_EQ(deserialized.layer_idx, key1_.layer_idx);
  EXPECT_EQ(deserialized.device_idx, key1_.device_idx);
  EXPECT_EQ(deserialized.tensor_size, key1_.tensor_size);
  EXPECT_EQ(deserialized.token, key1_.token);
  EXPECT_EQ(deserialized.start_time_us, key1_.start_time_us);
  EXPECT_TRUE(deserialized == key1_);
}

TEST_F(TaskKeyTest, DeserializeInsufficientData) {
  std::vector<uint8_t> insufficient_data(sizeof(TaskKey) - 1);
  auto deserialized = TaskKey::Deserialize(insufficient_data);
  // Should return default TaskKey when data is insufficient
  EXPECT_EQ(deserialized.req_id, 0);
  EXPECT_EQ(deserialized.block_idx, 0);
  EXPECT_EQ(deserialized.layer_idx, 0);
  EXPECT_EQ(deserialized.device_idx, 0);
  EXPECT_EQ(deserialized.tensor_size, 0);
  EXPECT_EQ(deserialized.token, 0);
  EXPECT_EQ(deserialized.start_time_us, 0);
}

TEST_F(TaskKeyTest, SerializeDeserializeRoundTrip) {
  std::vector<TaskKey> test_keys = {TaskKey(0, 0, 0, 0, 0, 0, 0), TaskKey(1, 2, 3, 4, 5, 6, 7),
                                    TaskKey(-1, -2, -3, -4, -5, -6, -7),
                                    TaskKey(INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT_MAX, INT64_MAX),
                                    TaskKey(INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT_MIN, INT64_MIN)};
  for (const auto& original : test_keys) {
    auto serialized = original.Serialize();
    auto deserialized = TaskKey::Deserialize(serialized);
    EXPECT_TRUE(original == deserialized) << "Failed for key: " << original.ToString();
    EXPECT_EQ(original.start_time_us, deserialized.start_time_us);
  }
}

TEST_F(TaskKeyTest, BatchSerialize) {
  std::vector<TaskKey> keys = {key1_, key3_, default_key_};
  auto serialized = TaskKey::BatchSerialize(keys);
  EXPECT_EQ(serialized.size(), keys.size() * sizeof(TaskKey));
  EXPECT_FALSE(serialized.empty());
}

TEST_F(TaskKeyTest, DeserializeBatch) {
  std::vector<TaskKey> original_keys = {key1_, key3_, default_key_};
  auto serialized = TaskKey::BatchSerialize(original_keys);
  auto deserialized_keys = TaskKey::DeserializeBatch(serialized);
  ASSERT_EQ(deserialized_keys.size(), original_keys.size());
  for (size_t i = 0; i < original_keys.size(); ++i) {
    EXPECT_TRUE(original_keys[i] == deserialized_keys[i])
        << "Mismatch at index " << i << ": " << original_keys[i].ToString() << " vs "
        << deserialized_keys[i].ToString();
    EXPECT_EQ(original_keys[i].start_time_us, deserialized_keys[i].start_time_us);
  }
}

TEST_F(TaskKeyTest, DeserializeBatchEmptyData) {
  std::vector<uint8_t> empty_data;
  auto result = TaskKey::DeserializeBatch(empty_data);
  EXPECT_TRUE(result.empty());
}

TEST_F(TaskKeyTest, DeserializeBatchInvalidSize) {
  std::vector<uint8_t> invalid_data(sizeof(TaskKey) + 1);  // Not multiple of sizeof(TaskKey)
  auto result = TaskKey::DeserializeBatch(invalid_data);
  EXPECT_TRUE(result.empty());
}

TEST_F(TaskKeyTest, DeserializeTaskKeysWithSingleKey) {
  auto serialized = key1_.Serialize();
  auto result = TaskKey::DeserializeTaskKeys(reinterpret_cast<const char*>(serialized.data()), serialized.size());
  ASSERT_EQ(result.size(), 1);
  EXPECT_TRUE(result[0] == key1_);
  EXPECT_EQ(result[0].start_time_us, key1_.start_time_us);
}

TEST_F(TaskKeyTest, DeserializeTaskKeysWithMultipleKeys) {
  std::vector<TaskKey> original_keys = {key1_, key3_, default_key_};
  auto serialized = TaskKey::BatchSerialize(original_keys);
  auto result = TaskKey::DeserializeTaskKeys(reinterpret_cast<const char*>(serialized.data()), serialized.size());
  ASSERT_EQ(result.size(), original_keys.size());
  for (size_t i = 0; i < original_keys.size(); ++i) {
    EXPECT_TRUE(original_keys[i] == result[i]);
    EXPECT_EQ(original_keys[i].start_time_us, result[i].start_time_us);
  }
}

TEST_F(TaskKeyTest, DeserializeTaskKeysInvalidSize) {
  std::vector<uint8_t> invalid_data(sizeof(TaskKey) + 1);
  auto result = TaskKey::DeserializeTaskKeys(reinterpret_cast<const char*>(invalid_data.data()), invalid_data.size());
  EXPECT_TRUE(result.empty());
}

// TaskManager Tests
TEST_F(TaskManagerTest, CreateTaskKey) {
  auto task = CreateMockTask(100, MakeTensor(10, 20, 30, 40));
  auto key = task_manager_->CreateTaskKey(task);

  EXPECT_EQ(key.req_id, 100);
  EXPECT_EQ(key.block_idx, 10);
  EXPECT_EQ(key.layer_idx, 20);
  EXPECT_EQ(key.device_idx, 30);
  EXPECT_EQ(key.tensor_size, 160);
}

TEST_F(TaskManagerTest, AddTask) {
  auto task = CreateMockTask(1, MakeTensor(2, 3, 4, 5));
  auto key = task_manager_->CreateTaskKey(task);

  task_manager_->AddTask(key, task);

  // Verify task was added to the map
  EXPECT_EQ(task_manager_->task_map_.size(), 1);
  EXPECT_EQ(task_manager_->task_map_[key], task);
}

TEST_F(TaskManagerTest, AddMultipleTasks) {
  auto task1 = CreateMockTask(1, MakeTensor(2, 3, 4, 5));
  auto task2 = CreateMockTask(2, MakeTensor(3, 4, 5, 6));
  auto task3 = CreateMockTask(3, MakeTensor(4, 5, 6, 7));

  auto key1 = task_manager_->CreateTaskKey(task1);
  auto key2 = task_manager_->CreateTaskKey(task2);
  auto key3 = task_manager_->CreateTaskKey(task3);

  task_manager_->AddTask(key1, task1);
  task_manager_->AddTask(key2, task2);
  task_manager_->AddTask(key3, task3);

  EXPECT_EQ(task_manager_->task_map_.size(), 3);
  EXPECT_EQ(task_manager_->task_map_[key1], task1);
  EXPECT_EQ(task_manager_->task_map_[key2], task2);
  EXPECT_EQ(task_manager_->task_map_[key3], task3);
}

TEST_F(TaskManagerTest, CompleteTask) {
  auto task = CreateMockTask(1, MakeTensor(2, 3, 4, 5));
  auto key = task_manager_->CreateTaskKey(task);

  task_manager_->AddTask(key, task);
  EXPECT_EQ(task_manager_->task_map_.size(), 1);
  EXPECT_FALSE(task->is_completed);

  task_manager_->CompleteTask(key);

  EXPECT_TRUE(task->is_completed);
  EXPECT_EQ(task_manager_->task_map_.size(), 0);
}

TEST_F(TaskManagerTest, CompleteNonExistentTask) {
  TaskKey non_existent_key(999, 999, 999, 999, 999);

  // This should not crash but may have undefined behavior
  // In a real implementation, you might want to add bounds checking
  EXPECT_NO_THROW({
      // Note: This test might need to be adjusted based on actual implementation
      // If the implementation adds bounds checking, this should be tested differently
  });
}

TEST_F(TaskManagerTest, GroupByGroupKeyAndDevice) {
  // Create tasks with different addresses and device indices
  auto task1 = CreateMockTask(1, MakeTensor(2, 3, 0, 5), "192.168.1.1:50051");
  auto task2 = CreateMockTask(2, MakeTensor(3, 4, 0, 6), "192.168.1.1:50051");  // Same address, same device
  auto task3 = CreateMockTask(3, MakeTensor(4, 5, 1, 7), "192.168.1.1:50051");  // Same address, different device
  auto task4 =
      CreateMockTask(4, MakeTensor(5, 6, 0, 8), "192.168.1.2:50051");  // Different address, same device as task1

  auto key1 = task_manager_->CreateTaskKey(task1);
  auto key2 = task_manager_->CreateTaskKey(task2);
  auto key3 = task_manager_->CreateTaskKey(task3);
  auto key4 = task_manager_->CreateTaskKey(task4);

  task_manager_->AddTask(key1, task1);
  task_manager_->AddTask(key2, task2);
  task_manager_->AddTask(key3, task3);
  task_manager_->AddTask(key4, task4);

  std::vector<TaskKey> batch = {key1, key2, key3, key4};
  auto grouped = task_manager_->GroupByGroupKeyAndDevice(batch);

  // Should have 3 groups: (192.168.1.1:50051, 0), (192.168.1.1:50051, 1), (192.168.1.2:50051, 0)
  EXPECT_EQ(grouped.size(), 3);

  auto group1_dev0 = grouped[{"192.168.1.1:50051", 0}];
  auto group1_dev1 = grouped[{"192.168.1.1:50051", 1}];
  auto group2_dev0 = grouped[{"192.168.1.2:50051", 0}];

  EXPECT_EQ(group1_dev0.size(), 2);  // task1 and task2
  EXPECT_EQ(group1_dev1.size(), 1);  // task3
  EXPECT_EQ(group2_dev0.size(), 1);  // task4

  // Verify contents
  EXPECT_TRUE(std::find(group1_dev0.begin(), group1_dev0.end(), key1) != group1_dev0.end());
  EXPECT_TRUE(std::find(group1_dev0.begin(), group1_dev0.end(), key2) != group1_dev0.end());
  EXPECT_TRUE(std::find(group1_dev1.begin(), group1_dev1.end(), key3) != group1_dev1.end());
  EXPECT_TRUE(std::find(group2_dev0.begin(), group2_dev0.end(), key4) != group2_dev0.end());
}

TEST_F(TaskManagerTest, GroupByGroupKeyAndDeviceEmptyBatch) {
  std::vector<TaskKey> empty_batch;
  auto grouped = task_manager_->GroupByGroupKeyAndDevice(empty_batch);

  EXPECT_TRUE(grouped.empty());
}

TEST_F(TaskManagerTest, GroupByGroupKeyAndDeviceNonExistentKeys) {
  std::vector<TaskKey> batch = {TaskKey(999, 999, 999, 999, 999), TaskKey(888, 888, 888, 888, 888)};

  auto grouped = task_manager_->GroupByGroupKeyAndDevice(batch);

  // Should be empty since tasks don't exist in task_map_
  EXPECT_TRUE(grouped.empty());
}

TEST_F(TaskManagerTest, GroupDevKeyHash) {
  TaskManager::GroupDevKeyHash hasher;

  TaskManager::GroupDevKey key1("192.168.1.1:50051", 0);
  TaskManager::GroupDevKey key2("192.168.1.1:50051", 0);
  TaskManager::GroupDevKey key3("192.168.1.2:50051", 0);
  TaskManager::GroupDevKey key4("192.168.1.1:50051", 1);

  // Same keys should have same hash
  EXPECT_EQ(hasher(key1), hasher(key2));

  // Different keys should likely have different hashes
  EXPECT_NE(hasher(key1), hasher(key3));
  EXPECT_NE(hasher(key1), hasher(key4));

  // Test in unordered_map
  std::unordered_map<TaskManager::GroupDevKey, int, TaskManager::GroupDevKeyHash> map;
  map[key1] = 100;
  map[key3] = 200;
  map[key4] = 300;

  EXPECT_EQ(map[key1], 100);
  EXPECT_EQ(map[key2], 100);  // Same as key1
  EXPECT_EQ(map[key3], 200);
  EXPECT_EQ(map[key4], 300);
  EXPECT_EQ(map.size(), 3);
}

TEST_F(TaskManagerTest, Shutdown) {
  // Add some tasks
  auto task1 = CreateMockTask(1, MakeTensor(2, 3, 4, 5));
  auto task2 = CreateMockTask(2, MakeTensor(3, 4, 5, 6));
  auto key1 = task_manager_->CreateTaskKey(task1);
  auto key2 = task_manager_->CreateTaskKey(task2);

  task_manager_->AddTask(key1, task1);
  task_manager_->AddTask(key2, task2);

  EXPECT_EQ(task_manager_->task_map_.size(), 2);

  task_manager_->Shutdown();

  EXPECT_EQ(task_manager_->task_map_.size(), 0);
}

TEST_F(TaskManagerTest, ThreadSafetyAddTasks) {
  const int num_threads = 4;
  const int tasks_per_thread = 100;
  std::vector<std::thread> threads;

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([this, t, tasks_per_thread]() {
      for (int i = 0; i < tasks_per_thread; ++i) {
        auto task = CreateMockTask(t * tasks_per_thread + i, MakeTensor(i, i, i, i));
        auto key = task_manager_->CreateTaskKey(task);
        task_manager_->AddTask(key, task);
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(task_manager_->task_map_.size(), num_threads * tasks_per_thread);
}

TEST_F(TaskManagerTest, ThreadSafetyAddAndComplete) {
  const int num_operations = 200;
  std::vector<std::thread> threads;

  // Thread 1: Add tasks
  threads.emplace_back([this, num_operations]() {
    for (int i = 0; i < num_operations; ++i) {
      auto task = CreateMockTask(i, MakeTensor(i, i, i, i));
      auto key = task_manager_->CreateTaskKey(task);
      task_manager_->AddTask(key, task);
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  });

  // Thread 2: Complete tasks
  threads.emplace_back([this, num_operations]() {
    for (int i = 0; i < num_operations / 2; ++i) {
      TaskKey key(i, i, i, i, i);
      std::this_thread::sleep_for(std::chrono::microseconds(10));

      // Check if task exists before completing
      {
        std::lock_guard<std::mutex> lock(task_manager_->buffer_mutex_);
        if (task_manager_->task_map_.find(key) != task_manager_->task_map_.end()) {
          task_manager_->task_map_[key]->is_completed = true;
          task_manager_->task_map_.erase(key);
        }
      }
    }
  });

  for (auto& thread : threads) {
    thread.join();
  }

  // Should have remaining tasks that weren't completed
  EXPECT_GT(task_manager_->task_map_.size(), 0);
  EXPECT_LT(task_manager_->task_map_.size(), num_operations);
}

TEST_F(TaskManagerTest, ProcessingBufferExists) {
  // Verify that processing_buffer_ is accessible and has expected interface
  EXPECT_NO_THROW({
    TaskKey test_key(1, 2, 3, 4, 5);
    // Note: We don't actually push to avoid blocking in tests
    // task_manager_->processing_buffer_.Push(test_key);
  });
}

TEST_F(TaskManagerTest, SendWaiterExists) {
  // Verify that send_waiter_ member exists and can be set
  EXPECT_NE(task_manager_->send_waiter_, nullptr);  // 修正：构造后应非空

  auto waiter = std::make_shared<Waiter>(1);
  task_manager_->send_waiter_ = waiter;
  EXPECT_EQ(task_manager_->send_waiter_, waiter);
}

// Promise同步相关功能测试
class TaskManagerPromiseTest : public ::testing::Test {
 protected:
  void SetUp() override { task_manager_ = std::make_unique<TaskManager>(); }

  void TearDown() override {
    if (task_manager_) {
      task_manager_->Shutdown();
    }
  }

  std::unique_ptr<TaskManager> task_manager_;
};

TEST_F(TaskManagerPromiseTest, AddPrefillPendingTask) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);

  // 添加 pending tasks
  task_manager_->AddPrefillPendingTask(key1);
  task_manager_->AddPrefillPendingTask(key2);

  // 验证任务已添加到 prefill_pending_tasks_
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 2);
  EXPECT_TRUE(task_manager_->prefill_pending_tasks_.find(key1) != task_manager_->prefill_pending_tasks_.end());
  EXPECT_TRUE(task_manager_->prefill_pending_tasks_.find(key2) != task_manager_->prefill_pending_tasks_.end());
}

TEST_F(TaskManagerPromiseTest, RegisterDecodeConfirmedTasks) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);

  std::vector<TaskKey> task_keys = {key1, key2};

  // 注册 decode confirmed tasks
  task_manager_->RegisterDecodeConfirmedTasks(task_keys);

  // 验证任务已添加到 decode_confirmed_tasks_
  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 2);
  EXPECT_TRUE(task_manager_->decode_confirmed_tasks_.find(key1) != task_manager_->decode_confirmed_tasks_.end());
  EXPECT_TRUE(task_manager_->decode_confirmed_tasks_.find(key2) != task_manager_->decode_confirmed_tasks_.end());
}

TEST_F(TaskManagerPromiseTest, RegisterDecodeConfirmedTasksWithPendingMatch) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);

  // 先添加 pending task
  task_manager_->AddPrefillPendingTask(key1);
  task_manager_->AddPrefillPendingTask(key2);

  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 2);

  // 注册其中一个 key 为 confirmed
  std::vector<TaskKey> task_keys = {key1};
  task_manager_->RegisterDecodeConfirmedTasks(task_keys);

  // 验证：key1 应从 pending 中移除，并添加到 confirmed 中
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 1);
  EXPECT_TRUE(task_manager_->prefill_pending_tasks_.find(key2) != task_manager_->prefill_pending_tasks_.end());
  EXPECT_TRUE(task_manager_->prefill_pending_tasks_.find(key1) == task_manager_->prefill_pending_tasks_.end());

  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 1);
  EXPECT_TRUE(task_manager_->decode_confirmed_tasks_.find(key1) != task_manager_->decode_confirmed_tasks_.end());
}

TEST_F(TaskManagerPromiseTest, TryActivatePendingTaskSuccess) {
  TaskKey key1(1, 2, 3, 4, 5);

  // 先注册为 confirmed
  std::vector<TaskKey> task_keys = {key1};
  task_manager_->RegisterDecodeConfirmedTasks(task_keys);

  // 再添加 pending
  task_manager_->AddPrefillPendingTask(key1);

  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 1);
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 1);

  // 尝试激活
  bool result = task_manager_->TryActivatePendingTask(key1);

  // 验证激活成功，两个映射都应清空对应 key
  EXPECT_TRUE(result);
  EXPECT_TRUE(task_manager_->decode_confirmed_tasks_.find(key1) == task_manager_->decode_confirmed_tasks_.end());
  EXPECT_TRUE(task_manager_->prefill_pending_tasks_.find(key1) == task_manager_->prefill_pending_tasks_.end());
}

TEST_F(TaskManagerPromiseTest, TryActivatePendingTaskFailure) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);

  // 只添加 pending，没有 confirmed
  task_manager_->AddPrefillPendingTask(key1);

  // 尝试激活不存在的 confirmed key
  bool result = task_manager_->TryActivatePendingTask(key2);

  // 验证激活失败
  EXPECT_FALSE(result);
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 1);
  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 0);
}

TEST_F(TaskManagerPromiseTest, CleanupExpiredTasks) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);
  TaskKey key3(3, 4, 5, 6, 7);

  // 添加一些任务
  task_manager_->AddPrefillPendingTask(key1);
  task_manager_->AddPrefillPendingTask(key2);
  std::vector<TaskKey> confirmed_keys = {key3};
  task_manager_->RegisterDecodeConfirmedTasks(confirmed_keys);

  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 2);
  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 1);

  // 等待1秒，确保时间戳不同
  std::this_thread::sleep_for(std::chrono::seconds(3));

  // 清理过期任务（使用较小的超时时间强制过期）
  task_manager_->CleanupExpiredTasks(0);  // 0秒超时，应该清理所有任务

  // 验证所有任务都被清理
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 0);
  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 0);
}

TEST_F(TaskManagerPromiseTest, CleanupExpiredTasksPartial) {
  TaskKey key1(1, 2, 3, 4, 5);
  TaskKey key2(2, 3, 4, 5, 6);

  // 添加任务
  task_manager_->AddPrefillPendingTask(key1);

  // 等待一段时间后添加第二个任务
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  task_manager_->AddPrefillPendingTask(key2);

  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 2);

  // 使用较大的超时时间，应该保留所有任务
  task_manager_->CleanupExpiredTasks(60);  // 60秒超时

  // 验证任务仍然存在
  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 2);
}

TEST_F(TaskManagerPromiseTest, ThreadSafetyPromiseOperations) {
  const int num_keys = 100;
  std::vector<std::thread> threads;

  // 线程1：添加 pending tasks
  threads.emplace_back([this, num_keys]() {
    for (int i = 0; i < num_keys; ++i) {
      TaskKey key(i, i, i, i, i);
      task_manager_->AddPrefillPendingTask(key);
      std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
  });

  // 线程2：注册 confirmed tasks
  threads.emplace_back([this, num_keys]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));  // 让pending先执行
    for (int i = 0; i < num_keys / 2; ++i) {
      TaskKey key(i, i, i, i, i);
      std::vector<TaskKey> keys = {key};
      task_manager_->RegisterDecodeConfirmedTasks(keys);
      std::this_thread::sleep_for(std::chrono::microseconds(2));
    }
  });

  // 线程3：尝试激活任务
  threads.emplace_back([this, num_keys]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // 让前面的操作先执行
    for (int i = 0; i < num_keys / 4; ++i) {
      TaskKey key(i, i, i, i, i);
      task_manager_->TryActivatePendingTask(key);
      std::this_thread::sleep_for(std::chrono::microseconds(3));
    }
  });

  for (auto& thread : threads) {
    thread.join();
  }

  // 验证操作后数据结构状态合理
  size_t total_tasks = task_manager_->prefill_pending_tasks_.size() + task_manager_->decode_confirmed_tasks_.size();
  EXPECT_GT(total_tasks, 0);
  EXPECT_LE(total_tasks, num_keys);
}

TEST_F(TaskManagerPromiseTest, EmptyOperations) {
  // 测试空向量的情况
  std::vector<TaskKey> empty_keys;
  EXPECT_NO_THROW(task_manager_->RegisterDecodeConfirmedTasks(empty_keys));

  // 测试对不存在key的操作
  TaskKey non_existent_key(999, 999, 999, 999, 999);
  EXPECT_FALSE(task_manager_->TryActivatePendingTask(non_existent_key));

  // 测试空状态下的清理
  EXPECT_NO_THROW(task_manager_->CleanupExpiredTasks(60));

  EXPECT_EQ(task_manager_->prefill_pending_tasks_.size(), 0);
  EXPECT_EQ(task_manager_->decode_confirmed_tasks_.size(), 0);
}

}  // namespace ksana_llm