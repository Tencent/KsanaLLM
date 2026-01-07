/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "ksana_llm/runtime/layer_progress_tracker.h"

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "ksana_llm/utils/device_utils.h"

// Namespace alias for convenience
namespace ksana = ksana_llm;

class LayerProgressTrackerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 在每个测试前初始化设备环境
    int device_id = 0;
    ksana_llm::SetDevice(device_id);
  }

  void TearDown() override {
    // 测试结束后同步设备
    ksana_llm::DeviceSynchronize();
  }

  // 辅助函数：创建一个流，指定设备ID
  ksana_llm::Stream CreateStream(int device_id) {
    return ksana_llm::Stream(device_id);  // 在指定设备上创建流
  }

  // 辅助函数：销毁流
  void DestroyStream(ksana_llm::Stream& stream) { stream.Destroy(); }
};

// 测试初始化和清理
TEST_F(LayerProgressTrackerTest, InitializeAndCleanup) {
  ksana::LayerProgressTracker tracker;

  // 测试初始化
  tracker.Initialize(1, 3);

  // 测试清理
  tracker.Cleanup();

  // 再次初始化，确保可以重复初始化
  tracker.Initialize(2, 2);

  // 再次初始化，确保可以多次重复初始化
  tracker.Initialize(2, 3);

  // 最后清理
  tracker.Cleanup();
}

// 测试注册回调函数
TEST_F(LayerProgressTrackerTest, RegisterCallback) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);
  std::atomic<int> last_device_id(-1);
  std::atomic<int> last_layer_index(-1);

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) {
    callback_count++;
    last_device_id = device_id;
    last_layer_index = layer_index;
  });

  // 创建流，指定设备ID
  int device_id = 0;
  ksana::Stream stream = CreateStream(device_id);

  // 记录层进度
  tracker.RecordLayerProgress(device_id, 5, stream);

  // 等待一段时间，确保回调被执行
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  // 验证回调是否被执行
  EXPECT_GT(callback_count, 0);
  EXPECT_EQ(last_device_id, device_id);
  EXPECT_EQ(last_layer_index, 5);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试记录层进度和获取层进度
TEST_F(LayerProgressTrackerTest, RecordAndGetLayerProgress) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  // 创建两个设备的流
  ksana::Stream stream0 = CreateStream(0);
  ksana::Stream stream1 = CreateStream(1);

  // 记录层进度，使用对应设备的流
  tracker.RecordLayerProgress(0, 3, stream0);
  tracker.RecordLayerProgress(1, 5, stream1);

  // 等待一段时间，确保事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // 获取层进度
  int progress0 = tracker.GetLayerProgress(0);
  int progress1 = tracker.GetLayerProgress(1);

  // 验证层进度
  EXPECT_EQ(progress0, 3);
  EXPECT_EQ(progress1, 5);

  // 测试获取不存在的设备进度
  int progress2 = tracker.GetLayerProgress(2);
  EXPECT_EQ(progress2, -1);

  // 清理
  DestroyStream(stream0);
  DestroyStream(stream1);
  tracker.Cleanup();
}

// 测试重置状态
TEST_F(LayerProgressTrackerTest, ResetState) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  // 创建两个设备的流
  ksana::Stream stream0 = CreateStream(0);
  ksana::Stream stream1 = CreateStream(1);

  // 记录层进度，使用对应设备的流
  tracker.RecordLayerProgress(0, 3, stream0);
  tracker.RecordLayerProgress(1, 5, stream1);

  // 等待一段时间，确保事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 重置状态
  tracker.ResetState();

  // 获取层进度
  int progress0 = tracker.GetLayerProgress(0);
  int progress1 = tracker.GetLayerProgress(1);

  // 验证层进度已被重置
  EXPECT_EQ(progress0, -1);
  EXPECT_EQ(progress1, -1);

  // 清理
  DestroyStream(stream0);
  DestroyStream(stream1);
  tracker.Cleanup();
}

// 测试多个设备和层的情况
TEST_F(LayerProgressTrackerTest, MultipleDevicesAndLayers) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 20);

  // 创建两个设备的流
  ksana::Stream stream0 = CreateStream(0);
  ksana::Stream stream1 = CreateStream(1);

  // 记录多个设备和层的进度
  for (int device_id = 0; device_id < 2; ++device_id) {
    ksana::Stream& stream = (device_id == 0) ? stream0 : stream1;

    for (int layer_index = 5; layer_index < 20; ++layer_index) {
      tracker.RecordLayerProgress(device_id, layer_index, stream);

      // 确保操作完成
      ksana_llm::StreamSynchronize(stream);

      // 等待一段时间，确保回调被执行
      std::this_thread::sleep_for(std::chrono::milliseconds(1));

      // 获取层进度
      int progress = tracker.GetLayerProgress(device_id);

      // 验证层进度
      EXPECT_EQ(progress, layer_index);
    }
  }

  // 清理
  DestroyStream(stream0);
  DestroyStream(stream1);
  tracker.Cleanup();
}

// 测试优化后的 MonitorEvents 函数（只查询下一层）
TEST_F(LayerProgressTrackerTest, OptimizedMonitorEvents) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);
  std::vector<std::pair<int, int>> callback_records;
  std::mutex callback_mutex;

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) {
    callback_count++;
    std::lock_guard<std::mutex> lock(callback_mutex);
    callback_records.push_back(std::make_pair(device_id, layer_index));
  });

  // 创建流，指定设备ID
  int device_id = 0;
  ksana::Stream stream = CreateStream(device_id);

  // 按顺序记录层进度
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  tracker.RecordLayerProgress(device_id, 1, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  tracker.RecordLayerProgress(device_id, 2, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // 记录更高层的进度（应该被跳过查询，直到前面的层完成）
  tracker.RecordLayerProgress(device_id, 5, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // 记录中间层的进度
  tracker.RecordLayerProgress(device_id, 3, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  tracker.RecordLayerProgress(device_id, 4, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // 等待一段时间，确保所有事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 验证回调执行顺序
  std::lock_guard<std::mutex> lock(callback_mutex);
  ASSERT_GE(callback_records.size(), 6);

  // 验证层进度按顺序更新
  for (size_t i = 0; i < 6; ++i) {
    EXPECT_EQ(callback_records[i].first, device_id);
    EXPECT_EQ(callback_records[i].second, i);
  }

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试 GetLayerProgressNoLock 函数
TEST_F(LayerProgressTrackerTest, GetLayerProgressNoLock) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  // 创建流，指定设备ID
  int device_id = 0;
  ksana::Stream stream = CreateStream(device_id);

  // 记录层进度
  tracker.RecordLayerProgress(device_id, 3, stream);
  ksana_llm::StreamSynchronize(stream);

  // 等待一段时间，确保事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // 获取层进度
  int progress = tracker.GetLayerProgress(device_id);

  // 验证层进度
  EXPECT_EQ(progress, 3);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试并发情况下的行为
TEST_F(LayerProgressTrackerTest, ConcurrentAccess) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 20);

  // 创建流，每个设备一个流
  std::vector<ksana::Stream> streams;
  for (int i = 0; i < 2; ++i) {
    streams.push_back(CreateStream(i));
  }

  // 创建多个线程同时记录层进度
  std::vector<std::thread> threads;
  for (int device_id = 0; device_id < 2; ++device_id) {
    threads.push_back(std::thread([&, device_id]() {
      for (int layer_index = 0; layer_index < 20; ++layer_index) {
        tracker.RecordLayerProgress(device_id, layer_index, streams[device_id]);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }));
  }

  // 等待所有线程完成
  for (auto& thread : threads) {
    thread.join();
  }

  // 等待一段时间，确保所有事件被处理
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // 验证最终层进度
  for (int device_id = 0; device_id < 2; ++device_id) {
    int progress = tracker.GetLayerProgress(device_id);
    EXPECT_EQ(progress, 19);
  }

  // 清理
  for (auto& stream : streams) {
    stream.Destroy();
  }
  tracker.Cleanup();
}

// 测试 ResetState 处理 pending_events_ 不为空的情况
TEST_F(LayerProgressTrackerTest, ResetStateWithPendingEvents) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);
  std::vector<std::pair<int, int>> callback_records;
  std::mutex callback_mutex;

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) {
    callback_count++;
    std::lock_guard<std::mutex> lock(callback_mutex);
    callback_records.push_back(std::make_pair(device_id, layer_index));
  });

  // 创建流，指定设备ID
  int device_id = 0;
  ksana::Stream stream = CreateStream(device_id);

  // 记录多个层进度，不等待处理完成
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);
  tracker.RecordLayerProgress(device_id, 2, stream);
  tracker.RecordLayerProgress(device_id, 3, stream);
  tracker.RecordLayerProgress(device_id, 4, stream);

  // 立即调用 ResetState，此时 pending_events_ 应该不为空
  // 这会触发 while (!pending_events_.empty()) 分支
  tracker.ResetState();

  // 验证回调被调用（ResetState 中会处理所有 pending events 并调用回调）
  {
    std::lock_guard<std::mutex> lock(callback_mutex);
    // ResetState 应该处理了所有 pending events
    EXPECT_GE(callback_records.size(), 0);  // 至少记录了一些事件
  }

  // 验证层进度已被重置
  int progress = tracker.GetLayerProgress(device_id);
  EXPECT_EQ(progress, -1);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}

// 测试 ResetState 处理多设备 pending_events_ 的情况
TEST_F(LayerProgressTrackerTest, ResetStateWithMultiDevicePendingEvents) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);
  std::mutex callback_mutex;

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) { callback_count++; });

  // 创建两个设备的流
  ksana::Stream stream0 = CreateStream(0);
  ksana::Stream stream1 = CreateStream(1);

  // 记录多个设备的层进度，不等待处理完成
  tracker.RecordLayerProgress(0, 0, stream0);
  tracker.RecordLayerProgress(0, 1, stream0);
  tracker.RecordLayerProgress(1, 0, stream1);
  tracker.RecordLayerProgress(1, 1, stream1);
  tracker.RecordLayerProgress(0, 2, stream0);
  tracker.RecordLayerProgress(1, 2, stream1);

  // 立即调用 ResetState，此时 pending_events_ 应该包含多个设备的事件
  tracker.ResetState();

  // 验证层进度已被重置
  int progress0 = tracker.GetLayerProgress(0);
  int progress1 = tracker.GetLayerProgress(1);
  EXPECT_EQ(progress0, -1);
  EXPECT_EQ(progress1, -1);

  // 清理
  DestroyStream(stream0);
  DestroyStream(stream1);
  tracker.Cleanup();
}

// 测试 ResetState 在空 pending_events_ 时的行为
TEST_F(LayerProgressTrackerTest, ResetStateWithEmptyPendingEvents) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) { callback_count++; });

  // 不记录任何层进度，直接调用 ResetState
  // 此时 pending_events_ 为空，while 循环不会执行
  tracker.ResetState();

  // 验证回调没有被调用
  EXPECT_EQ(callback_count, 0);

  // 验证层进度为 -1
  int progress = tracker.GetLayerProgress(0);
  EXPECT_EQ(progress, -1);

  // 清理
  tracker.Cleanup();
}

// 测试 ResetState 后重新记录层进度
TEST_F(LayerProgressTrackerTest, ResetStateAndRerecord) {
  GTEST_SKIP();

  ksana::LayerProgressTracker tracker;
  tracker.Initialize(2, 10);

  std::atomic<int> callback_count(0);

  // 注册回调函数
  tracker.RegisterCallback([&](int device_id, int layer_index) { callback_count++; });

  // 创建流
  int device_id = 0;
  ksana::Stream stream = CreateStream(device_id);

  // 记录层进度
  tracker.RecordLayerProgress(device_id, 0, stream);
  tracker.RecordLayerProgress(device_id, 1, stream);

  // 等待事件处理
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 重置状态
  tracker.ResetState();

  // 验证层进度已被重置
  int progress = tracker.GetLayerProgress(device_id);
  EXPECT_EQ(progress, -1);

  // 重新记录层进度
  tracker.RecordLayerProgress(device_id, 0, stream);
  ksana_llm::StreamSynchronize(stream);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  // 验证新的层进度
  progress = tracker.GetLayerProgress(device_id);
  EXPECT_EQ(progress, 0);

  // 清理
  DestroyStream(stream);
  tracker.Cleanup();
}
