/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/runtime/layer_progress_tracker.h"

#include <string.h>

namespace ksana_llm {

// The current LayerProgressTracker may cause deadlocks. It will be temporarily disabled via an environment variable
// until the bug is resolved in a subsequent update.
const char* const enable_tracker_env = std::getenv("ENABLE_LAYER_TRACKER");
const bool kEnableTracker = (enable_tracker_env != nullptr && strcmp(enable_tracker_env, "1") == 0);

LayerProgressTracker::LayerProgressTracker() : monitor_thread_running_(false) {}

LayerProgressTracker::~LayerProgressTracker() { Cleanup(); }

void LayerProgressTracker::Initialize(int max_devices, int max_layers) {
  if (!kEnableTracker) {
    return;
  }

  Cleanup();
  std::lock_guard<std::mutex> lock(mutex_);

  // 为每个设备的每一层创建 event
  for (int device_id = 0; device_id < max_devices; ++device_id) {
    SetDevice(device_id);
    for (int layer_index = 0; layer_index < max_layers; ++layer_index) {
      EventInfo event_info;
      EventCreateWithFlags(&event_info.event, EVENT_DISABLE_TIMING);
      event_info.device_id = device_id;
      event_info.layer_index = layer_index;
      event_info.processed = true;  // 初始状态为已处理

      events_map_[std::make_pair(device_id, layer_index)] = event_info;
    }
  }

  // 启动监控线程
  if (!monitor_thread_running_) {
    monitor_thread_running_ = true;
    monitor_thread_ = std::thread(&LayerProgressTracker::MonitorEvents, this);
  }
}

void LayerProgressTracker::Cleanup() {
  if (!kEnableTracker) {
    return;
  }

  // 先停止监控线程，避免死锁
  if (monitor_thread_running_) {
    monitor_thread_running_ = false;  // 这是一个简单的布尔值赋值，是原子操作
    if (monitor_thread_.joinable()) {
      monitor_thread_.join();  // 等待线程结束
    }
  }

  // 然后获取锁进行其他清理操作
  std::lock_guard<std::mutex> lock(mutex_);

  // 销毁所有 event
  for (auto& pair : events_map_) {
    EventDestroy(pair.second.event);
  }
  events_map_.clear();

  // 清空优先队列
  while (!pending_events_.empty()) {
    pending_events_.pop();
  }
}

void LayerProgressTracker::RegisterCallback(LayerProgressCallback callback) {
  if (!kEnableTracker) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  callbacks_.push_back(callback);
}

void LayerProgressTracker::RecordLayerProgress(int device_id, int layer_index, Stream& stream) {
  if (!kEnableTracker) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  auto key = std::make_pair(device_id, layer_index);
  auto it = events_map_.find(key);
  if (it != events_map_.end()) {
    // 记录 event
    EventRecord(it->second.event, stream);
    it->second.processed = false;  // 标记为未处理

    // 将事件添加到优先队列中
    pending_events_.push(it->second);
  }
}

int LayerProgressTracker::GetLayerProgress(int device_id) {
  if (!kEnableTracker) {
    return 0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  return GetLayerProgressNoLock(device_id);
}

// 内部辅助函数，不获取锁，用于在已持有锁的情况下获取层进度
int LayerProgressTracker::GetLayerProgressNoLock(int device_id) const {
  if (!kEnableTracker) {
    return 0;
  }

  auto it = device_layer_progress_.find(device_id);
  if (it != device_layer_progress_.end()) {
    return it->second;
  }
  return -1;  // 表示未找到该设备的进度
}

void LayerProgressTracker::MonitorEvents() {
  if (!kEnableTracker) {
    return;
  }

  while (monitor_thread_running_) {
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // 使用临时队列存储未完成的事件
      std::priority_queue<EventInfo> temp_queue;

      // 处理优先队列中的事件
      while (!pending_events_.empty()) {
        EventInfo event_info = pending_events_.top();
        pending_events_.pop();

        // 如果事件未处理，检查是否需要查询
        if (!event_info.processed) {
          // 获取该设备当前已完成的最大层索引
          int current_progress_layer_index = GetLayerProgressNoLock(event_info.device_id);

          // 只处理下一层的事件，跳过更高层的事件
          if (current_progress_layer_index != -1 && event_info.layer_index > current_progress_layer_index + 1) {
            // 该层比当前需要处理的下一层大，不需要查询，直接放回临时队列
            temp_queue.push(event_info);
            continue;
          }

          // 查询事件状态
          bool event_completed = EventQuery(event_info.event);
          if (event_completed) {
            // event 已完成，更新进度并通知
            device_layer_progress_[event_info.device_id] = event_info.layer_index;
            event_info.processed = true;

            // 调用所有注册的回调函数
            for (const auto& callback : callbacks_) {
              callback(event_info.device_id, event_info.layer_index);
            }

            // 更新 events_map_ 中的状态
            auto key = std::make_pair(event_info.device_id, event_info.layer_index);
            auto it = events_map_.find(key);
            if (it != events_map_.end()) {
              it->second.processed = true;
            }
          } else {
            // 事件未完成，放回临时队列
            temp_queue.push(event_info);
          }
        }
      }

      // 将临时队列中的事件放回优先队列
      pending_events_.swap(temp_queue);
    }

    // 让出当前线程的时间片，避免过度消耗 CPU，同时提高响应及时性
    std::this_thread::yield();
  }
}

void LayerProgressTracker::ResetState() {
  if (!kEnableTracker) {
    return;
  }

  while (true) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pending_events_.empty()) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(1));
  }
  std::lock_guard<std::mutex> lock(mutex_);

  // 清除所有设备的层进度记录
  device_layer_progress_.clear();

  // 重置所有事件的处理状态为已处理（不需要监控）
  for (auto& pair : events_map_) {
    pair.second.processed = true;
  }
}

}  // namespace ksana_llm